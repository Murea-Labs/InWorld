from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from .checkpoint import CHECKPOINT_SHA256, verify
from .client import write_json

# The inference/ checkout. The launcher needs its scripts/ and vendor/ trees, which are not
# part of the wheel, so it only works from the checkout or an editable install.
ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ("python", "model_dir", "checkpoint", "runtime_dir", "gpus", "parallel", "attention",
            "port", "scheduler_port", "master_port")
WILDCARD_HOSTS = {"": "127.0.0.1", "0.0.0.0": "127.0.0.1", "::": "::1"}


def require_checkout() -> Path:
    if not ((ROOT / "vendor/sglang/python/sglang").is_dir()
            and (ROOT / "scripts/materialize_connectors.py").is_file()):
        raise RuntimeError(
            f"The Murea-Fast launcher must run from the source checkout, but {ROOT} has no "
            "vendor/sglang/ or scripts/. Run scripts/murea-fast from the inference/ directory, "
            "or install the package with 'pip install -e .' from there.")
    return ROOT


def source_manifest():
    paths = sorted((ROOT / "src/murea_fast").glob("*.py"))
    paths += [ROOT / "scripts/materialize_connectors.py"]
    paths += sorted((ROOT / "vendor/sglang/python/sglang/multimodal_gen").rglob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def config(path: Path) -> dict:
    require_checkout()
    c = json.loads(path.read_text())
    missing = [key for key in REQUIRED if key not in c]
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")
    c["config_path"] = str(path.resolve())
    c.setdefault("host", "127.0.0.1")
    if type(c["host"]) is not str:
        raise ValueError("host must be a string, for example \"127.0.0.1\"")
    for key in ("packed_qkv", "compile", "allow_server_paths"):
        if type(c.get(key, False)) is not bool:
            raise ValueError(f"{key} must be true or false")
    for key in ("python", "model_dir", "checkpoint", "runtime_dir"):
        if not Path(c[key]).is_absolute():
            raise ValueError(f"{key} must be absolute")
    if type(c["gpus"]) is not str:
        raise ValueError("gpus must be a string of comma-separated GPU indices, "
                         "for example \"0\" or \"0,1,2,3\"")
    ids = c["gpus"].split(",")
    if not ids or len(ids) != len(set(ids)) or any(not s.isdigit() for s in ids):
        raise ValueError("gpus must be unique numeric IDs")
    if len(ids) not in (1, 2, 4, 8):
        raise ValueError("Supported GPU counts: 1,2,4,8")
    if c["parallel"] not in ("ulysses", "tp"):
        raise ValueError("parallel must be ulysses or tp")
    if c["attention"] not in ("sdpa", "sage_attn"):
        raise ValueError("attention must be sdpa or sage_attn")
    ports = [c[k] for k in ("port", "scheduler_port", "master_port")]
    if len(set(ports)) != 3 or any(type(p) is not int or not 1024 <= p <= 65535 for p in ports):
        raise ValueError("Use three distinct unprivileged ports")
    return c


def runtime(c):
    return Path(c["runtime_dir"]) / f"{socket.gethostname()}-{c['port']}"


def health_url(c) -> str:
    """URL for /health from this host: loopback when the service listens on all interfaces."""
    host = WILDCARD_HOSTS.get(c["host"], c["host"])
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{c['port']}/health"


def prepare(c):
    record = verify(Path(c["checkpoint"]))
    connectors = Path(c["runtime_dir"]) / "connectors-murea-fast"
    subprocess.run([
        c["python"], str(ROOT / "scripts/materialize_connectors.py"),
        "--checkpoint", c["checkpoint"], "--base-dir", str(Path(c["model_dir"]) / "connectors"),
        "--output-dir", str(connectors), "--verify-existing-hashes",
    ], check=True)
    record["connector_manifest"] = str(connectors / "connector-overlay-manifest.json")
    manifest = json.loads(Path(record["connector_manifest"]).read_text())
    record["connector_checkpoint_sha256_verified"] = manifest["checkpoint_sha256"] == CHECKPOINT_SHA256
    return record


def command(c):
    n = len(c["gpus"].split(","))
    return [c["python"], "-c", "from sglang.cli.main import main; main()", "serve",
            "--model-path", c["model_dir"], "--model-id", "LTX-2.5-Diffusers",
            "--served-model-name", "Murea-Fast",
            "--model-type", "diffusion", "--pipeline-class-name", "LTX2Pipeline",
            "--transformer-weights-path", c["checkpoint"],
            "--component-paths.connectors", str(Path(c["runtime_dir"]) / "connectors-murea-fast"),
            "--num-gpus", str(n), "--base-gpu-id", "0",
            "--ulysses-degree" if c["parallel"] == "ulysses" else "--tp-size", str(n),
            "--attention-backend", "torch_sdpa" if c["attention"] == "sdpa" else c["attention"],
            "--enable-torch-compile", str(c.get("compile", False)).lower(),
            "--host", c["host"], "--port", str(c["port"]),
            "--scheduler-port", str(c["scheduler_port"]), "--master-port", str(c["master_port"]),
            "--strict-ports", "--output-path", str(runtime(c) / "outputs")]


def environment(c):
    env = os.environ.copy()
    # Keep this service independent of prior shell experiments.
    for key in list(env):
        if key.startswith(("SGLANG_CACHE_DIT", "SGLANG_LTX_SAGE", "SGLANG_LTX2_REFERENCE")):
            env.pop(key)
    cache = runtime(c) / "cache"
    for sub in ("torch", "triton", "inductor", "tmp", "sglang"):
        (cache / sub).mkdir(parents=True, exist_ok=True)
    paths = [str(ROOT / "vendor/sglang/python"), str(ROOT / "src")]
    if c["attention"] == "sage_attn":
        sage = Path(c.get("sage_pythonpath", ""))
        if not (sage / "sageattention").is_dir():
            raise ValueError("Set sage_pythonpath to the tested SageAttention build")
        paths.insert(0, str(sage))
    env.update({"PYTHONPATH": os.pathsep.join(paths), "PYTHONNOUSERSITE": "1",
                # Number GPUs as nvidia-smi does, so `gpus` means what check_resources checked.
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": c["gpus"],
                "PATH": str(Path(c["python"]).parent) + os.pathsep + env["PATH"],
                "TORCH_HOME": str(cache / "torch"), "TRITON_CACHE_DIR": str(cache / "triton"),
                "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"), "TMPDIR": str(cache / "tmp"),
                "SGLANG_DIFFUSION_CACHE_ROOT": str(cache / "sglang"),
                "SGLANG_DIFFUSION_STAGE_LOGGING": "1",
                "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING": "1", "SGLANG_CACHE_DIT_ENABLED": "false",
                "MUREA_FAST_PACKED_QKV": str(c.get("packed_qkv", False)).lower(),
                # Off unless the configuration says so: requests may then name files on
                # the server to read (video_path, reference_image_paths) and to write
                # (output_path, perf_dump_path).
                "MUREA_FAST_ALLOW_SERVER_PATHS": str(c.get("allow_server_paths", False)).lower()})
    return env


def proc_identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return fields[19]  # field 22 starttime; robust to spaces in comm
    except FileNotFoundError:
        return None


def state(c):
    path = runtime(c) / "service.json"
    if not path.exists():
        return {"running": False}
    data = json.loads(path.read_text())
    data["running"] = (data["host"] == socket.gethostname()
                       and proc_identity(data["pid"]) == data["starttime"])
    return data


def check_resources(c):
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"
    ], text=True)
    devices = {v[0].strip(): (v[1].strip(), int(v[2]))
               for v in (line.split(",") for line in rows.strip().splitlines())}
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
    ], text=True)
    active = {line.split(",")[0].strip() for line in apps.splitlines()}
    for gpu in c["gpus"].split(","):
        if gpu not in devices or devices[gpu][1] > 1024 or devices[gpu][0] in active:
            raise RuntimeError(f"GPU {gpu} is unavailable; no existing task will be stopped")
    for key in ("port", "scheduler_port", "master_port"):
        with socket.socket() as sock:
            sock.bind(("0.0.0.0", c[key]))


def start(c):
    slot = runtime(c)
    slot.mkdir(parents=True, exist_ok=True)
    with (slot / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if state(c)["running"]:
            raise RuntimeError("Service is already running")
        prepared = prepare(c)
        check_resources(c)
        env = environment(c)
        cmd = command(c)
        sources = source_manifest()
        with (slot / "server.log").open("ab") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        starttime = proc_identity(proc.pid)
        if starttime is None:
            raise RuntimeError("Server exited immediately; inspect server.log")
        record = {"pid": proc.pid, "starttime": starttime, "host": socket.gethostname(),
                  "root": str(ROOT), "config": c, "command": cmd,
                  "checkpoint": prepared, "checkpoint_sha256": CHECKPOINT_SHA256,
                  "source_sha256": sources,
                  "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        write_json(slot / "service.json", record)
        print(json.dumps({"pid": proc.pid, "log": str(slot / "server.log"),
                          "status": "starting; poll /health before inference"}))


def stop(c):
    slot = runtime(c)
    slot.mkdir(parents=True, exist_ok=True)
    with (slot / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = state(c)
        if not data["running"]:
            print("Service is not running")
            return
        pid = data["pid"]
        if (data["root"] != str(ROOT) or os.getpgid(pid) != pid or
                Path(f"/proc/{pid}/cwd").resolve() != ROOT):
            raise RuntimeError("Process identity mismatch; refusing to stop")
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        if b"sglang.cli.main" not in cmdline:
            raise RuntimeError("Process command mismatch; refusing to stop")
        os.killpg(pid, signal.SIGTERM)
        if wait_for_group(pid):
            print("Service stopped")
            return
        raise TimeoutError("SIGTERM sent; processes of the service are still running, "
                           "inspect logs (no SIGKILL sent)")


def group_alive(pgid) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # a member exists but belongs to someone else
    return True


def wait_for_group(pgid, timeout=60) -> bool:
    """Wait until every process of the group is gone, not only its leader: the GPU
    workers can outlive the server process and still hold memory and ports."""
    deadline = time.monotonic() + timeout
    while group_alive(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(1)
    return True
