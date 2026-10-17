# Deployment

**English** | [中文](deployment_zh.md)

All commands below run from the `inference/` directory.

## Layout

- `src/murea_fast/`: launcher, checkpoint checks, generation and continuation client.
- `vendor/sglang/python/sglang/multimodal_gen/`: model, sampling, VAE, distributed and REST
  implementation.
- `scripts/materialize_connectors.py`: merges the trained connectors into the base
  components and verifies every file.
- `scripts/acceptance.py`: end-to-end check against a running service.
- `configs/`: service configurations and an example request.
- `tests/`: client and launcher checks, plus a runtime smoke test.

The source tree can live anywhere. Paths in the configuration must be absolute paths on the
machine that runs the service. `MUREA_PYTHON` selects the Python used by the
`scripts/murea-fast` wrapper; the service itself uses the `python` from the configuration.

## Environment

The verified environment used these versions: Torch 2.13.0, TorchVision 0.28.0,
TorchAudio 2.11.0, Transformers 5.12.1, Diffusers 0.37.0, Safetensors 0.8.0,
Cache-DiT 1.3.0, Triton 3.7.1, sglang-kernel 0.4.6.post1.

Install into a dedicated virtual environment. CUDA and the driver must match the PyTorch
build, and optional dependencies may need to be built beforehand:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.18.dev634 \
  .venv/bin/python -m pip install './vendor/sglang/python[diffusion]'
.venv/bin/python -m pip install -e .
```

This installs the dependencies declared by the vendored SGLang. A fresh install on a new
machine or GPU architecture has not been validated end to end yet; check the result with
`tests/smoke_runtime.py` and the acceptance run below.

Only an editable install (`pip install -e .`, as above) or running `scripts/murea-fast` from
this directory is supported. The launcher (`prepare`, `start`, `stop`, `status`, `command`)
needs `vendor/` and `scripts/` from the checkout, which a regular `pip install .` does not
install; it then stops with an error saying so. `verify`, `generate` and `continue` work from
any install.

The launcher always puts the bundled `vendor/sglang/python` first on `PYTHONPATH`, so an
SGLang installed elsewhere is not picked up by mistake.

You also need `ffmpeg` and `ffprobe` on `PATH`, or set `MUREA_FFMPEG` and `MUREA_FFPROBE` to
the full executable paths. The client checks for both before submitting a job, so a missing
tool does not surface only after generation has finished.

SageAttention is optional. Install a compatible build separately and set `sage_pythonpath`
in the configuration.

## Base components

The service needs the complete `LTX-2.5-Diffusers` directory: `model_index.json`, tokenizer
and text encoder, connectors, video and audio VAE, and vocoder. The Murea-Fast checkpoint
replaces the DiT. Download the base components at the pinned revision:

```bash
hf auth login   # the repository requires accepting its license on Hugging Face first
hf download Lightricks/LTX-2.5-Diffusers \
  --revision a6de4b5354f078db24d9cf4778c14846788aea3d \
  --local-dir /path/to/LTX-2.5-Diffusers
```

## Checkpoint

Download the Murea-Fast checkpoint (see the [README](../README.md#model)) and verify it:

```bash
bash scripts/murea-fast verify /path/to/model_ema_distilled8_step_07000.safetensors --sha256
```

Without `--sha256`, `verify` checks the file size, the tensor count and the sampling
contract stored in the checkpoint metadata. With `--sha256` it also hashes the whole
42 GB file.

The checkpoint and the base components are covered by the
[LTX-2.x Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x).
If you pass them or a model derived from them on to others, for example by publishing a
mirror or a container image with them, include a complete copy of the agreement, make its use
restrictions (Attachment A) enforceable terms of your own license or agreement, and tell
recipients in writing that entities above its revenue threshold need a paid license from
Lightricks. See [License](../README.md#license).

## Configuration

Start from one of the files in `configs/`:

| File | Setup | Validated with this checkpoint |
| --- | --- | --- |
| `single-gpu.json` | 1 GPU, SDPA | Yes |
| `4gpu-ulysses.json` | 4 GPUs, Ulysses SP4 | Yes, on 141 GB H20s, including clips up to 20 s ([Performance](performance.md#four-gpus)) |
| `4gpu-ulysses-packed-qkv.json` | 4 GPUs, Ulysses SP4, packed QKV, GPUs 4-7 and separate ports | Not yet |

| Field | Meaning |
| --- | --- |
| `python` | Absolute path of the Python that runs the service |
| `model_dir` | Absolute path of the `LTX-2.5-Diffusers` directory |
| `checkpoint` | Absolute path of the Murea-Fast checkpoint |
| `runtime_dir` | Absolute path for logs, caches, outputs and the merged connectors |
| `gpus` | String of comma-separated GPU indices as `nvidia-smi` numbers them, for example `"0"` or `"0,1,2,3"`; 1, 2, 4 or 8 unique GPUs. The launcher sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` so that CUDA uses the same numbering |
| `parallel` | `ulysses` (sequence parallel) or `tp` (tensor parallel) |
| `attention` | `sdpa` or `sage_attn` |
| `sage_pythonpath` | Directory containing the `sageattention` package; required for `sage_attn` |
| `packed_qkv` | Exchange Q/K/V in one all-to-all under Ulysses; default `false` |
| `compile` | Enable `torch.compile`; default `false` |
| `host` | Listen address; default `127.0.0.1`. With `0.0.0.0` or `::`, `status` reports a loopback health URL |
| `allow_server_paths` | Let requests name files on the server; default `false`. See [Server paths](#server-paths) |
| `port`, `scheduler_port`, `master_port` | Three distinct ports between 1024 and 65535 |

Fields without a default are required, except `sage_pythonpath`, which only `sage_attn`
needs; the launcher names any missing field.

Choosing a GPU count does not make a configuration fit every memory size; see
[Performance](performance.md) for measured memory use.

### Server paths

By default the service accepts images as `data:` or `http(s)` URLs and the continuation
prefix as a multipart upload, which is what Murea-Mind sends. It refuses with HTTP 400 any
request that names a file on the server: `output_path`, `perf_dump_path`, and a local path in
`video_path`, `input_reference` or `reference_image_paths`.

`"allow_server_paths": true` accepts them. Then every client that can reach the service can
make it read any image or video and write files wherever the service account can, so turn
it on only when all clients are trusted, for example on a host used by one person. The
shipped configurations leave it off. `murea-fast continue`, `scripts/acceptance.py` and the
example request `configs/request.example.json` pass server paths and need it; against a
service without it they stop with an error that names the setting.

Frame interpolation, upscaling and `diffusers_kwargs` are refused whatever this setting
says: Murea-Fast does not use them, and their model fields would make the server download
and load other models.

## Service lifecycle

```bash
bash scripts/murea-fast prepare --config configs/single-gpu.json
bash scripts/murea-fast start   --config configs/single-gpu.json
bash scripts/murea-fast status  --config configs/single-gpu.json
bash scripts/murea-fast stop    --config configs/single-gpu.json
bash scripts/murea-fast command --config configs/single-gpu.json   # print the launch command
```

- **prepare** verifies the checkpoint metadata and size. The first time, it computes the
  full SHA-256 and builds the merged connectors in `runtime_dir/connectors-murea-fast/`.
  Later runs reuse that directory after checking the hash of every file in it. If the
  checkpoint's size, timestamps or inode change, the full SHA-256 is recomputed. An
  incomplete or mismatching connector directory is never overwritten silently: remove it and
  run `prepare` again.
- **start** runs `prepare`, then refuses to launch if any selected GPU already runs a
  process or uses more than 1 GiB of memory, or if a port is taken. It never stops other
  jobs. It returns as soon as the process starts; the service is ready once
  `GET /health` succeeds.
- **status** reports whether the registered process is still running.
- **stop** sends SIGTERM to the process group it started, after checking the recorded
  process ID, start time, working directory and command, and reports the service stopped
  once every process of the group has exited (the GPU workers can outlive the server
  process). If any is still running after 60 s it says so. It does not send SIGKILL and does
  not touch any other process. Run it on the same host with the same configuration.

State is kept per host and port in `runtime_dir/<hostname>-<port>/`: `server.log`,
`service.json` (launch command, checkpoint record and SHA-256 of the running source files),
caches and outputs.

When moving to another machine or cluster, copy and verify the checkpoint and every base
component explicitly.

## Acceptance run

With the service healthy, run an end-to-end check on the service host. The script passes
reference images and the continuation prefix as server paths and asks the service to write
timing files, so the service must run with `"allow_server_paths": true`
(see [Server paths](#server-paths)):

```bash
.venv/bin/python scripts/acceptance.py --case /path/to/case --output /path/to/new-run-dir
```

The case directory holds `prompt.txt`, ordered reference images `ref_00.jpg`,
`ref_01.png`, ..., and optionally `continuation_prompt.txt`. The script generates the case
twice, continues the second video, validates every output and writes `summary.json`.
`--resume` re-validates finished outputs and runs only the missing steps.

## Troubleshooting

- **The process runs but `/health` fails**: check `runtime_dir/<hostname>-<port>/server.log`;
  the model may still be loading.
- **File not found in a request**: request paths are read by the server. A path on your
  laptop is not visible to a remote service; send the file as a `data:` URL or an upload.
- **400 naming `allow_server_paths`**: the request names a file on the server; see
  [Server paths](#server-paths).
- **Out of memory**: lower the resolution or frame count, or use more GPUs.
- **GPU or port busy**: pick free GPUs and three unused ports. The launcher does not stop
  other jobs.
- **Frame counts**: 24 fps, `8n+1` frames: 121 frames is 5 seconds, 241 is 10, at most 481
  (20 seconds). A 17-frame prefix plus 120 new frames gives 137 output frames.
- **Continuation audio**: it is regenerated. If audio continuity matters, handle the audio
  track separately and check the result.
- **Stopping**: use `stop` with the same host and configuration, not `pkill python`.
