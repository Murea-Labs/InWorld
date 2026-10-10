# Murea-Fast

**English** | [中文](README_zh.md)

> **Status:** the Murea-Fast weights are not published yet. They will be released on Hugging
> Face (see [Model](#model)). Until then the service cannot be started; the code, the
> documentation and the standard-library tests are available.

Murea-Fast is the inference service of [InWorld](../README.md). It generates videos with
synchronized audio from a text prompt and an ordered list of reference images, and can
continue an existing video from its last frames. The model runs behind an HTTP API.

This directory is self-contained: it ships a pinned SGLang snapshot with the model
implementation (`vendor/sglang/`), a small launcher and client (`src/murea_fast/`), and
ready-made configurations. You still need the model weights, the LTX-2.5 base components,
and a CUDA environment.

## Features

- **Multi-reference generation**: reference images are bound in request order, for
  example a person, an object and a scene, and referred to from the prompt.
- **Clean-latent continuation**: the last 17 frames of a video are encoded as a fixed
  latent prefix and the rest of the clip is generated after it.
- **8-step distilled sampling**: Euler, CFG = 1, 24 fps.
- **Multi-GPU**: Ulysses sequence parallelism or tensor parallelism on 1, 2, 4 or 8 GPUs;
  optional SageAttention, packed-QKV all-to-all and `torch.compile`.
- **Traceable runs**: every request records its input, timings, peak GPU memory and an
  FFprobe report, and the output is fully decoded to check frame count and audio track.

## Model

| Item | Value |
| --- | --- |
| Checkpoint | `model_ema_distilled8_step_07000.safetensors` |
| Size | 42,018,192,440 bytes |
| SHA-256 | `1a2c0d8e23032994b52f2c67e14c930d3be8224f321c51353f0a2954ccbb9385` |
| Base model | Fine-tuned and distilled from [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) |
| Download | Will be published on Hugging Face; the link will be added here |

The checkpoint holds the DiT and 258 trained text-connector tensors. `prepare` merges those
connectors into the base components: the 258 tensors replace their base counterparts, and the
four text-projection tensors that exist only in the base components (`audio_text_proj_in`
and `video_text_proj_in`, weight and bias) are kept. The checkpoint therefore cannot be used
as a drop-in DiT replacement on its own. The launcher only accepts this exact checkpoint.

## Requirements

- Linux, NVIDIA GPU with CUDA, Python 3.10+, FFmpeg and ffprobe.
- Verified at 1280×704: one NVIDIA H20 for 5-second clips (44–101 s per clip depending on
  the size of the reference images, peak 80,472 MiB reserved GPU memory), and four 141 GB
  H20s (Ulysses SP4) for clips of 5–20 seconds (peak 81,486–114,206 MiB, about 80–112 GiB,
  per GPU). On four GPUs, clips of 10 seconds and longer reserved more than 96 GB per GPU, so
  they have not been shown to fit on the 96 GB H20. Clips can be up to 20 seconds long; see
  [Clip length](docs/performance.md#clip-length). Smaller GPUs, longer clips on one GPU, and
  other multi-GPU configurations have not been validated with this checkpoint yet.

## Quick start

1. Install the environment and download the LTX-2.5 base components, as described in
   [Deployment](docs/deployment.md).
2. Download the Murea-Fast checkpoint and verify it:

   ```bash
   bash scripts/murea-fast verify /path/to/model_ema_distilled8_step_07000.safetensors --sha256
   ```

3. Edit the four paths in `configs/single-gpu.json` (Python, base components, checkpoint,
   runtime directory) and choose a free GPU in `gpus`.
4. Prepare and start the service, then wait for the health check:

   ```bash
   bash scripts/murea-fast prepare --config configs/single-gpu.json
   bash scripts/murea-fast start --config configs/single-gpu.json
   curl --fail http://127.0.0.1:30260/health
   ```

   `start` returns once the process is launched. The service is ready only when
   `/health` succeeds; until then the model may still be loading.
5. Stop the service with the same config on the same host:

   ```bash
   bash scripts/murea-fast stop --config configs/single-gpu.json
   ```

## Multi-reference generation

Copy `configs/request.example.json`, set the prompt and reference image paths in order,
then run:

```bash
bash scripts/murea-fast generate --request my-request.json --output outputs/sample
```

Reference paths are read by the server, so they must exist on the service host, and the
service accepts them only when its configuration sets `"allow_server_paths": true` (see
[Server paths](docs/deployment.md#server-paths)); it is off by default. Without it, or from
another machine, send the images as `data:` URLs instead (see
[API](docs/api.md#clients-on-another-machine)). The output
directory contains `video.mp4`, the request, the submit and final responses, `timing.json`
(API inference time, client wait and download time, peak memory) and `ffprobe.json`.
With server paths allowed, set `perf_dump_path` to an absolute server-side path to also
record per-stage and per-step timings.

`generate` and `continue` talk to `http://127.0.0.1:30260` by default. Pass `--base-url` for
another address and `--timeout` to change how long the client waits for a job, in seconds
(default 1800).

## Continuation

Run on the service host, or on a filesystem shared with it. The command passes the trimmed
prefix and `perf_dump_path` as server paths, so the service must run with
`"allow_server_paths": true` ([Server paths](docs/deployment.md#server-paths)):

```bash
bash scripts/murea-fast continue \
  --parent outputs/sample/video.mp4 \
  --prompt 'The person continues walking, turns towards the camera and smiles. Natural footsteps.' \
  --prefix-frames 17 --new-frames 120 --seed 42 \
  --output outputs/continuation
```

The client cuts the **last 17 frames** of the parent video. The service VAE-encodes them as
a clean latent prefix: the prefix is excluded from denoising and restored after every
step. The output has **137 frames: the 17-frame prefix followed by 120 new frames**.

The output directory holds `prefix.mp4` (the trimmed tail sent to the service), `parent.json`
(parent path, resolution and frame counts) and `perf.json` (per-stage timings). The video,
together with its request, responses, `timing.json` and `ffprobe.json`, is in `generated/`,
so the result of the command above is `outputs/continuation/generated/video.mp4`.

Constraints and caveats:

- Prefix and total frame counts must be `8n+1`, and the total must exceed the prefix.
- Continuation cannot be combined with reference images.
- Continuation supports one-stage inference only. The launcher always starts the one-stage
  `LTX2Pipeline`; two-stage pipelines are not supported.
- Audio is regenerated for the whole clip; it does not inherit the parent's audio.
- The VAE-reconstructed prefix is not guaranteed to be pixel-identical to the parent.
- The output repeats the prefix. When joining clips, drop the first 17 frames of each
  continuation.
- This is whole-clip denoising, not chunked streaming generation or KV-cache reuse.

## API

`POST /v1/videos` → `GET /v1/videos/{id}` → `GET /v1/videos/{id}/content`.
Request fields and the sampling contract are listed in [API](docs/api.md).

The service listens on `127.0.0.1` by default and has no authentication. To expose it,
put it behind a reverse proxy that handles authentication, and read
[Security](../SECURITY.md#running-murea-fast-safely). Requests that name server files are
refused unless the configuration allows them, and so are frame interpolation, upscaling and
`diffusers_kwargs` ([Refused fields](docs/api.md#refused-fields)).

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=vendor/sglang/python /path/to/.venv/bin/python tests/smoke_runtime.py
```

The first command (or `pytest`, if it is installed) runs client, launcher and request-field
checks with the standard library only. The second needs the full environment and checks
sampling parameters, HTTP field lowering, frame alignment, Ulysses prefix slicing and the
clean-latent mask. Our tests inside the vendored engine, such as
`test/unit/test_ltx2_ordered_multiref.py` under `vendor/sglang/python/sglang/multimodal_gen/`,
run with `pytest` in the full environment; they also cover the refused request fields and
the input limits.

## Documentation

- [Deployment](docs/deployment.md): installation, base components, configuration
  reference, service lifecycle, troubleshooting.
- [API](docs/api.md): endpoints, request and response fields.
- [Performance](docs/performance.md): optimization switches and measured results.
- [Provenance](docs/provenance.md): upstream sources, modified files, licenses.

## License

- Code in this directory is released under the [Apache License 2.0](../LICENSE). The
  vendored SGLang is Apache-2.0 ([license](vendor/sglang/LICENSE)), apart from third-party
  files that keep their own MIT or BSD-3-Clause notices; files we modified carry a notice at
  the top. See [Provenance](docs/provenance.md) for both.
- The Murea-Fast weights are a derivative of LTX-2.5 and are distributed under the
  [LTX-2.x Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x).
  Among other terms, it requires entities with annual revenue of USD 10 million or more to
  obtain a commercial license from Lightricks, and it includes use restrictions. The LTX-2.5
  base components are covered by the same agreement. Read it before using the weights.
- If you redistribute the weights or a model derived from them, the agreement requires you to
  include a complete copy of it, to pass on its use restrictions (Attachment A) as enforceable
  terms of your own license or agreement, and to tell recipients in writing that entities
  above the revenue threshold need a paid license from Lightricks.
- Murea-Fast is not affiliated with or endorsed by Lightricks.
