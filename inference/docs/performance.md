# Performance

**English** | [中文](performance_zh.md)

## Measured results

The results below come from separate runs on two machines with different inputs. All used
BF16, PyTorch SDPA, 8 steps, 1280×704 and 24 fps. The DiT stays resident on the GPU; the VAE
uses the layerwise offload chosen automatically at runtime. SageAttention, Cache-DiT,
`torch.compile` and packed QKV were off. The service had finished its built-in warm-up;
"first request" is the first real request after start-up and does not include model loading.

### One GPU

One NVIDIA H20, SP1, seed 42. The input was one three-reference case (a person, a dog and a
building) from OpenS2V-Eval with a rewritten prompt; the case's original photos were sent as
the references. This is a functional check, not an official OpenS2V-Eval result, and a single
case is not a systematic quality evaluation.

| Request | Frames | API inference time | Client wait until completed | Peak reserved memory |
| --- | ---: | ---: | ---: | ---: |
| Three-reference generation, first request | 121 | 100.60 s | 103.42 s | 80,472 MiB |
| Three-reference generation, repeated request | 121 | 98.80 s | 100.86 s | 80,436 MiB |
| Continuation: 17-frame prefix + 120 new frames | 137 | 45.23 s | 48.25 s | 76,422 MiB |

Stage timings:

| Stage | Three-reference generation (first) | Continuation |
| --- | ---: | ---: |
| Text encoding | 0.234 s | 0.239 s |
| Reference / prefix VAE encoding | 0.975 s | 0.756 s |
| DiT, 8 steps | 95.652 s | 40.048 s |
| Audio and video decoding | 3.674 s | 4.112 s |

The continuation used the tail of the generated video and no reference images, so its token
count differs from the generation request: the 137-frame clip has 15,840 tokens, 2,640 of them
for the 17-frame prefix (3 latent frames). The time difference between the two rows is not a
speed-up of the same task; see [Reference image size](#reference-image-size).

### Four GPUs

Four NVIDIA H20 GPUs of the 141 GB (HBM3e) variant with `configs/4gpu-ulysses.json`
(Ulysses SP4), seed 42. This run used a different machine and a different case from the
one-GPU run above: the acceptance run (`scripts/acceptance.py`) on one synthetic case with
three small reference images (768×768, 768×768 and 1280×704 pixels). The "One H20" column is
the same case on one GPU of the same machine.

| Request | Frames | One H20 | Four H20s | Peak reserved memory per GPU (four H20s) |
| --- | ---: | ---: | ---: | ---: |
| Generation, first request | 121 | 44.8 s | 16.1 s | 78,852 MiB |
| Generation, repeated request | 121 | 43.7 s | 15.4 s | 79,766 MiB |
| Continuation, 17 + 120 frames | 137 | 44.8 s | 18.8 s | 80,866 MiB |

Times are API inference times. The outputs are not bit-identical to the single-GPU ones
(the parallel computation is ordered differently; 34–37 dB PSNR against them) and show the
same content.

### Reference image size

Reference images are not resized to the output resolution. Each one is encoded at about its
own size, rounded to multiples of 32 pixels with its aspect ratio kept, and adds
(width / 32) × (height / 32) tokens to the sequence that every DiT step processes, on top of
the 14,080 tokens of a 121-frame clip at 1280×704. A 768×768 image adds 576 tokens, a
1280×704 image 880, a 2048×2048 image 4,096 and a 4000×3000 photo 11,750.

Measured on four 141 GB H20s, 5-second clips, seed 7 (and seeds 1–3 where the run count is
above one), with real references from an agent world: characters drawn at 1328×1328 on a
plain studio backdrop and a 1664×928 place, scaled to each size. A: two characters and the
place; B: three characters and the place.

| References | A: reference tokens | A: inference time | B: reference tokens | B: inference time |
| --- | ---: | ---: | ---: | ---: |
| Scaled to 512×512 | 764 | 14.8 s | 1,020 | 14.8 s (4 runs) |
| Scaled to 768×768 | 1,728 | 15.4 s (4 runs) | 2,304 | 16.9 s |
| Scaled to 1024×1024 | 3,080 | 16.9 s | 4,104 | 17.9 s |
| Original size, backdrop cropped | 3,230 | 17.0 s (3 runs) | 3,986 | 17.8 s (4 runs) |
| Original size (about 1.8 MP each) | 5,036 | 18.5 s (4 runs) | 6,800 | 20.1 s (4 runs) |
| Scaled to 2048×2048 | 12,320 | 26.5 s (3 runs) | 16,416 | 31.4 s |
| One character as a 12 MP photo | 15,153 | 29.9 s | | |
| All three as 12 MP photos | 35,507 | 58.5 s | | |

Repeated runs of one request differed by less than 0.5 s. Encoding the references takes
0.1–1.4 s; the rest of the difference is in the DiT steps, and it grows faster than the token
count. Peak reserved memory stayed at 84,098–88,742 MiB in all 40 runs, so at 5 seconds the
reference size does not change memory noticeably.

In the frames, references larger than the original drawings (2048×2048, 12 MP) did not look
more like the characters, and at 768×768 the woman's face looked less like her reference.
Cropping the plain backdrop around a drawn character kept the look of the original. With
three characters, about half the clips at every size lost a character or merged two, so that
case is unreliable regardless of size. A few seeds per size are not a quality evaluation.

Keep references at about the size the image model draws (around 1.8 megapixels) and crop
plain backdrops; [Murea Agent](../../agent/README.md#clip-length) does both before it sends
them.

The reference size is the likely reason the two one-GPU runs differ. The continuation, which has no
references, took about 45 s in both. The synthetic references add 2,032 tokens, so that
generation (16,112 tokens) took about as long as the 15,840-token continuation. In the
OpenS2V run, each DiT step of the generation took about 11.8 s after the first, against
4.8 s for the continuation on the same machine, which indicates that the three photos added
far more tokens. Their pixel sizes were not recorded with that run, and machine and
references were not varied separately, so the cause was not isolated by measurement.

### Clip length

`num_frames` sets the length: any `8n+1` up to 481 frames (20 seconds), the model's
positional range. A continuation's prefix counts toward it. Four 141 GB H20s, two real-photo
references, one prompt, seed 7:

| Clip | Frames | API inference time | Client wait | Peak reserved memory per GPU |
| --- | ---: | ---: | ---: | ---: |
| 5 s | 121 | 17.0 s | 19.6 s | 81,486 MiB |
| 10 s | 241 | 34.7 s | 38.9 s | 92,260 MiB |
| 15 s | 361 | 56.1 s | 62.2 s | 103,396 MiB |
| 20 s | 481 | 80.8 s | 88.6 s | 114,206 MiB |
| Continuation, 17-frame prefix + 10 s | 257 | 37.0 s | 42.0 s | 93,238 MiB |

Every output was decoded in full with its audio track as long as the video. Frames sampled
across each clip keep both people and the scene consistent up to 20 seconds, and the
continuation picks up from the last frame of its parent. One prompt is not a systematic
quality evaluation. Inference time grows a little faster than linearly with length: from
3.4 seconds of inference time on four GPUs per second of video for the 5-second clip to 4.0
seconds for the 20-second clip.

Peak reserved memory counts PyTorch's allocations only; the CUDA context and communication
buffers come on top. Only the 5-second clip stayed below 96 GB (96 × 10⁹ bytes, about
91,553 MiB) per GPU. From 10 seconds on, including the continuation, each GPU reserved more
than that (92,260 MiB, about 96.7 GB, for 10 seconds), so these lengths have not been shown
to fit on the 96 GB H20.

On one H20 only 5-second clips were measured; they reserved up to 80,472 MiB (about 79 GiB),
so use several GPUs for longer clips.

Software: Torch 2.13.0, Transformers 5.12.1, Diffusers 0.37.0, Triton 3.7.1,
sglang-kernel 0.4.6.post1 (full list in [Deployment](deployment.md#environment)).

**Not yet validated with this checkpoint**: clips longer than 5 seconds on one GPU, clips of
10 seconds or longer on GPUs with less than 141 GB, two or eight GPUs, TP, SageAttention,
packed QKV, `torch.compile`, GPUs other than the H20, long-running concurrent load, audio
continuity across continuation seams, and multi-case quality. The configurations for these
remain in the repository, but no quality or speed gains are claimed for them.

## Optimization switches

The baseline is BF16, SDPA, 8 steps, one-stage. Compare any change against it with the same
checkpoint, inputs, seed, resolution, duration, parallelism and software versions.

| Mechanism | Status | Configuration and limits |
| --- | --- | --- |
| Resident service | Included | One model load serves many requests; per-stage offload is decided by the underlying component manager |
| Ulysses sequence parallelism | Included | `parallel: ulysses` on 1/2/4/8 GPUs; splits the sequence, not the layers |
| Tensor parallelism | Upstream path | `parallel: tp`; not the default, measure it separately |
| Packed QKV all-to-all | Optional | `packed_qkv: true`; sends Q/K/V in one collective instead of three. Only data movement changes, and ineligible inputs fall back to the unpacked path. It is not low-precision communication |
| SageAttention | Optional, approximate | `attention: sage_attn` plus `sage_pythonpath`; results are not pixel-identical to SDPA |
| `torch.compile` | Optional, experimental | `compile: true`; exclude the first compilation from warm timings and validate separately |
| Cache-DiT / step skipping | Off | With only 8 steps, no further steps are skipped |
| KV cache across requests | Not used | Bidirectional attention and changing timesteps prevent reuse |
| Clean-latent continuation | Implemented | Fixed prefix, only the new region is denoised; audio is regenerated; not chunked streaming |

## Measuring

Each request records its input, `timing.json`, the optional `perf_dump_path` output and an
FFprobe report. `service.json` in the runtime directory records the launch command and the
SHA-256 of every running source file, so results can be traced back to the exact code.

`inference_time_s` (server) and the client wait time are recorded separately.
`peak_memory_mb` is PyTorch's peak reserved memory divided by 1024², so the unit is MiB;
with several GPUs it is not the sum across GPUs.

For a benchmark, warm the service up first, then run the same case and seed several times.
`scripts/acceptance.py` runs a generation twice and a continuation, and validates each
output; see [Deployment](deployment.md#acceptance-run).
