# API

**English** | [中文](api_zh.md)

The service exposes an asynchronous video job API on `http://<host>:<port>`
(default `http://127.0.0.1:30260`). It has no authentication; put an authenticating reverse
proxy in front of it before exposing it to other machines, and read
[Security](../../SECURITY.md#running-murea-fast-safely) first.

## Endpoints

| Method and path | Purpose |
| --- | --- |
| `GET /health` | 200 once the model is loaded and warm-up has finished; 503 before that |
| `GET /v1/models` | The served model; `pipeline_class` is `LTX2Pipeline` for Murea-Fast |
| `POST /v1/videos` | Submit a generation or continuation job; returns the job with its `id` |
| `GET /v1/videos/{id}` | Poll a job |
| `GET /v1/videos/{id}/content` | Download the finished MP4; 404 while the job is still running |
| `GET /v1/videos` | List jobs |
| `DELETE /v1/videos/{id}` | Remove a job from the job list; the response has `status: "deleted"`. A running job is not stopped, and the output file stays on disk |

The SGLang endpoints that load other weights or LoRA adapters, release GPU memory or run RL
rollouts (`/update_weights_from_*`, `/release_memory_occupation`, `/v1/set_lora`, `/rollout/...`
and the like) are not mounted.

A job starts as `queued` and ends as `completed` or `failed`. On failure, `error.message`
holds the reason. The job list is kept in memory and is lost when the service restarts.

## Example

```bash
curl -s http://127.0.0.1:30260/v1/videos \
  -H 'Content-Type: application/json' -d @request.json
# {"id": "0c2dc7e2-...", "status": "queued", ...}

curl -s http://127.0.0.1:30260/v1/videos/0c2dc7e2-...
# poll until "status" is "completed" or "failed"

curl -s -o video.mp4 http://127.0.0.1:30260/v1/videos/0c2dc7e2-.../content
```

From Python, `murea_fast.client.generate(base_url, payload, output_dir)` submits a job,
polls it, downloads the video and validates it. It is what `scripts/murea-fast generate`
uses. [Murea Agent](../../agent/README.md) calls the same API from
`agent/murea_agent/providers/video/murea.py`. It always sends requests the way a
[client on another machine](#clients-on-another-machine) does, so the service may run on the
same machine, on another one or behind an SSH tunnel.

## Request fields

Send a JSON body with `Content-Type: application/json`.

| Field | Type | Notes |
| --- | --- | --- |
| `prompt` | string | Required |
| `reference_image_paths` | list of strings | Reference images in order. Each is a `data:` URL, an `http(s)` URL that the server downloads (64 MiB at most), or, only when the service allows [server paths](deployment.md#server-paths), an absolute path on the server. The acceptance runs used three. One end-to-end test also generated a 5-second clip from five (three player characters, a monster and a place) without errors; its quality was not evaluated. More than five has not been tested. Large images make every step slower; about 1.8 megapixels each is enough ([Performance](performance.md#reference-image-size)) |
| `video_path` | string | Continuation only: the prefix clip, as an `http(s)` URL (2 GiB at most, downloaded within 300 s) or, with [server paths](deployment.md#server-paths) allowed, an absolute server path. Without them, upload the clip ([Clients on another machine](#clients-on-another-machine)) |
| `continuation_prefix_frames` | integer | Continuation only: number of prefix frames taken from the start of `video_path`. A JSON integer: `17.0` or `"17"` is refused |
| `width`, `height` | integer | Positive multiples of 32; validated at 1280×704 |
| `num_frames` | integer | `8n+1`, from 9 to 481 (20 seconds); 121 frames is 5 seconds, 241 is 10. A continuation's prefix counts toward it |
| `fps` | integer | Must be 24 |
| `num_inference_steps` | integer | Must be 8 |
| `guidance_scale` | number | Must be 1.0 |
| `seed` | integer | Random seed |
| `generator_device` | string | `cuda` |
| `perf_dump_path` | string | Optional absolute server path; writes per-stage and per-step timings. Only with [server paths](deployment.md#server-paths) allowed |

`fps`, `num_inference_steps` and `guidance_scale` are fixed by the distilled checkpoint. The
bundled client fills them in and rejects other values; when calling the API directly, send
exactly these values.

Continuation rules: `continuation_prefix_frames` and `num_frames` are both `8n+1`, the prefix
is shorter than `num_frames`, and `video_path` cannot be combined with
`reference_image_paths`. The service takes the prefix from the **start** of `video_path`, so
trim the parent video to its last frames first. The `continue` command of
`scripts/murea-fast` does this for you.

### Clients on another machine

A remote client cannot pass its own file paths. This includes a client that reaches the
service through an SSH tunnel at `127.0.0.1`. This is also the only way to call a service
that does not allow [server paths](deployment.md#server-paths), the default. Send reference images as `data:` URLs in
`reference_image_paths`. For continuation, send the request as `multipart/form-data`, with
the prefix clip as the file `video_reference` and the other fields as form fields. A
multipart request reads fewer fields than a JSON one:

- Give the resolution as `size`, for example `1280x704`. `width` and `height` are ignored,
  and the service would fall back to its default resolution.
- `continuation_prefix_frames` is not read. The uploaded clip is the prefix, rounded down to
  `8n+1` frames and shorter than `num_frames`, so upload exactly the frames you want as the
  prefix, for example the last 17 frames of the parent video.
- `perf_dump_path` is not read.

The service keeps uploaded clips in `inputs/uploads/` under its working directory (the
`inference/` directory when started with `scripts/murea-fast`) and does not delete them.

### Refused fields

The underlying SGLang server has further fields that Murea-Fast does not use. Requests that
set `enable_frame_interpolation`, `frame_interpolation_exp`, `frame_interpolation_scale`,
`frame_interpolation_model_path`, `enable_upscaling`, `upscaling_model_path`,
`upscaling_scale` or `diffusers_kwargs` (to anything but `null` or `false`) are refused with
HTTP 400. So is `output_path` unless the service allows
[server paths](deployment.md#server-paths); outputs then go to the service's runtime
directory.

## Prompt format

Our validation used prompts that first describe each reference and then the shot, referring
to references by their position in `reference_image_paths`:

```text
<Subject 1> is a single close-up view of a woman with long brown hair, wearing a light blue sweater, as shown in <Picture 1>.
<Subject 2> is a single white ceramic coffee cup on a plain background, as shown in <Picture 2>.

<Shot 1> at 0.00s, A woman, <Subject 1> from <Picture 1>, sits at a wooden table by a window in soft daylight. She picks up a white cup, <Subject 2> from <Picture 2>, smiles and takes a sip. The camera holds a steady medium shot. Quiet room ambience.
```

## Response fields

| Field | Notes |
| --- | --- |
| `id` | Job ID |
| `status` | `queued`, `completed` or `failed`; `deleted` in the response to `DELETE` |
| `error` | `{"message": ...}` when the job failed |
| `inference_time_s` | Server-side inference time in seconds |
| `peak_memory_mb` | Peak PyTorch reserved GPU memory in MiB (bytes / 1024²); with several GPUs this is not the sum across GPUs |
| `file_path` | Server-side path of the output video |
| `created_at`, `completed_at` | Unix timestamps |
