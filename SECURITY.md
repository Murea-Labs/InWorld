# Security

**English** | [中文](SECURITY_zh.md)

## Reporting a vulnerability

Please do not open a public issue for security problems. Report them through
GitHub's private vulnerability reporting: on this repository, open the
**Security** tab and choose **Report a vulnerability**. Only the maintainers see
the report. Include what you found, how to reproduce it and what an attacker
could do with it. We will acknowledge the report within a week.

## Running Murea-Fast safely

The inference service is meant for a trusted network. It listens on `127.0.0.1` by
default and has no authentication. Anyone who can reach it can:

- make it fetch any `http(s)` URL given as a reference image or continuation video,
  from the server's network (up to 64 MiB per image, and 2 GiB within 300 s per video);
- use the GPUs for as long as they like.

The endpoints of the underlying SGLang server that load other weights or LoRA adapters,
release GPU memory or run RL rollouts (`/update_weights_from_disk`, `/v1/set_lora`,
`/release_memory_occupation`, `/rollout/...` and the like) are not mounted.

Request fields that would make the service read or write files of its own choosing are
refused by default (HTTP 400): `output_path`, `perf_dump_path`, and local paths in
`video_path`, `input_reference` and `reference_image_paths`. Images are sent as `data:` or
`http(s)` URLs and continuation prefixes as uploads instead. The launcher setting
`"allow_server_paths": true` turns those fields back on; then every client can make the
service read any image or video the service account can read and write files wherever it
can write, so use it only when all clients are trusted
([Server paths](inference/docs/deployment.md#server-paths)). Frame interpolation, upscaling
and `diffusers_kwargs` have been removed: requests that set them are always refused, since
their model fields would make the server download and load other models.

Do not expose the service directly. Put it behind a reverse proxy that authenticates
clients; forwarding only `/health`, `/v1/models` and `/v1/videos` is enough for Murea-Mind.
Run it under an account that can read only the model files and write only its runtime
directory.

## Running Murea-Mind safely

Murea-Mind is built to run on your own machine or a small trusted server.
Before exposing it to the internet, keep in mind:

- **API keys** live in `.env` and `config.yaml`, both ignored by git. Never
  commit them; the server never sends them to the browser.
- **Users** are local profiles. Passwords are optional (PBKDF2 with a
  per-user salt) and a profile without a password can be selected by anyone
  who can open the page. Set passwords, or put the app behind your own
  authentication, on any shared deployment.
- **Data** (`data/`) holds worlds, generated media and per-world call logs
  that include full prompts. Treat it as private.
- **Cost**: every beat calls paid model APIs, and co-op rooms with
  `coop.speculative_video` generate extra clips. Rate-limit or restrict access
  on a public server.
- **Docker**: `docker-compose.yml` publishes the port on `127.0.0.1` only. Publish
  it on other interfaces only behind your own authentication.
- Uploaded images and audio are size-limited and re-encoded or type-checked
  from their bytes, and file paths from requests are validated against `../`
  traversal. An imported world writes only image and video files, its media
  names must be plain file names (no directories, `..` or leading dot), and
  export packs only files inside the media directories. Reference images are
  only ever served as images. Report anything that gets around this.
