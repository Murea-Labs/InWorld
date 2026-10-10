# Added by the InWorld authors; see inference/docs/provenance.md.
"""Request fields the Murea-Fast service refuses.

Standard library only, so it can be tested without torch or a GPU.

* Frame interpolation, upscaling and ``diffusers_kwargs`` are never used by Murea-Fast.
  The interpolation and upscaling fields name model repositories the server would download
  and load, so a request that sets any of them is refused.
* Fields that name files on the server (``output_path``, ``perf_dump_path`` and local
  ``video_path`` / ``input_reference`` / ``reference_image_paths``) are refused unless the
  launcher sets ``MUREA_FAST_ALLOW_SERVER_PATHS=true`` (``"allow_server_paths": true`` in
  its configuration). ``data:`` and ``http(s)`` URLs and multipart uploads still work.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Iterable, Mapping
from typing import Any

SERVER_PATHS_ENV = "MUREA_FAST_ALLOW_SERVER_PATHS"

UNSUPPORTED_FIELDS = (
    "enable_frame_interpolation",
    "frame_interpolation_exp",
    "frame_interpolation_scale",
    "frame_interpolation_model_path",
    "enable_upscaling",
    "upscaling_model_path",
    "upscaling_scale",
    "diffusers_kwargs",
)
# Files the server would write.
OUTPUT_FILE_FIELDS = ("output_path", "perf_dump_path")
# Images and videos the server would read: a path is read from the server's disk, a URL
# is fetched or decoded.
INPUT_FILE_FIELDS = (
    "video_path",
    "video_url",
    "input_reference",
    "reference_url",
    "reference_image_paths",
)

_URL_PREFIXES = ("data:", "http://", "https://")


class RequestRefused(ValueError):
    """The request uses a field this service does not accept."""


def server_paths_allowed(environ: Mapping[str, str] | None = None) -> bool:
    value = (os.environ if environ is None else environ).get(SERVER_PATHS_ENV, "")
    return value.strip().lower() in ("1", "true", "yes", "on")


def is_url(value: Any) -> bool:
    """Whether `value` is a data: or http(s) URL rather than a server file path."""
    return isinstance(value, str) and value.strip().lower().startswith(_URL_PREFIXES)


def _is_set(value: Any) -> bool:
    return value is not None and value is not False and value != ""


def _server_paths_hint(field: str) -> str:
    return (
        f"{field} names a file on the server, which this service does not accept. "
        "Send images and videos as data: or http(s) URLs or as multipart uploads, or "
        'set "allow_server_paths": true in the launcher configuration if every client '
        "may read and write the service account's files (see inference/docs/deployment.md)."
    )


def upload_file_name(name: Any, default: str = "upload") -> str:
    """A client-supplied upload file name, reduced to a plain last path component.

    Multipart file names come from the client as they are; joined onto the upload
    directory unchecked, "../x" or "a/b" would put the file somewhere else."""
    text = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    text = "".join(char for char in text if char.isprintable()).strip().lstrip(".")
    return text[:200] or default


def check_server_file(
    name: str,
    value: Any,
    *,
    server_inputs: Collection[str] = (),
    allow_server_paths: bool | None = None,
) -> None:
    """Raise RequestRefused if field `name` names a server file the client chose.

    `value` may be a single path or URL or a list of them. `server_inputs` are files the
    server itself saved for this request (multipart uploads, downloaded images); they are
    accepted."""
    if allow_server_paths is None:
        allow_server_paths = server_paths_allowed()
    if allow_server_paths or value is None:
        return
    saved = {str(path) for path in server_inputs}
    items: Iterable[Any] = value if isinstance(value, (list, tuple)) else [value]
    if any(_is_set(item) and not is_url(item) and str(item) not in saved for item in items):
        raise RequestRefused(_server_paths_hint(name))


def check_video_request(
    fields: Mapping[str, Any],
    *,
    server_inputs: Collection[str] = (),
    allow_server_paths: bool | None = None,
) -> None:
    """Raise RequestRefused if `fields` (the request's values by name) use a refused field."""
    for name in UNSUPPORTED_FIELDS:
        if _is_set(fields.get(name)):
            raise RequestRefused(f"{name} is not supported by Murea-Fast.")
    if allow_server_paths is None:
        allow_server_paths = server_paths_allowed()
    if allow_server_paths:
        return
    for name in OUTPUT_FILE_FIELDS:
        if _is_set(fields.get(name)):
            raise RequestRefused(_server_paths_hint(name))
    for name in INPUT_FILE_FIELDS:
        check_server_file(
            name, fields.get(name), server_inputs=server_inputs, allow_server_paths=False
        )
