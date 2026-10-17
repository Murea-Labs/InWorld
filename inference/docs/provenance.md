# Provenance and licenses

**English** | [中文](provenance_zh.md)

The service is published as Murea-Fast. Internally it keeps the upstream identifiers
`LTX2Pipeline` and `LTX-2.5-Diffusers`: SGLang uses them to select the model implementation,
and keeping them makes it easier to follow upstream. Third-party notices are retained.

## Sources

| Item | Source and version |
| --- | --- |
| Inference engine | [sgl-project/sglang](https://github.com/sgl-project/sglang) at commit `ace7314173c8221ecf5f213575302eab98f4e84f`, with the changes listed below |
| Base components | [Lightricks/LTX-2.5-Diffusers](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) at revision `a6de4b5354f078db24d9cf4778c14846788aea3d` |
| Model weights | Murea-Fast checkpoint [murea-fast.safetensors](https://huggingface.co/MureaLabs/InWorld/blob/main/murea-fast.safetensors), 42,018,192,440 bytes, SHA-256 `1a2c0d8e23032994b52f2c67e14c930d3be8224f321c51353f0a2954ccbb9385` |

`vendor/sglang/python/` and `vendor/sglang/LICENSE` come from that SGLang commit. Of its
4,315 files, 4,288 are byte-identical to upstream, 22 were modified and 3 were added (two
test files and the request policy module). The remaining 2, `python/LICENSE` and `python/README.md`, are copies of upstream's
top-level `LICENSE` and `README.md`, which `python/pyproject.toml` needs to build the
package. The repository does not depend on any other source tree.

SGLang's benchmark tools are not included: the 48 files under `sglang/benchmark/`,
`sglang/kernels/aot/benchmark/` and `sglang/kernels/jit/benchmark/` were left out. The
service does not use them, and the results in [Performance](performance.md) were measured
with this file set. Without them, the deprecated `sglang/bench_*.py` entry points and
`sglang/test/test_utils.py`, along with the upstream tests that import it, cannot be imported.

## Changes to SGLang

Each modified file starts with the line
`# Modified by the InWorld authors from SGLang; see inference/docs/provenance.md.`
and the added module with `# Added by the InWorld authors; see inference/docs/provenance.md.`
Paths are relative to `vendor/sglang/python/sglang/multimodal_gen/`.

| File | Change |
| --- | --- |
| `configs/models/dits/ltx_2.py` | Ref-ID embedding disabled for this checkpoint (`reference_id_max_refs = 0`); packed-QKV switch read from `MUREA_FAST_PACKED_QKV` |
| `configs/pipeline_configs/ltx_2.py` | Records sequence-parallel padding and trims it from gathered latents |
| `configs/sample/ltx_2.py` | `reference_image_paths` and `continuation_prefix_frames` sampling parameters, request lowering and validation (`continuation_prefix_frames` must be an integer); refuses frame interpolation and upscaling on every endpoint, and `perf_dump_path` unless server paths are allowed |
| `runtime/entrypoints/openai/protocol.py` | `reference_image_paths` request field |
| `runtime/entrypoints/openai/video_api.py` | Passes the new fields from HTTP requests to sampling parameters; refuses frame interpolation, upscaling, `diffusers_kwargs` and, unless `MUREA_FAST_ALLOW_SERVER_PATHS` is set, fields that name server files, in multipart and JSON requests alike, and no longer forwards the refused fields; uploaded file names are reduced to their last path component |
| `runtime/entrypoints/openai/request_policy.py` | Added: the request fields Murea-Fast refuses, the `MUREA_FAST_ALLOW_SERVER_PATHS` switch and upload file-name cleaning (standard library only) |
| `runtime/entrypoints/openai/image_api.py`, `runtime/entrypoints/openai/mesh_api.py` | Uploaded file names are reduced to their last path component before saving, as in `video_api.py` |
| `runtime/entrypoints/openai/utils.py` | Image URL downloads are streamed under a 64 MiB size limit and a 60 s total deadline |
| `runtime/entrypoints/http_server.py` | The Vertex route refuses server image paths unless server paths are allowed; the weight, LoRA, memory-release and rollout endpoints are not mounted |
| `utils.py` | Expands `~` in list fields ending in `_paths`, such as `reference_image_paths` |
| `runtime/pipelines_core/stages/image_encoding.py` | Ordered-reference image encoding at training-time sizes; VAE encoding of the continuation prefix and frame-count alignment, decoding only the frames the prefix can use |
| `runtime/pipelines_core/stages/model_specific_stages/ltx_2/denoising.py` | Ordered-reference coordinates and attention routing (negative temporal slots, references excluded from audio-video cross-attention); clean-latent prefix mask restored after each step; sequence-parallel slicing of conditions |
| `runtime/pipelines_core/stages/denoising.py` | Passes the sequence-parallel padding metadata; logs Cache-DiT statistics |
| `runtime/layers/attention/layer.py` | Ordered-reference attention under Ulysses sequence parallelism |
| `runtime/models/dits/ltx_2.py` | Ref-ID embedding applied only when present in the weights; selective SageAttention per attention component |
| `runtime/loader/component_loaders/adapter_loader.py` | Logs the loaded connector path and tensor count |
| `runtime/models/schedulers/scheduling_flow_match_euler_discrete.py`, `runtime/pipelines/ltx_2_pipeline.py` | Sigma schedule computed in float64 so adjacent steps do not collapse; the schedule is checked to be finite, strictly decreasing and of the right length |
| `runtime/cache/cache_dit_integration.py` | Cache-DiT statistics and context refresh between stages (Cache-DiT stays off by default) |
| `runtime/utils/vision.py` | Remote video download with timeouts, a 300 s total deadline, a 2 GiB size limit and temporary-file cleanup; `load_video` can stop after a given number of frames |
| `test/unit/test_cache_dit_integration.py`, `test/unit/test_ltx2_5_config.py` | Tests for the changes above |
| `test/unit/test_ltx2_ordered_multiref.py`, `test/unit/test_ltx2_ti2v_cache.py` | Added tests, including the refused request fields and the input limits |

Outside the vendored engine, this directory adds the launcher, checkpoint checks, connector
merge, client, configurations and tests.

## Licenses

| Component | License |
| --- | --- |
| InWorld code (`src/`, `scripts/`, `tests/`, `configs/`, docs) | [Apache License 2.0](../../LICENSE) |
| Vendored SGLang, including our modifications | [Apache License 2.0](../vendor/sglang/LICENSE), apart from the third-party files in the next row |
| Third-party files within the vendored SGLang | Their own MIT or BSD-3-Clause license; the headers and notices are kept as they are. Examples, relative to `vendor/sglang/python/sglang/`: CUTLASS-derived kernels in `kernels/aot/csrc/attention/cutlass_sm100_mla/` and `kernels/aot/csrc/cutlass_extensions/`, and the flash-attention CuTe kernels in `kernels/ops/attention/flash_attn/cute/` (with its `LICENSE`), BSD-3-Clause; `multimodal_gen/third_party/pynvml.py`, BSD-3-Clause; `multimodal_gen/runtime/models/vaes/dac.py`, MIT. `kernels/aot/THIRDPARTYNOTICES.txt` collects the notices of the projects the AOT kernels come from (flashinfer, TensorRT-LLM, DeepGEMM, flash-attention) |
| Murea-Fast weights | [LTX-2.x Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x) |
| LTX-2.5 base components | [LTX-2.x Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x) |

The Murea-Fast weights are fine-tuned from LTX-2.5 and are therefore a derivative of LTX-2.x
under that agreement. Its terms apply to them regardless of the project name or the code
license. Neither the weights nor the base components are stored in this repository.
