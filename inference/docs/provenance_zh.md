# 来源与许可证

[English](provenance.md) | **中文**

对外名称为 Murea-Fast。内部仍保留上游标识 `LTX2Pipeline` 和 `LTX-2.5-Diffusers`：
SGLang 依靠它们选择模型实现，保留它们也便于跟进上游。第三方署名均已保留。

## 来源

| 项目 | 来源与版本 |
| --- | --- |
| 推理引擎 | [sgl-project/sglang](https://github.com/sgl-project/sglang)，提交 `ace7314173c8221ecf5f213575302eab98f4e84f`，加上下文列出的修改 |
| 基础组件 | [Lightricks/LTX-2.5-Diffusers](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)，revision `a6de4b5354f078db24d9cf4778c14846788aea3d` |
| 模型权重 | Murea-Fast 权重 `model_ema_distilled8_step_07000.safetensors`，42,018,192,440 字节，SHA-256 `1a2c0d8e23032994b52f2c67e14c930d3be8224f321c51353f0a2954ccbb9385` |

`vendor/sglang/python/` 和 `vendor/sglang/LICENSE` 来自上述 SGLang 提交。其中 4,315 个文件里，
4,288 个与上游逐字节一致，22 个经过修改，另新增 3 个（2 个测试文件和请求字段策略模块）；其余 2 个文件
`python/LICENSE` 和 `python/README.md` 复制自上游顶层的 `LICENSE` 和 `README.md`，
`python/pyproject.toml` 构建安装包时需要它们。本仓库不依赖其他源码目录。

未包含 SGLang 的 benchmark 工具：`sglang/benchmark/`、`sglang/kernels/aot/benchmark/` 和
`sglang/kernels/jit/benchmark/` 下共 48 个文件没有收录。服务不使用这些文件，
[性能](performance_zh.md)中的结果也是在这套文件上测得的。缺少它们时，已弃用的
`sglang/bench_*.py` 入口和 `sglang/test/test_utils.py`（以及导入它的上游测试）无法导入。

## 对 SGLang 的修改

每个修改过的文件开头都有一行
`# Modified by the nWorld authors from SGLang; see inference/docs/provenance.md.`，
新增的模块开头是 `# Added by the nWorld authors; see inference/docs/provenance.md.`。
下表路径相对于 `vendor/sglang/python/sglang/multimodal_gen/`。

| 文件 | 修改内容 |
| --- | --- |
| `configs/models/dits/ltx_2.py` | 对此权重关闭 Ref-ID embedding（`reference_id_max_refs = 0`）；从 `MUREA_FAST_PACKED_QKV` 读取合并 QKV 开关 |
| `configs/pipeline_configs/ltx_2.py` | 记录序列并行的补齐信息，并在汇聚 latent 后去掉补齐部分 |
| `configs/sample/ltx_2.py` | 新增采样参数 `reference_image_paths`、`continuation_prefix_frames`，以及请求下沉和校验（`continuation_prefix_frames` 必须是整数）；在所有接口上拒绝插帧和超分，未允许服务器路径时拒绝 `perf_dump_path` |
| `runtime/entrypoints/openai/protocol.py` | 新增请求字段 `reference_image_paths` |
| `runtime/entrypoints/openai/video_api.py` | 把 HTTP 请求中的新字段传给采样参数；对 multipart 和 JSON 请求一样拒绝插帧、超分、`diffusers_kwargs`，以及在未设置 `MUREA_FAST_ALLOW_SERVER_PATHS` 时指定服务器文件的字段，且不再传递这些被拒绝的字段；上传文件名只保留最后一级路径 |
| `runtime/entrypoints/openai/request_policy.py` | 新增：Murea-Fast 拒绝的请求字段、`MUREA_FAST_ALLOW_SERVER_PATHS` 开关和上传文件名清理（只依赖标准库） |
| `runtime/entrypoints/openai/image_api.py`、`runtime/entrypoints/openai/mesh_api.py` | 与 `video_api.py` 一样，保存前把上传文件名截成最后一级路径 |
| `runtime/entrypoints/openai/utils.py` | 图片 URL 改为流式下载，限制 64 MiB，总时限 60 秒 |
| `runtime/entrypoints/http_server.py` | 未允许服务器路径时，Vertex 接口拒绝服务器上的图片路径；不挂载权重、LoRA、释放显存和 rollout 接口 |
| `utils.py` | 对以 `_paths` 结尾的列表字段（如 `reference_image_paths`）展开 `~` |
| `runtime/pipelines_core/stages/image_encoding.py` | 按训练时尺寸编码有序参考图；续写前缀的 VAE 编码与帧数对齐，只解码前缀用得到的帧 |
| `runtime/pipelines_core/stages/model_specific_stages/ltx_2/denoising.py` | 有序参考图的坐标与注意力路由（负时间槽、参考图不参与音视频 cross attention）；clean-latent 前缀 mask 及每步后的前缀恢复；序列并行下的条件切片 |
| `runtime/pipelines_core/stages/denoising.py` | 传递序列并行补齐信息；记录 Cache-DiT 统计 |
| `runtime/layers/attention/layer.py` | Ulysses 序列并行下的有序参考图注意力 |
| `runtime/models/dits/ltx_2.py` | 仅当权重中存在 Ref-ID embedding 时才使用；按注意力组件选择性启用 SageAttention |
| `runtime/loader/component_loaders/adapter_loader.py` | 记录加载的 connector 路径和张量数 |
| `runtime/models/schedulers/scheduling_flow_match_euler_discrete.py`、`runtime/pipelines/ltx_2_pipeline.py` | 用 float64 计算 sigma 序列，避免相邻步骤因精度不足而重合；并检查序列有限、严格递减且长度正确 |
| `runtime/cache/cache_dit_integration.py` | Cache-DiT 统计和阶段间的上下文刷新（Cache-DiT 默认关闭） |
| `runtime/utils/vision.py` | 远程视频下载增加超时、300 秒总时限、2 GiB 大小上限和临时文件清理；`load_video` 可以在读到指定帧数后停止 |
| `test/unit/test_cache_dit_integration.py`、`test/unit/test_ltx2_5_config.py` | 上述修改的测试 |
| `test/unit/test_ltx2_ordered_multiref.py`、`test/unit/test_ltx2_ti2v_cache.py` | 新增的测试，包括被拒绝的请求字段和输入限制 |

在 vendor 推理引擎之外，本目录新增了启动器、权重校验、connector 合并、客户端、配置和测试。

## 许可证

| 组成部分 | 许可证 |
| --- | --- |
| nWorld 代码（`src/`、`scripts/`、`tests/`、`configs/`、文档） | [Apache License 2.0](../../LICENSE) |
| vendor 中的 SGLang，含我们的修改 | [Apache License 2.0](../vendor/sglang/LICENSE)，下一行的第三方文件除外 |
| vendor 中 SGLang 内的第三方文件 | 各自的 MIT 或 BSD-3-Clause 许可证；文件头和声明均原样保留。例如（路径相对于 `vendor/sglang/python/sglang/`）：`kernels/aot/csrc/attention/cutlass_sm100_mla/` 与 `kernels/aot/csrc/cutlass_extensions/` 中源自 CUTLASS 的 kernel，以及 `kernels/ops/attention/flash_attn/cute/` 中的 flash-attention CuTe kernel（附带 `LICENSE`），为 BSD-3-Clause；`multimodal_gen/third_party/pynvml.py`，BSD-3-Clause；`multimodal_gen/runtime/models/vaes/dac.py`，MIT。`kernels/aot/THIRDPARTYNOTICES.txt` 汇总了 AOT kernel 所依据的上游项目（flashinfer、TensorRT-LLM、DeepGEMM、flash-attention）的声明 |
| Murea-Fast 权重 | [LTX-2.x Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x) |
| LTX-2.5 基础组件 | [LTX-2.x Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x) |

Murea-Fast 权重由 LTX-2.5 微调而来，因此属于该协议定义的 LTX-2.x 衍生模型，
无论项目名称或代码许可证如何，该协议的条款都适用于这些权重。
权重和基础组件都不存放在本仓库中。
