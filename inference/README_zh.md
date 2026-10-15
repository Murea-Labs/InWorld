# Murea-Fast

[English](README.md) | **中文**

> **状态：** Murea-Fast 权重尚未发布，将在 Hugging Face 上发布（见[模型](#模型)）。在此之前服务无法启动；
> 代码、文档和只依赖标准库的测试可以使用。

Murea-Fast 是 [Murea-InWorld](../README_zh.md) 的推理服务：根据文本提示和按顺序排列的参考图，
生成带同步音频的视频；也可以从一段已有视频的末尾帧接着往下生成。模型通过 HTTP API 提供服务。

本目录可以独立使用：其中包含带模型实现的固定版本 SGLang（`vendor/sglang/`）、
启动器与客户端（`src/murea_fast/`）以及现成的配置。运行时仍需模型权重、
LTX-2.5 基础组件和 CUDA 环境。

## 功能

- **多参考图生成**：参考图按请求中的顺序绑定，例如人物、物体、场景，并在提示词中引用。
- **clean-latent 续写**：把视频末尾 17 帧编码为固定的 latent 前缀，在其后生成新内容。
- **8 步蒸馏采样**：Euler，CFG = 1，24 fps。
- **多卡**：在 1/2/4/8 卡上使用 Ulysses 序列并行或张量并行；
  可选 SageAttention、合并 QKV 的 all-to-all 通信和 `torch.compile`。
- **可追溯**：每个请求都记录输入、耗时、峰值显存和 FFprobe 报告，
  并完整解码输出视频，核对帧数和音频轨道。

## 模型

| 项目 | 值 |
| --- | --- |
| 权重文件 | `model_ema_distilled8_step_07000.safetensors` |
| 大小 | 42,018,192,440 字节 |
| SHA-256 | `1a2c0d8e23032994b52f2c67e14c930d3be8224f321c51353f0a2954ccbb9385` |
| 基础模型 | 基于 [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) 微调并蒸馏 |
| 下载 | 将发布到 Hugging Face，链接发布后补充 |

权重中包含 DiT 和 258 个训练过的 text connector 张量。`prepare` 会把这些 connector
合并进基础组件：这 258 个张量替换基础组件中的对应张量，基础组件独有的 4 个 text projection
张量（`audio_text_proj_in` 和 `video_text_proj_in` 的 weight 与 bias）保持不变。
因此不能只把它当作 DiT 替换文件使用。启动器只接受这一个权重。

## 环境要求

- Linux、NVIDIA GPU 与 CUDA、Python 3.10+、FFmpeg 和 ffprobe。
- 已验证（1280×704）：单张 NVIDIA H20 生成 5 秒片段（每段 44–101 秒，取决于参考图尺寸；
  峰值 reserved 显存 80,472 MiB）；4 张 141 GB 版 H20（Ulysses SP4）生成 5–20 秒片段
  （每卡峰值 81,486–114,206 MiB，约 80–112 GiB）。4 卡生成 10 秒及更长的片段时，每卡 reserved
  显存超过 96 GB，因此尚未证明能在 96 GB 版 H20 上运行。片段最长 20 秒，
  详见[片段时长](docs/performance_zh.md#片段时长)。更小显存的 GPU、单卡生成更长片段以及其他多卡配置尚未用此权重验证。

## 快速开始

1. 按[部署说明](docs/deployment_zh.md)安装环境，并下载 LTX-2.5 基础组件。
2. 下载 Murea-Fast 权重并校验：

   ```bash
   bash scripts/murea-fast verify /path/to/model_ema_distilled8_step_07000.safetensors --sha256
   ```

3. 修改 `configs/single-gpu.json` 中的四个路径（Python、基础组件、权重、运行目录），
   并在 `gpus` 中选择一张空闲的卡。
4. 准备并启动服务，然后等待健康检查通过：

   ```bash
   bash scripts/murea-fast prepare --config configs/single-gpu.json
   bash scripts/murea-fast start --config configs/single-gpu.json
   curl --fail http://127.0.0.1:30260/health
   ```

   `start` 返回只代表进程已启动；`/health` 通过才算就绪，在此之前模型可能仍在加载。
5. 在同一台机器上用同一个配置停止服务：

   ```bash
   bash scripts/murea-fast stop --config configs/single-gpu.json
   ```

## 多参考图生成

复制 `configs/request.example.json`，填写提示词，并按顺序填写参考图路径，然后运行：

```bash
bash scripts/murea-fast generate --request my-request.json --output outputs/sample
```

参考图路径由服务端读取，必须是服务所在机器上存在的路径，而且只有服务配置设置了
`"allow_server_paths": true` 时才会被接受（见[服务器路径](docs/deployment_zh.md#服务器路径)），默认关闭。
未打开时，或在其他机器上调用时，请改用 `data:` URL（见 [API 文档](docs/api_zh.md#在其他机器上调用)）。输出目录中包含 `video.mp4`、
原始请求、提交与完成时的响应、`timing.json`（API 推理耗时、客户端等待与下载耗时、
峰值显存）和 `ffprobe.json`。允许服务器路径时，把 `perf_dump_path` 设为服务端绝对路径，
还可以额外记录各阶段及每步的耗时。

`generate` 和 `continue` 默认连接 `http://127.0.0.1:30260`。服务在其他地址时用 `--base-url`
指定；`--timeout` 设置客户端等待任务完成的时长，单位为秒，默认 1800。

## 续写

在服务所在机器或与其共享的文件系统上运行。该命令以服务器路径传递截取后的前缀和 `perf_dump_path`，
所以服务必须设置 `"allow_server_paths": true`（见[服务器路径](docs/deployment_zh.md#服务器路径)）：

```bash
bash scripts/murea-fast continue \
  --parent outputs/sample/video.mp4 \
  --prompt 'The person continues walking, turns towards the camera and smiles. Natural footsteps.' \
  --prefix-frames 17 --new-frames 120 --seed 42 \
  --output outputs/continuation
```

客户端截取父视频的**末尾 17 帧**，服务对其做 VAE 编码，作为 clean latent 前缀：
前缀不参与去噪，并在每步更新后恢复。输出共 **137 帧：17 帧前缀加 120 帧新内容**。

输出目录中包含 `prefix.mp4`（截取后提交给服务的尾部片段）、`parent.json`（父视频路径、
分辨率与帧数）和 `perf.json`（各阶段耗时）。成片及其请求、响应、`timing.json` 和
`ffprobe.json` 位于 `generated/` 子目录，因此上面命令的成片是
`outputs/continuation/generated/video.mp4`。

限制与注意事项：

- 前缀帧数和总帧数都必须是 `8n+1`，且总帧数大于前缀帧数。
- 续写不能和参考图同时使用。
- 续写仅支持 one-stage 推理。启动器固定启动 one-stage 的 `LTX2Pipeline`，不支持 two-stage pipeline。
- 音频对整段重新生成，不继承父视频的音频。
- VAE 重建的前缀不保证与父视频逐像素一致。
- 输出包含重复的前缀；拼接长片时，应去掉每段续写的前 17 帧。
- 这是整段去噪的续写，不是分块流式生成，也不复用 KV cache。

## API

`POST /v1/videos` → `GET /v1/videos/{id}` → `GET /v1/videos/{id}/content`。
请求字段和采样约束见 [API 文档](docs/api_zh.md)。

服务默认只监听 `127.0.0.1`，且没有鉴权。如需对外提供服务，请在前面加一层负责鉴权的反向代理，
并先阅读[安全说明](../SECURITY_zh.md#安全运行-murea-fast)。除非配置允许，指定服务器文件的请求会被拒绝；
插帧、超分和 `diffusers_kwargs` 也一律拒绝（见[会被拒绝的字段](docs/api_zh.md#会被拒绝的字段)）。

## 测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=vendor/sglang/python /path/to/.venv/bin/python tests/smoke_runtime.py
```

第一条（装有 `pytest` 时也可以直接运行 `pytest`）只依赖标准库，检查客户端、启动器和请求字段。第二条需要完整环境，检查采样参数、
HTTP 字段下沉、帧对齐、Ulysses 前缀切片和 clean-latent mask。我们在内置引擎中的测试，例如
`vendor/sglang/python/sglang/multimodal_gen/` 下的 `test/unit/test_ltx2_ordered_multiref.py`，
需要在完整环境中用 `pytest` 运行，其中也覆盖了会被拒绝的请求字段和输入限制。

## 文档

- [部署说明](docs/deployment_zh.md)：安装、基础组件、配置项、服务生命周期、常见问题。
- [API](docs/api_zh.md)：接口、请求与响应字段。
- [性能](docs/performance_zh.md)：优化开关与实测结果。
- [来源说明](docs/provenance_zh.md)：上游来源、修改过的文件、许可证。

## 许可证

- 本目录中的代码以 [Apache License 2.0](../LICENSE) 发布。vendor 中的 SGLang 同为
  Apache-2.0（[许可证](vendor/sglang/LICENSE)），其中保留自身 MIT 或 BSD-3-Clause 声明的第三方文件除外；
  我们修改过的文件在开头带有修改声明。两者详见[来源说明](docs/provenance_zh.md)。
- Murea-Fast 权重是 LTX-2.5 的衍生模型，按
  [LTX-2.x Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x)
  分发。该协议要求年收入达到 1000 万美元及以上的实体向 Lightricks 获取商业许可，
  并包含使用限制等条款。LTX-2.5 基础组件同样适用该协议。使用权重前请先阅读协议全文。
- 再分发权重或由其衍生的模型时，该协议要求：附上协议的完整副本；把协议中的使用限制
  （Attachment A）作为可执行条款写进你自己的许可或协议；并书面告知接收方，
  年收入达到门槛的实体需要向 Lightricks 获取付费许可。
- Murea-Fast 与 Lightricks 无隶属关系，也未获其背书。
