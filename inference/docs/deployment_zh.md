# 部署说明

[English](deployment.md) | **中文**

以下命令均在 `inference/` 目录下执行。

## 目录结构

- `src/murea_fast/`：启动器、权重校验、生成与续写客户端。
- `vendor/sglang/python/sglang/multimodal_gen/`：模型、采样、VAE、分布式和 REST 实现。
- `scripts/materialize_connectors.py`：把训练过的 connector 合并进基础组件，并逐文件校验。
- `scripts/acceptance.py`：对运行中的服务做端到端检查。
- `configs/`：服务配置和示例请求。
- `tests/`：客户端与启动器检查，以及运行时冒烟测试。

源码可以放在任意目录。配置中的路径必须是运行服务那台机器上的绝对路径。
`MUREA_PYTHON` 用于指定 `scripts/murea-fast` 包装脚本使用的 Python；
服务本身使用配置中 `python` 指定的解释器。

## 环境

已验证环境的版本：Torch 2.13.0、TorchVision 0.28.0、TorchAudio 2.11.0、
Transformers 5.12.1、Diffusers 0.37.0、Safetensors 0.8.0、Cache-DiT 1.3.0、
Triton 3.7.1、sglang-kernel 0.4.6.post1。

请安装到独立的虚拟环境中。CUDA 和驱动需要与 PyTorch 版本匹配，部分可选依赖可能需要预先编译：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.18.dev634 \
  .venv/bin/python -m pip install './vendor/sglang/python[diffusion]'
.venv/bin/python -m pip install -e .
```

这会安装 vendor 中 SGLang 声明的依赖。在新机器或新 GPU 架构上从零安装尚未做过完整验证，
装好后请用 `tests/smoke_runtime.py` 和下文的端到端检查确认。

只支持可编辑安装（如上的 `pip install -e .`）或在本目录中直接运行 `scripts/murea-fast`。
启动器（`prepare`、`start`、`stop`、`status`、`command`）需要源码目录中的 `vendor/` 和 `scripts/`，
普通的 `pip install .` 不会安装它们，此时启动器会报错并说明原因。`verify`、`generate` 和
`continue` 不受安装方式限制。

启动器总是把仓库自带的 `vendor/sglang/python` 放在 `PYTHONPATH` 最前面，
避免误用其他地方安装的 SGLang。

还需要 `ffmpeg` 和 `ffprobe`：放在 `PATH` 上，或者把 `MUREA_FFMPEG`、`MUREA_FFPROBE`
设为可执行文件的完整路径。客户端在提交任务前会检查这两个工具，
避免生成完成后才发现缺少媒体工具。

SageAttention 是可选项：单独安装兼容的版本，并在配置中设置 `sage_pythonpath`。

## 基础组件

服务需要完整的 `LTX-2.5-Diffusers` 目录：`model_index.json`、tokenizer 与 text encoder、
connectors、视频与音频 VAE、vocoder。DiT 由 Murea-Fast 权重替换。按固定版本下载：

```bash
hf auth login   # 需要先在 Hugging Face 上同意该仓库的许可协议
hf download Lightricks/LTX-2.5-Diffusers \
  --revision a6de4b5354f078db24d9cf4778c14846788aea3d \
  --local-dir /path/to/LTX-2.5-Diffusers
```

## 权重

下载 Murea-Fast 权重（见 [README](../README_zh.md#模型)）并校验：

```bash
bash scripts/murea-fast verify /path/to/model_ema_distilled8_step_07000.safetensors --sha256
```

不加 `--sha256` 时，`verify` 检查文件大小、张量数量以及权重元数据中记录的采样约束；
加上 `--sha256` 会额外对整个 42 GB 文件计算哈希。

权重和基础组件适用
[LTX-2.x Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x)。
如果把它们或由其衍生的模型提供给他人，例如搭建镜像站点或发布包含它们的容器镜像，需要附上协议的完整副本，
把协议中的使用限制（Attachment A）作为可执行条款写进你自己的许可或协议，并书面告知接收方，
年收入达到门槛的实体需要向 Lightricks 获取付费许可。详见[许可证](../README_zh.md#许可证)。

## 配置

从 `configs/` 中的某个文件开始修改：

| 文件 | 配置 | 是否已用此权重验证 |
| --- | --- | --- |
| `single-gpu.json` | 单卡，SDPA | 是 |
| `4gpu-ulysses.json` | 4 卡，Ulysses SP4 | 是，在 141 GB 版 H20 上验证，包括最长 20 秒的片段（[性能](performance_zh.md#4-卡)） |
| `4gpu-ulysses-packed-qkv.json` | 4 卡，Ulysses SP4，合并 QKV，使用 4-7 号卡和另一组端口 | 尚未 |

| 字段 | 含义 |
| --- | --- |
| `python` | 运行服务的 Python 的绝对路径 |
| `model_dir` | `LTX-2.5-Diffusers` 目录的绝对路径 |
| `checkpoint` | Murea-Fast 权重的绝对路径 |
| `runtime_dir` | 存放日志、缓存、输出和合并后 connector 的绝对路径 |
| `gpus` | 逗号分隔的 GPU 编号组成的字符串，编号与 `nvidia-smi` 一致，例如 `"0"` 或 `"0,1,2,3"`；1、2、4 或 8 张互不重复的卡。启动器会设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，让 CUDA 使用相同的编号 |
| `parallel` | `ulysses`（序列并行）或 `tp`（张量并行） |
| `attention` | `sdpa` 或 `sage_attn` |
| `sage_pythonpath` | 包含 `sageattention` 包的目录；使用 `sage_attn` 时必填 |
| `packed_qkv` | Ulysses 下用一次 all-to-all 交换 Q/K/V；默认 `false` |
| `compile` | 启用 `torch.compile`；默认 `false` |
| `host` | 监听地址；默认 `127.0.0.1`。设为 `0.0.0.0` 或 `::` 时，`status` 给出回环地址的健康检查 URL |
| `allow_server_paths` | 是否允许请求指定服务器上的文件；默认 `false`，见[服务器路径](#服务器路径) |
| `port`、`scheduler_port`、`master_port` | 三个互不相同、位于 1024 到 65535 之间的端口 |

没有默认值的字段都必须填写（`sage_pythonpath` 只在使用 `sage_attn` 时需要）；缺少字段时启动器会指出是哪一个。

选择卡数并不意味着配置能适配任意显存大小，实测显存占用见[性能](performance_zh.md)。

### 服务器路径

默认情况下，服务接受 `data:` 或 `http(s)` URL 形式的图片，续写前缀以 multipart 上传，
Murea-Mind 发送的正是这两种。凡是指定服务器上文件的请求都会被拒绝并返回 HTTP 400：
`output_path`、`perf_dump_path`，以及 `video_path`、`input_reference` 或
`reference_image_paths` 中的本地路径。

设为 `"allow_server_paths": true` 后服务会接受这些字段。此时任何能访问服务的客户端都能让它
读取服务账号可读的任意图片或视频，并在服务账号可写的任何位置写文件，所以只有在所有客户端都可信时
（例如只有一个人使用的机器）才打开。随附的配置都没有打开它。`murea-fast continue`、
`scripts/acceptance.py` 和示例请求 `configs/request.example.json` 使用服务器路径，需要打开它；
对没有打开的服务运行时，它们会报错并指出这个设置。

无论这个设置如何，插帧、超分和 `diffusers_kwargs` 都会被拒绝：Murea-Fast 不使用它们，
而其中的模型字段会让服务器下载并加载其他模型。

## 服务生命周期

```bash
bash scripts/murea-fast prepare --config configs/single-gpu.json
bash scripts/murea-fast start   --config configs/single-gpu.json
bash scripts/murea-fast status  --config configs/single-gpu.json
bash scripts/murea-fast stop    --config configs/single-gpu.json
bash scripts/murea-fast command --config configs/single-gpu.json   # 打印启动命令
```

- **prepare**：校验权重元数据和大小。首次运行时计算完整 SHA-256，
  并在 `runtime_dir/connectors-murea-fast/` 中生成合并后的 connector。
  之后会先校验该目录中每个文件的哈希再复用。如果权重文件的大小、时间戳或 inode 发生变化，
  会重新计算完整 SHA-256。不完整或不匹配的 connector 目录不会被静默覆盖：
  删除该目录后重新运行 `prepare`。
- **start**：先执行 `prepare`；如果所选 GPU 上已有进程或显存占用超过 1 GiB，
  或者端口被占用，则拒绝启动，且不会停止其他任务。进程启动后立即返回，
  `GET /health` 通过才算就绪。
- **status**：报告登记的进程是否仍在运行。
- **stop**：核对登记的进程号、启动时间、工作目录和启动命令后，向它启动的进程组发送 SIGTERM，
  等进程组中所有进程都退出后才报告已停止（GPU worker 可能比服务主进程退出得晚）。
  60 秒后仍有进程在运行时会给出提示。不发送 SIGKILL，也不影响其他进程。
  请在同一台机器上用同一个配置执行。

状态按主机和端口保存在 `runtime_dir/<hostname>-<port>/` 中：`server.log`、
`service.json`（启动命令、权重校验记录、运行源码的 SHA-256）、缓存和输出。

迁移到其他机器或集群时，请显式拷贝并校验权重和所有基础组件。

## 端到端检查

服务健康检查通过后，在服务所在机器上运行。脚本以服务器路径传递参考图和续写前缀，
并让服务写出计时文件，所以服务必须设置 `"allow_server_paths": true`（见[服务器路径](#服务器路径)）：

```bash
.venv/bin/python scripts/acceptance.py --case /path/to/case --output /path/to/new-run-dir
```

case 目录中放 `prompt.txt`、按顺序命名的参考图 `ref_00.jpg`、`ref_01.png`……，
以及可选的 `continuation_prompt.txt`。脚本会把同一个 case 生成两次，再对第二段视频做续写，
校验每个输出并写出 `summary.json`。`--resume` 会复核已完成的输出，只执行缺失的步骤。

## 常见问题

- **进程在运行但 `/health` 不通过**：查看 `runtime_dir/<hostname>-<port>/server.log`，
  模型可能仍在加载。
- **请求中的文件找不到**：请求里的路径由服务端读取，你本机上的路径对远程服务不可见；
  请以 `data:` URL 或上传的方式发送文件。
- **返回 400 并提到 `allow_server_paths`**：请求指定了服务器上的文件，见[服务器路径](#服务器路径)。
- **显存不足**：降低分辨率或帧数，或者增加卡数。
- **GPU 或端口被占用**：换用空闲的卡和三个未占用的端口。启动器不会停止其他任务。
- **帧数**：24 fps，帧数为 `8n+1`：121 帧为 5 秒，241 帧为 10 秒，最多 481 帧（20 秒）。17 帧前缀加 120 帧新内容，输出共 137 帧。
- **续写音频**：会重新生成。如果对音频连续性有要求，需要单独处理音轨并检查效果。
- **停止服务**：在同一台机器上用同一个配置执行 `stop`，不要用 `pkill python`。
