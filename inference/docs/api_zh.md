# API

[English](api.md) | **中文**

服务在 `http://<host>:<port>`（默认 `http://127.0.0.1:30260`）上提供异步的视频任务 API。
服务没有鉴权；如需供其他机器访问，请在前面加一层负责鉴权的反向代理，并先阅读
[安全说明](../../SECURITY_zh.md#安全运行-murea-fast)。

## 接口

| 方法与路径 | 用途 |
| --- | --- |
| `GET /health` | 模型加载并完成预热后返回 200，此前返回 503 |
| `GET /v1/models` | 当前服务的模型信息；Murea-Fast 的 `pipeline_class` 为 `LTX2Pipeline` |
| `POST /v1/videos` | 提交生成或续写任务，返回带 `id` 的任务 |
| `GET /v1/videos/{id}` | 查询任务 |
| `GET /v1/videos/{id}/content` | 下载生成好的 MP4；任务未完成时返回 404 |
| `GET /v1/videos` | 列出任务 |
| `DELETE /v1/videos/{id}` | 从任务列表中移除任务，响应中 `status` 为 `"deleted"`。不会中止正在运行的任务，输出文件也会保留在磁盘上 |

SGLang 中加载其他权重或 LoRA、释放显存、运行 RL rollout 的接口（`/update_weights_from_*`、
`/release_memory_occupation`、`/v1/set_lora`、`/rollout/...` 等）都没有挂载。

任务初始状态为 `queued`，最终变为 `completed` 或 `failed`。失败时，`error.message` 给出原因。
任务列表保存在内存中，服务重启后清空。

## 示例

```bash
curl -s http://127.0.0.1:30260/v1/videos \
  -H 'Content-Type: application/json' -d @request.json
# {"id": "0c2dc7e2-...", "status": "queued", ...}

curl -s http://127.0.0.1:30260/v1/videos/0c2dc7e2-...
# 轮询，直到 "status" 变为 "completed" 或 "failed"

curl -s -o video.mp4 http://127.0.0.1:30260/v1/videos/0c2dc7e2-.../content
```

在 Python 中可以调用 `murea_fast.client.generate(base_url, payload, output_dir)`：
它会提交任务、轮询、下载视频并校验结果，`scripts/murea-fast generate` 用的就是它。
[Murea-Mind](../../agent/README_zh.md) 通过 `agent/murea_agent/providers/video/murea.py`
调用同一套 API。它始终按[在其他机器上调用](#在其他机器上调用)的方式发送请求，因此服务可以在
同一台机器上、其他机器上，或通过 SSH 隧道访问。

## 请求字段

以 `Content-Type: application/json` 发送 JSON 请求体。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `prompt` | 字符串 | 必填 |
| `reference_image_paths` | 字符串列表 | 按顺序排列的参考图。每一项可以是 `data:` URL、由服务端下载的 `http(s)` URL（最大 64 MiB），或者（仅在服务允许[服务器路径](deployment_zh.md#服务器路径)时）服务端上的绝对路径。验收测试使用 3 张；另有一次端到端测试用 5 张（三个玩家角色、一只怪物和一个地点）生成了一段 5 秒片段，运行无误，但未评估画质。超过 5 张未测试。参考图越大，每一步都越慢；每张约 180 万像素就够了（见[性能](performance_zh.md#参考图尺寸)） |
| `video_path` | 字符串 | 仅续写：前缀视频，可以是 `http(s)` URL（最大 2 GiB，须在 300 秒内下载完），或者在允许[服务器路径](deployment_zh.md#服务器路径)时为服务端上的绝对路径。不允许时请上传视频（见[在其他机器上调用](#在其他机器上调用)） |
| `continuation_prefix_frames` | 整数 | 仅续写：从 `video_path` 开头取多少帧作为前缀。必须是 JSON 整数，`17.0` 或 `"17"` 会被拒绝 |
| `width`、`height` | 整数 | 32 的正整数倍；已验证 1280×704 |
| `num_frames` | 整数 | `8n+1`，9 到 481（20 秒）；121 帧为 5 秒，241 帧为 10 秒。续写时前缀也计入其中 |
| `fps` | 整数 | 必须为 24 |
| `num_inference_steps` | 整数 | 必须为 8 |
| `guidance_scale` | 数值 | 必须为 1.0 |
| `seed` | 整数 | 随机种子 |
| `generator_device` | 字符串 | `cuda` |
| `perf_dump_path` | 字符串 | 可选，服务端绝对路径；写入各阶段及每步耗时。仅在允许[服务器路径](deployment_zh.md#服务器路径)时可用 |

`fps`、`num_inference_steps` 和 `guidance_scale` 由蒸馏权重决定。自带客户端会自动填入这些值，
并拒绝其他取值；直接调用 API 时，请严格使用上述取值。

续写规则：`continuation_prefix_frames` 和 `num_frames` 都必须是 `8n+1`，前缀帧数小于
`num_frames`，且 `video_path` 不能与 `reference_image_paths` 同时使用。服务从 `video_path`
的**开头**取前缀，因此需要先把父视频裁成它的末尾几帧。`scripts/murea-fast` 的 `continue`
命令会自动完成这一步。

### 在其他机器上调用

远程客户端无法传自己机器上的文件路径，通过 SSH 隧道以 `127.0.0.1` 访问服务的客户端也一样。
服务默认不允许[服务器路径](deployment_zh.md#服务器路径)，这时也只能这样调用。
参考图请以 `data:` URL 形式放进 `reference_image_paths`。续写时请用 `multipart/form-data` 发送请求：
前缀视频作为文件 `video_reference` 上传，其余字段作为表单字段。multipart 请求能读取的字段比 JSON 请求少：

- 分辨率请用 `size` 指定，例如 `1280x704`。`width` 和 `height` 会被忽略，服务会退回默认分辨率。
- 不读取 `continuation_prefix_frames`。服务把上传的视频整体作为前缀，帧数向下取整到 `8n+1`，
  并且要小于 `num_frames`，因此请只上传想作为前缀的那几帧，例如父视频的末尾 17 帧。
- 不读取 `perf_dump_path`。

服务会把上传的视频保存在其工作目录下的 `inputs/uploads/` 中（用 `scripts/murea-fast` 启动时即
`inference/` 目录），并且不会自动删除。

### 会被拒绝的字段

底层 SGLang 服务还有一些 Murea-Fast 不使用的字段。请求中设置了 `enable_frame_interpolation`、
`frame_interpolation_exp`、`frame_interpolation_scale`、`frame_interpolation_model_path`、
`enable_upscaling`、`upscaling_model_path`、`upscaling_scale` 或 `diffusers_kwargs`
（取值不是 `null` 或 `false`）时，服务会返回 HTTP 400。服务不允许[服务器路径](deployment_zh.md#服务器路径)时，
`output_path` 也会被拒绝，输出始终写到服务的运行目录。

## 提示词格式

验证时使用的提示词先逐个描述参考图，再描述镜头，并按参考图在 `reference_image_paths`
中的位置引用它们：

```text
<Subject 1> is a single close-up view of a woman with long brown hair, wearing a light blue sweater, as shown in <Picture 1>.
<Subject 2> is a single white ceramic coffee cup on a plain background, as shown in <Picture 2>.

<Shot 1> at 0.00s, A woman, <Subject 1> from <Picture 1>, sits at a wooden table by a window in soft daylight. She picks up a white cup, <Subject 2> from <Picture 2>, smiles and takes a sip. The camera holds a steady medium shot. Quiet room ambience.
```

## 响应字段

| 字段 | 说明 |
| --- | --- |
| `id` | 任务 ID |
| `status` | `queued`、`completed` 或 `failed`；`DELETE` 的响应中为 `deleted` |
| `error` | 任务失败时为 `{"message": ...}` |
| `inference_time_s` | 服务端推理耗时（秒） |
| `peak_memory_mb` | PyTorch reserved 显存峰值，单位 MiB（字节数 / 1024²）；多卡时不是各卡之和 |
| `file_path` | 输出视频在服务端上的路径 |
| `created_at`、`completed_at` | Unix 时间戳 |
