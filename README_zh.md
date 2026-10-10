# InWorld

[English](README.md) | **中文**

> **状态：** Murea-Fast 模型权重尚未发布。权重将在 Hugging Face 上发布，届时链接会补充到
> [Murea-Fast README](inference/README_zh.md#模型)。在此之前推理服务无法启动，Agent 也就没有可用的视频模型。

InWorld 是一个开源的视频生成系统：根据文本提示和按顺序排列的参考图生成带同步音频的视频，
可以接着已有视频继续生成，还包含一个 agent，把玩家的行动拍成一段段连续的故事。

| 组成部分 | 目录 | 说明 |
| --- | --- | --- |
| **Murea-Fast** | [`inference/`](inference/README_zh.md) | 推理服务：通过 HTTP API 提供模型服务 |
| **Murea Agent** | [`agent/`](agent/README_zh.md) | 带网页界面的开放世界视频 agent：用语言模型规划每一拍，再用 Murea-Fast 拍出来 |

## 快速上手

1. 在 GPU 机器上启动 Murea-Fast 服务，见 [Murea-Fast README](inference/README_zh.md)。
2. 用 `MUREA_URL` 把 Murea Agent 指向这个服务并运行，见 [Murea Agent README](agent/README_zh.md)。
3. 可选：把 Agent 切换为双人模式（2–4 人），见[双人模式](agent/README_zh.md#双人模式)。

两者之间的接口见 [API 文档](inference/docs/api_zh.md)，其他客户端也可以使用。

## 目录结构

```text
InWorld/
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── inference/            Murea-Fast 推理服务
│   ├── configs/          服务配置和示例请求
│   ├── docs/             部署、API、性能、来源说明
│   ├── scripts/          命令行入口、connector 合并、端到端检查
│   ├── src/murea_fast/   启动器和客户端
│   ├── tests/
│   └── vendor/sglang/    带模型实现的固定版本 SGLang
└── agent/                Murea Agent
    ├── murea_agent/      agent 本体（Python 包）
    ├── app/              FastAPI 服务端
    ├── frontend/         React 界面（构建产物在 frontend/dist）
    ├── docs/             双人模式设计说明
    └── tests/
```

## 参与贡献

提交 pull request 前各部分需要运行的检查见 [CONTRIBUTING_zh.md](CONTRIBUTING_zh.md)。

## 安全

推理服务和 agent 都不适合原样暴露在公网上。在共享网络中部署之前，请先阅读
[安全说明](SECURITY_zh.md)。

## 许可证

代码以 [Apache License 2.0](LICENSE) 发布。`inference/vendor/` 中的 SGLang 保留其自身的版权和许可证声明，
其中包括以 MIT 和 BSD-3-Clause 许可的文件。Murea-Fast 模型权重是 LTX-2.5 的衍生模型，受
[LTX-2.x Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x)
约束，详见[来源说明](inference/docs/provenance_zh.md)。agent 会调用第三方语言模型和图像模型 API，
这些服务另有各自的条款。
