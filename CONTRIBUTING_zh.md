# 贡献指南

[English](CONTRIBUTING.md) | **中文**

nWorld 由两部分组成，各有各的检查。提交 pull request 前，请在你改动的那部分目录下运行它的检查。
对应目录有改动时，CI 会运行同样的检查。

## Murea Agent（`agent/`）

见 [Agent 贡献指南](agent/CONTRIBUTING_zh.md)：ruff、离线测试、前端类型检查、lint 和测试，以及随仓库提交的前端构建产物。

## Murea-Fast（`inference/`）

在 `inference/` 目录下运行，只依赖标准库：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m compileall -q src scripts tests
bash -n scripts/murea-fast
```

改动内置引擎（`inference/vendor/sglang/`）时还需要：

- 在每个改过的文件第一行加上修改声明
  （`# Modified by the nWorld authors from SGLang; see inference/docs/provenance.md.`），
  并在[来源说明](inference/docs/provenance_zh.md)及其英文版中为它加一行，同时更新其中的文件数；
- 在完整环境中用 `pytest` 运行所改代码对应的内置测试，例如
  `vendor/sglang/python/sglang/multimodal_gen/` 下的 `test/unit/test_ltx2_ordered_multiref.py`
  （环境见[部署](inference/docs/deployment_zh.md#环境)），并在 GPU 机器上运行 `tests/smoke_runtime.py`。
  CI 不运行这些测试。

可能影响生成视频的改动，例如采样、注意力、VAE 或续写流程，还需要在 GPU 上实际运行。
请在 pull request 中说明使用的配置（例如[端到端检查](inference/docs/deployment_zh.md#端到端检查)），
以及输出和耗时有什么变化。

## 文档

每份文档都有英文版和中文版（`X.md` 和 `X_zh.md`），请在同一个 pull request 中一起修改。

## 安全

请按 [SECURITY_zh.md](SECURITY_zh.md) 中的方式私下报告漏洞，不要发公开 issue。

## 许可证

与其余代码一样，贡献的内容以 [Apache License 2.0](LICENSE) 发布。
