# 贡献指南

[English](CONTRIBUTING.md) | **中文**

本指南针对 agent，以下命令均在 `agent/` 目录下执行。

1. 按 [README](README_zh.md#快速开始) 准备好环境。
2. 提交改动前，先运行代码检查和离线测试：

   ```bash
   uvx ruff check . && uvx ruff format --check .
   .venv/bin/python tests/run_all.py
   cd frontend && npx tsc --noEmit && npm run lint && npm test
   ```

   构建好的前端（`frontend/dist`）提交在仓库里，用户无需安装 Node.js。
   修改前端后，运行 `python scripts/frontend.py build`，并把 `frontend/dist` 一起提交；
   如果构建产物比源码旧，CI 会失败。

   每个 `tests/test_*.py` 都是独立脚本；`run_all.py` 会用一份不含 API key 的临时配置逐个运行它们
   （直接运行单个测试会读到你自己的 `config.yaml`）。调用真实模型的测试放在 `tests/live/`，
   不在默认运行范围内。

   依赖声明在 `pyproject.toml` 中；`requirements.txt` 是所有安装方式共用的锁定文件。
   修改依赖后重新生成：
   `uv pip compile pyproject.toml -o requirements.txt --universal --python-version 3.12`。

3. 新增配置项时同步更新 `config.example.yaml`：部署时可以修改的每个选项都要列在那里，
   并附一行注释。代码只通过 `murea_agent/config.py` 读取配置。

4. 代码注释、docstring、异常信息和日志一律用英文。玩家看到的文字跟随 `agent.language`：
   后端用 `_say(zh, en)`（`murea_agent/util.py`），双人模式用 `msg()` 并在
   `murea_agent/coop/messages.py` 中添加条目；前端字符串用中文书写，在
   `frontend/src/i18n/en.ts` 中翻译。两种语言要一起添加，只在英文下出现的行为要在
   `tests/test_language.py` 中覆盖。

5. 修改提示词（`murea_agent/director/prompts/`）会改变拍出来的内容。请在 pull request
   中说明你尝试了什么，以及修改前后拍出的片段是什么样子。
