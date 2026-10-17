# Contributing

**English** | [中文](CONTRIBUTING_zh.md)

This guide covers the agent; run every command below from the `agent/` directory.

1. Set up as in the [README](README.md#quick-start).
2. Run the checks and the offline tests before sending a change:

   ```bash
   uvx ruff check . && uvx ruff format --check .
   .venv/bin/python tests/run_all.py
   cd frontend && npx tsc --noEmit && npm run lint && npm test
   ```

   The built front end (`frontend/dist`) is committed so users need no Node.js.
   After changing the front end, run `python scripts/frontend.py build` and commit
   `frontend/dist` with it; CI fails when the build is older than its sources.

   Every `tests/test_*.py` is a standalone script; `run_all.py` runs each one
   against a throw-away config with no API keys (running them directly would
   pick up your own `config.yaml`). Tests that call real models live in
   `tests/live/` and are not part of the default run.

   Dependencies are declared in `pyproject.toml`; `requirements.txt` is the
   lock everything installs from. Regenerate it after changing them:
   `uv pip compile pyproject.toml -o requirements.txt --universal --python-version 3.12`.

3. Keep `config.example.yaml` in sync when adding a setting: every option a
   deployment can change belongs there, with a one-line comment. Code reads
   settings only through `murea_agent/config.py`.

4. Write code comments, docstrings, exception messages and logs in English.
   Text a player sees follows `agent.language`: in the backend use
   `_say(zh, en)` (`murea_agent/util.py`) or, in co-op, `msg()` with an entry in
   `murea_agent/coop/messages.py`; in the frontend, strings are written in
   Chinese and translated in `frontend/src/i18n/en.ts`. Add both languages
   together, and cover English-only behaviour in `tests/test_language.py`.

5. Prompt changes (`murea_agent/director/prompts/`) change what gets filmed. Describe
   in the pull request what you tried and what the clips looked like before
   and after.
