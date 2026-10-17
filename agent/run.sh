#!/usr/bin/env bash
# One command to install everything and start Murea-Mind.
#
#   ./run.sh            # http://127.0.0.1:8080 (host/port from config.yaml)
#
# Needs only uv (https://docs.astral.sh/uv/). Front-end development: run
# `cd frontend && npm install && npm run dev` in another terminal (http://127.0.0.1:5173,
# hot reload, proxies /api to the server).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env. Put your API keys and MUREA_URL in it to film clips."
fi

command -v uv >/dev/null || { echo "Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

[ -x .venv/bin/python ] || uv venv --python 3.12 -q
uv pip install -q -p .venv -r requirements.txt
# Clips need ffmpeg; without a system one, install a static build (~50 MB).
command -v ffmpeg >/dev/null || uv pip install -q -p .venv "imageio-ffmpeg>=0.5,<1"

# The repo ships the built front end. Rebuild only if you changed its sources (needs Node.js).
if ! .venv/bin/python scripts/frontend.py check 2>/dev/null; then
  if command -v npm >/dev/null; then
    echo "Building the front end..."
    .venv/bin/python scripts/frontend.py build
  else
    echo "Front-end sources changed but Node.js is not installed; using the bundled build."
  fi
fi

exec .venv/bin/python -m app
