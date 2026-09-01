#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if curl --silent --fail http://127.0.0.1:8001/api/health >/dev/null 2>&1; then
  echo "E-commerce is already running at http://127.0.0.1:8001"
  open http://127.0.0.1:8001
  exit 0
fi

SHARED_VENV="$(cd "$PROJECT_DIR/.." && pwd)/.venv"

if [ -x "$SHARED_VENV/bin/python" ]; then
  VENV_DIR="$SHARED_VENV"
else
  VENV_DIR="$PROJECT_DIR/.venv"
fi

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if ! python -c "import fastapi, sqlalchemy, pandas, sklearn" >/dev/null 2>&1; then
  pip install -r requirements.txt
fi

if [ ! -f src/artifacts/conversion.joblib ] || [ ! -f src/artifacts/segments.joblib ]; then
  python src/main.py --train-demo
fi

echo "E-commerce is starting at http://127.0.0.1:8001"
open http://127.0.0.1:8001
python -m uvicorn main:app --app-dir src --host 127.0.0.1 --port 8001
