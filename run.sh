#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
python data/generate_data.py
python models/train_model.py

cd "$(dirname "$0")"

cleanup() {
	if [[ -n "${BACKEND_PID:-}" ]]; then
		kill "$BACKEND_PID" >/dev/null 2>&1 || true
	fi
	if [[ -n "${FRONTEND_PID:-}" ]]; then
		kill "$FRONTEND_PID" >/dev/null 2>&1 || true
	fi
}

trap cleanup EXIT INT TERM

python -m uvicorn api:app --reload --port 8000 &
BACKEND_PID=$!

(
	cd frontend
	npm run dev
) &
FRONTEND_PID=$!

echo "Backend  -> http://localhost:8000/docs"
echo "Frontend -> http://localhost:3000"

wait "$BACKEND_PID" "$FRONTEND_PID"
