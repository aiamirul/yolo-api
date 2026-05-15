#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

export YOLO_MODEL="${YOLO_MODEL:-yolov8n}"
export PORT="${PORT:-5000}"

source .venv/bin/activate
python app.py
