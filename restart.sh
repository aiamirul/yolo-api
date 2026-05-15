#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ -f server.pid ]; then
    PID=$(cat server.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping server (PID $PID)..."
        kill "$PID"
        sleep 2
    fi
fi

if [ -f config.json ]; then
    CFG_MODEL=$(python3 -c "import json; print(json.load(open('config.json')).get('model',''))" 2>/dev/null || true)
    export YOLO_MODEL="${CFG_MODEL:-yolov8n}"
else
    export YOLO_MODEL="${YOLO_MODEL:-yolov8n}"
fi
export PORT="${PORT:-5000}"

source .venv/bin/activate
nohup python app.py > server.log 2>&1 &
echo $! > server.pid
echo "Server started (PID $!). Logs: server.log"
