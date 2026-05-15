#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Kill whatever is on port 5000
fuser -k 5000/tcp 2>/dev/null || true

# Also kill by PID file
if [ -f server.pid ]; then
    PID=$(cat server.pid)
    kill "$PID" 2>/dev/null || true
fi

# Wait for port to be free (up to 10s)
for i in $(seq 1 20); do
    fuser 5000/tcp >/dev/null 2>&1 || break
    sleep 0.5
done

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
