#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -f test_bus.jpg ]; then
    echo "Downloading test image..."
    curl -sL -o test_bus.jpg https://ultralytics.com/images/bus.jpg
fi

python3 -c "
import base64, json
with open('test_bus.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()
print(json.dumps({'image': b64}))
" > /tmp/payload.json

echo "=== Health Check ==="
curl -s http://127.0.0.1:5000/health | python3 -m json.tool

echo ""
echo "=== Predict (test_bus.jpg) ==="
curl -s -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/payload.json | python3 -m json.tool

rm -f /tmp/payload.json
