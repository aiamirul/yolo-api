#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Run ./start.sh to start the server."
