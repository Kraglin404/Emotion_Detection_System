#!/usr/bin/env bash

CAMERA=${1:-0}
SERVER=${2:-http://localhost:8000/predict}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "  Emotion Detection – Webcam Client"
echo "========================================"
echo "[INFO] Camera index : $CAMERA"
echo "[INFO] ML service   : $SERVER"
echo "[INFO] Controls     : q=quit  s=save snapshot  p=pause/resume"
echo ""

cd "$SCRIPT_DIR"
python webcam_client.py --camera "$CAMERA" --server "$SERVER"
