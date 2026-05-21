#!/usr/bin/env bash
# startwebcam.sh – Start the PC webcam emotion detection client
# Usage:  bash startwebcam.sh  [camera_index]  [server_url]
#
# Examples:
#   bash startwebcam.sh                          # default camera 0, localhost
#   bash startwebcam.sh 1                        # camera index 1
#   bash startwebcam.sh 0 http://192.168.1.10:8000/predict  # remote server

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
