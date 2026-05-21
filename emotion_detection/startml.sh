#!/usr/bin/env bash
# startml.sh – Start the FastAPI ML inference server
# Usage:  bash startml.sh  [--port 8000]

set -e

PORT=${1:-8000}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$SCRIPT_DIR/ml_service/emotion_model.tflite"

echo "========================================"
echo "  Emotion Detection – ML Service"
echo "========================================"

if [ ! -f "$MODEL" ]; then
  echo ""
  echo "[ERROR] TFLite model not found at:"
  echo "        $MODEL"
  echo ""
  echo "  Train it first:"
  echo "    cd training"
  echo "    python train_model.py"
  echo "    python convert_to_tflite.py"
  echo "  Then copy emotion_model.tflite → ml_service/"
  echo ""
  exit 1
fi

echo "[INFO] Model found  ✓"
echo "[INFO] Starting server on port $PORT …"
echo "[INFO] API docs → http://localhost:$PORT/docs"
echo ""

cd "$SCRIPT_DIR"
uvicorn ml_service.app:app --host 0.0.0.0 --port "$PORT" --reload
