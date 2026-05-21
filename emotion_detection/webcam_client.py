"""
webcam_client.py
----------------
PC webcam client for real-time emotion detection.
Captures frames from your camera, sends each face to the ML service,
and overlays the predicted emotion on the live video window.

Prerequisites:
    - ML service must be running (python ml_service/app.py or bash startml.sh)
    - pip install opencv-python requests

Usage:
    python webcam_client.py [--camera 0] [--server http://localhost:8000]

Controls (in the video window):
    q  → quit
    s  → save current frame as snapshot.jpg
    p  → pause / resume
"""

import argparse
import base64
import sys
import time

import cv2
import requests

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_API = "http://localhost:8000/predict"
DEFAULT_CAM = 0   # 0 = built-in webcam; try 1,2… for external cameras

# Haar cascade for face detection (bundled with OpenCV)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Emotion → colour mapping (BGR)
EMOTION_COLORS = {
    "Angry":    (0,   0,   220),
    "Disgust":  (0,   140, 0),
    "Fear":     (128, 0,   128),
    "Happy":    (0,   200, 200),
    "Sad":      (200, 100, 0),
    "Surprise": (0,   165, 255),
    "Neutral":  (180, 180, 180),
}
DEFAULT_COLOR = (255, 255, 255)


# ── Helpers ───────────────────────────────────────────────────────────────────
def encode_frame(frame) -> str:
    """Encode an OpenCV BGR frame to a base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def call_api(b64_img: str, api_url: str, timeout: float = 5.0) -> dict | None:
    """POST the base64 image to the ML service. Returns parsed JSON or None."""
    try:
        resp = requests.post(api_url, json={"image": b64_img}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print("[WARN] Cannot connect to ML service. Is it running?")
    except requests.exceptions.Timeout:
        print("[WARN] API request timed out.")
    except requests.exceptions.RequestException as exc:
        print(f"[WARN] API error: {exc}")
    return None


def detect_faces(frame):
    """Return list of (x, y, w, h) bounding boxes for detected faces."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )
    return faces if len(faces) else []


def draw_overlay(frame, x, y, w, h, emotion: str, confidence: float):
    """Draw bounding box + emotion label on frame."""
    color = EMOTION_COLORS.get(emotion, DEFAULT_COLOR)

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label background
    label      = f"{emotion}  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    label_y    = max(y - 10, th + 10)
    cv2.rectangle(frame,
                  (x, label_y - th - 8),
                  (x + tw + 8, label_y + 4),
                  color, cv2.FILLED)

    text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
    cv2.putText(frame, label,
                (x + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)


def draw_bar_chart(frame, scores: dict, x0=10, y0=None, bar_w=120):
    """Draw a small confidence bar chart in the top-left corner."""
    if y0 is None:
        y0 = frame.shape[0] - len(scores) * 22 - 10
    emotions_sorted = sorted(scores.items(), key=lambda kv: -kv[1])
    for i, (emo, score) in enumerate(emotions_sorted):
        y = y0 + i * 22
        color = EMOTION_COLORS.get(emo, DEFAULT_COLOR)
        filled = int(score * bar_w)
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + 16), (50, 50, 50), cv2.FILLED)
        cv2.rectangle(frame, (x0, y), (x0 + filled, y + 16), color, cv2.FILLED)
        cv2.putText(frame, f"{emo[:3]} {score:.0%}",
                    (x0 + bar_w + 5, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(camera_index: int, api_url: str):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera_index}.")
        sys.exit(1)

    # Try to set a decent resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"[INFO] Camera opened  (index={camera_index})")
    print(f"[INFO] ML service URL: {api_url}")
    print("[INFO] Controls: q=quit  s=save  p=pause/resume")

    # State
    last_emotion  : str   = "—"
    last_conf     : float = 0.0
    last_scores   : dict  = {}
    last_api_time : float = 0.0
    api_interval  : float = 0.1   # seconds between API calls (≈10 fps)
    paused        : bool  = False
    frame_count   : int   = 0
    fps_time      : float = time.time()
    fps           : float = 0.0

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("snapshot.jpg", frame if "frame" in dir() else np.zeros((1,1,3), np.uint8))
            print("[INFO] Snapshot saved → snapshot.jpg")
        elif key == ord("p"):
            paused = not paused
            print("[INFO] " + ("Paused." if paused else "Resumed."))

        if paused:
            continue

        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed. Retrying …")
            time.sleep(0.05)
            continue

        frame_count += 1
        now = time.time()

        # FPS calculation (rolling over 30 frames)
        if frame_count % 30 == 0:
            fps = 30 / (now - fps_time)
            fps_time = now

        # ── Face detection + API call ─────────────────────────────────────
        faces = detect_faces(frame)

        if len(faces) > 0 and (now - last_api_time) >= api_interval:
            # Pick the largest face
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            face_crop   = frame[y:y+h, x:x+w]
            b64         = encode_frame(face_crop)
            result      = call_api(b64, api_url)
            if result:
                last_emotion  = result.get("prediction", "—")
                last_conf     = result.get("confidence", 0.0)
                last_scores   = result.get("all_scores", {})
                last_api_time = now

        # ── Draw overlays ─────────────────────────────────────────────────
        for (x, y, w, h) in faces:
            draw_overlay(frame, x, y, w, h, last_emotion, last_conf)

        if last_scores:
            draw_bar_chart(frame, last_scores)

        # HUD: FPS + status
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (frame.shape[1] - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1, cv2.LINE_AA)
        if len(faces) == 0:
            cv2.putText(frame, "No face detected",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2, cv2.LINE_AA)

        cv2.imshow("Emotion Detection  [q=quit  s=save  p=pause]", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam client stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam emotion detection client")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAM,
                        help="Camera index (default: 0)")
    parser.add_argument("--server", type=str, default=DEFAULT_API,
                        help="ML service URL (default: http://localhost:8000/predict)")
    args = parser.parse_args()

    run(args.camera, args.server)
