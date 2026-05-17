"""
FastAPI backend for Driver Distraction Detection.
- POST /api/predict  — accepts a JPEG frame, returns prediction JSON
- GET  /video_feed   — MJPEG stream from server-side webcam (optional)
- GET  /api/health   — health check
- GET  /api/config   — model config + class names
- POST /api/threshold/{t} — update confidence threshold
- GET  /api/stats    — session statistics
- GET  /api/alerts   — recent high-risk alerts
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from collections import deque
from datetime import datetime

# ── make parent directory importable ─────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import DistractionDetector
from config import CONFIG, CLASS_NAMES, SEVERITY
from utils import load_checkpoint

# ── app setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Driver Distraction Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── inference transform (must match detect.py / train.py) ─────────────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── shared state ──────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.model = None
        self.confidence_threshold = CONFIG.get("confidence_threshold", 0.6)
        self.alerts = deque(maxlen=50)
        self.stats = {
            "total_frames": 0,
            "detections_by_class": {str(i): 0 for i in range(10)},
            "high_risk_count": 0,
        }


state = AppState()


@app.on_event("startup")
async def startup():
    print("[INFO] Loading model…")
    state.model = DistractionDetector(num_classes=CONFIG["num_classes"])
    checkpoint_path = os.path.join(ROOT, CONFIG["checkpoint_dir"], "best_model.pth")
    if os.path.exists(checkpoint_path):
        state.model, _, _, _ = load_checkpoint(
            state.model, checkpoint_path, device="cpu"
        )
        print(f"[INFO] Checkpoint loaded from {checkpoint_path}")
    else:
        print(f"[WARN] No checkpoint at {checkpoint_path} — using random weights")
    state.model.eval()


# ── helpers ───────────────────────────────────────────────────────────────────
def run_inference(frame_bgr: np.ndarray):
    """
    Run model on a BGR frame.
    Returns (class_idx, confidence, all_probs_list).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = INFER_TRANSFORM(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(state.model(tensor), dim=1).squeeze().cpu().numpy()
    class_idx = int(probs.argmax())
    return class_idx, float(probs[class_idx]), probs.tolist()


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": state.model is not None}


@app.get("/api/config")
async def get_config():
    return {
        "num_classes": CONFIG["num_classes"],
        "class_names": CLASS_NAMES,
        "severity": SEVERITY,
        "confidence_threshold": state.confidence_threshold,
    }


@app.post("/api/threshold/{threshold}")
async def set_threshold(threshold: float):
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be 0–1")
    state.confidence_threshold = threshold
    return {"threshold": threshold, "status": "updated"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a JPEG/PNG image (one webcam frame) and return prediction JSON.
    Called by the React frontend ~5 times per second.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    class_idx, confidence, all_probs = run_inference(frame)
    severity = SEVERITY.get(class_idx, 0)
    uncertain = confidence < state.confidence_threshold

    # update session stats
    state.stats["total_frames"] += 1
    state.stats["detections_by_class"][str(class_idx)] += 1
    if severity == 2:
        state.stats["high_risk_count"] += 1

    result = {
        "class": class_idx,
        "class_name": CLASS_NAMES[class_idx],
        "confidence": round(confidence, 4),
        "severity": severity,
        "uncertain": uncertain,
        "all_probs": [round(p, 4) for p in all_probs],
        "timestamp": datetime.now().isoformat(),
    }

    if severity == 2 and not uncertain:
        state.alerts.append({**result, "message": f"High Risk: {CLASS_NAMES[class_idx]}"})

    return result


@app.get("/api/stats")
async def get_stats():
    return state.stats


@app.get("/api/alerts")
async def get_alerts(limit: int = 20):
    return {"alerts": list(state.alerts)[-limit:], "total": len(state.alerts)}


# ── optional MJPEG stream (server webcam) ─────────────────────────────────────
def _generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if state.model is not None:
                idx, conf, _ = run_inference(frame)
                sev = SEVERITY.get(idx, 0)
                color = (0, 220, 80) if sev == 0 else (0, 165, 255) if sev == 1 else (0, 0, 220)
                label = CLASS_NAMES[idx] if conf >= state.confidence_threshold else "Uncertain"
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (18, 18, 30), -1)
                cv2.putText(frame, label, (12, 33),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"{conf*100:.1f}%", (frame.shape[1] - 90, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
    finally:
        cap.release()


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
