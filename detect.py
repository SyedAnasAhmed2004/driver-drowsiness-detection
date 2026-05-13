"""
Real-time Driver Distraction Detection
- Works with webcam feed OR a video file
- Overlays class label, confidence bar, and severity colour
- Press 'q' to quit, 's' to save a screenshot
"""

import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import os
import argparse

from config import CONFIG, CLASS_NAMES, SEVERITY, SEVERITY_COLOR
from model import DistractionDetector
from utils import load_checkpoint


# ─────────────────────────────────────────────
# INFERENCE TRANSFORM (no augmentation)
# ─────────────────────────────────────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────
# PREDICTOR CLASS
# ─────────────────────────────────────────────
class DistractionPredictor:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DistractionDetector(num_classes=CONFIG["num_classes"])
        self.model, _, _, _ = load_checkpoint(
            self.model, checkpoint_path, device=self.device)
        self.model.to(self.device).eval()
        print(f"[INFO] Predictor ready on {self.device}")

    def predict(self, frame_bgr: np.ndarray):
        """
        Args:
            frame_bgr: OpenCV BGR frame (H, W, 3)
        Returns:
            class_idx (int), confidence (float), probabilities (np.ndarray)
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = INFER_TRANSFORM(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        class_idx = int(probs.argmax())
        confidence = float(probs[class_idx])
        return class_idx, confidence, probs


# ─────────────────────────────────────────────
# FRAME ANNOTATION
# ─────────────────────────────────────────────
def annotate_frame(frame, class_idx, confidence, probs, fps):
    h, w = frame.shape[:2]
    severity = SEVERITY[class_idx]
    color = SEVERITY_COLOR[severity]      # BGR
    label = CLASS_NAMES[class_idx]
    threshold = CONFIG["confidence_threshold"]

    # ── Top banner ──────────────────────────────
    banner_h = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if confidence < threshold:
        display_label = "Uncertain — reposition camera"
        color = (180, 180, 180)
    else:
        display_label = label

    cv2.putText(frame, display_label,
                (12, 38), cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2, cv2.LINE_AA)

    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (w - 130, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ── Confidence bar ──────────────────────────
    bar_x, bar_y, bar_w, bar_h = 12, h - 30, 220, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    fill = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                  color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (120, 120, 120), 1)
    cv2.putText(frame, f"{confidence * 100:.1f}%",
                (bar_x + bar_w + 8, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # ── Severity indicator dot ───────────────────
    dot_labels = {0: "SAFE", 1: "MILD", 2: "DANGER"}
    cv2.circle(frame, (w - 24, h - 24), 16, color, -1)
    cv2.putText(frame, dot_labels[severity],
                (w - 90, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (220, 220, 220), 1)

    return frame


# ─────────────────────────────────────────────
# MAIN DETECTION LOOP
# ─────────────────────────────────────────────
def run_detector(source=0, checkpoint="checkpoints/best_model.pth"):
    predictor = DistractionPredictor(checkpoint)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    os.makedirs("results/screenshots", exist_ok=True)
    prev_time = time.time()
    frame_count = 0

    print("\n[INFO] Running detector. Press 'q' to quit, 's' to screenshot.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        # Predict every frame (throttle to every 2nd for speed if needed)
        class_idx, confidence, probs = predictor.predict(frame)

        # FPS calculation
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        frame = annotate_frame(frame, class_idx, confidence, probs, fps)
        cv2.imshow("Driver Distraction Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"results/screenshots/frame_{frame_count:04d}.jpg"
            cv2.imwrite(path, frame)
            print(f"[INFO] Screenshot saved → {path}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time driver distraction detector")
    parser.add_argument("--source", default=0,
                        help="0 = webcam | path to video file")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth",
                        help="Path to trained .pth checkpoint")
    args = parser.parse_args()

    source = args.source if args.source == 0 else args.source
    run_detector(source=source, checkpoint=args.checkpoint)
