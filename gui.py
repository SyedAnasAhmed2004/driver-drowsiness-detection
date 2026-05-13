"""
Driver Distraction Detection — GUI
Inference interface supporting single images, video files, and live webcam.
"""

import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from torchvision import transforms

from config import CONFIG, CLASS_NAMES, SEVERITY, SEVERITY_COLOR
from model import DistractionDetector
from utils import load_checkpoint, GradCAM

# ── Inference transform (no augmentation) ─────────────────────────────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Colour palette ─────────────────────────────────────────────────────────────
BG_DARK   = "#12121e"
BG_CARD   = "#1c1c2e"
BG_PANEL  = "#252540"
ACCENT    = "#7c5cbf"
TXT_LIGHT = "#e8e8f0"
TXT_MUTED = "#6b7280"
COL_GREEN  = "#22c55e"
COL_ORANGE = "#f97316"
COL_RED    = "#ef4444"

SEV_HEX   = {0: COL_GREEN, 1: COL_ORANGE, 2: COL_RED}
SEV_LABEL = {0: "SAFE", 1: "MILD RISK", 2: "DANGER"}


# ═══════════════════════════════════════════════════════════════════════════════
# Predictor
# ═══════════════════════════════════════════════════════════════════════════════

class Predictor:
    def __init__(self, checkpoint_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DistractionDetector(num_classes=CONFIG["num_classes"])
        self.model, _, _, _ = load_checkpoint(
            self.model, checkpoint_path, device=self.device)
        self.model.to(self.device).eval()

        try:
            target_layer = self.model.model.features[-1]
            self.gradcam = GradCAM(self.model, target_layer)
        except Exception:
            self.gradcam = None

    def predict(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = INFER_TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1).squeeze().cpu().numpy()
        idx = int(probs.argmax())
        return idx, float(probs[idx]), probs

    def get_gradcam(self, frame_bgr: np.ndarray, class_idx: int) -> np.ndarray:
        if self.gradcam is None:
            return frame_bgr
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = INFER_TRANSFORM(Image.fromarray(rgb))
        heatmap = self.gradcam(tensor, class_idx)
        return GradCAM.overlay(frame_bgr, heatmap)


# ═══════════════════════════════════════════════════════════════════════════════
# Frame annotation helper
# ═══════════════════════════════════════════════════════════════════════════════

def annotate(frame: np.ndarray, class_idx: int, confidence: float) -> np.ndarray:
    out = frame.copy()
    color_bgr = SEVERITY_COLOR[SEVERITY[class_idx]]
    label = CLASS_NAMES[class_idx] if confidence >= CONFIG["confidence_threshold"] else "Uncertain"
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 46), (18, 18, 32), -1)
    cv2.putText(out, label,         (10, 32), cv2.FONT_HERSHEY_DUPLEX, 0.72, color_bgr, 2, cv2.LINE_AA)
    cv2.putText(out, f"{confidence*100:.1f}%", (w - 96, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.64, color_bgr, 1, cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Main application
# ═══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Driver Distraction Detector")
        self.configure(bg=BG_DARK)
        self.geometry("1160x720")
        self.minsize(900, 580)

        self.predictor: Predictor | None = None
        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self.current_frame: np.ndarray | None = None
        self._after_id = None
        self._prev_t = time.time()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=12, pady=(6, 12))
        self._build_sidebar(body)
        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True)
        self._build_canvas(right)
        self._build_results_bar(right)

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG_PANEL, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  Driver Distraction Detector",
                 font=("Segoe UI", 15, "bold"), bg=BG_PANEL, fg=TXT_LIGHT
                 ).pack(side="left", padx=8, pady=10)
        self._status_dot = tk.Label(hdr, text="● No model loaded",
                                    font=("Segoe UI", 10), bg=BG_PANEL, fg=COL_RED)
        self._status_dot.pack(side="right", padx=20)

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG_CARD, width=248)
        sb.pack(side="left", fill="y", padx=(0, 10))
        sb.pack_propagate(False)

        self._sec(sb, "MODEL")
        self._btn(sb, "📂  Load Checkpoint", self._load_checkpoint, ACCENT)
        self._model_lbl = tk.Label(sb, text="No checkpoint loaded",
                                   font=("Segoe UI", 9), bg=BG_CARD, fg=TXT_MUTED,
                                   wraplength=210, justify="left")
        self._model_lbl.pack(anchor="w", padx=16, pady=(2, 8))

        self._hsep(sb)
        self._sec(sb, "INPUT SOURCE")
        self._btn(sb, "🖼   Open Image",      self._open_image)
        self._btn(sb, "🎬  Open Video File",  self._open_video)
        self._btn(sb, "📷  Start Webcam",     self._start_webcam)
        self._stop_btn = self._btn(sb, "⏹   Stop",  self._stop,
                                   color="#3a3a52", state="disabled")

        self._hsep(sb)
        self._sec(sb, "OPTIONS")
        self._gradcam_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sb, text=" Show Grad-CAM heatmap",
                       variable=self._gradcam_var,
                       font=("Segoe UI", 10), bg=BG_CARD, fg=TXT_LIGHT,
                       selectcolor=BG_PANEL, activebackground=BG_CARD,
                       activeforeground=TXT_LIGHT, cursor="hand2"
                       ).pack(anchor="w", padx=16, pady=4)

        self._hsep(sb)
        self._btn(sb, "💾  Save Screenshot", self._save_screenshot)

    def _build_canvas(self, parent):
        self._canvas_frame = tk.Frame(parent, bg="#0a0a14")
        self._canvas_frame.pack(fill="both", expand=True)
        self._img_label = tk.Label(self._canvas_frame, bg="#0a0a14",
                                   text="Load a model then choose an input source",
                                   font=("Segoe UI", 13), fg=TXT_MUTED)
        self._img_label.pack(fill="both", expand=True)

    def _build_results_bar(self, parent):
        bar = tk.Frame(parent, bg=BG_CARD, height=108)
        bar.pack(fill="x", pady=(8, 0))
        bar.pack_propagate(False)

        # Severity badge
        badge_wrap = tk.Frame(bar, bg=BG_CARD, width=116)
        badge_wrap.pack(side="left", fill="y", padx=(14, 0), pady=10)
        badge_wrap.pack_propagate(False)
        self._badge = tk.Label(badge_wrap, text="—",
                               font=("Segoe UI", 13, "bold"),
                               bg=BG_PANEL, fg=TXT_MUTED, relief="flat")
        self._badge.pack(fill="both", expand=True)

        # Details
        det = tk.Frame(bar, bg=BG_CARD)
        det.pack(side="left", fill="both", expand=True, padx=14, pady=10)

        self._pred_lbl = tk.Label(det, text="Waiting for input…",
                                  font=("Segoe UI", 14, "bold"),
                                  bg=BG_CARD, fg=TXT_LIGHT, anchor="w")
        self._pred_lbl.pack(fill="x")

        conf_row = tk.Frame(det, bg=BG_CARD)
        conf_row.pack(fill="x", pady=(5, 0))
        tk.Label(conf_row, text="Confidence:", font=("Segoe UI", 10),
                 bg=BG_CARD, fg=TXT_MUTED).pack(side="left")
        self._conf_lbl = tk.Label(conf_row, text="—",
                                  font=("Segoe UI", 10, "bold"),
                                  bg=BG_CARD, fg=TXT_LIGHT)
        self._conf_lbl.pack(side="left", padx=(6, 0))

        self._conf_bar = tk.Canvas(det, height=10, bg="#2a2a40",
                                   highlightthickness=0)
        self._conf_bar.pack(fill="x", pady=(6, 0))

        self._fps_lbl = tk.Label(bar, text="", font=("Segoe UI", 9),
                                 bg=BG_CARD, fg=TXT_MUTED)
        self._fps_lbl.pack(side="right", padx=14, anchor="s", pady=12)

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _sec(self, parent, text):
        tk.Label(parent, text=text, font=("Segoe UI", 8, "bold"),
                 bg=BG_CARD, fg=TXT_MUTED).pack(anchor="w", padx=16, pady=(12, 2))

    def _hsep(self, parent):
        tk.Frame(parent, bg="#2e2e46", height=1).pack(fill="x", padx=16, pady=8)

    def _btn(self, parent, text, cmd, color=BG_PANEL, state="normal"):
        b = tk.Button(parent, text=text, command=cmd,
                      font=("Segoe UI", 10), bg=color, fg=TXT_LIGHT,
                      activebackground=ACCENT, activeforeground="white",
                      relief="flat", cursor="hand2", pady=7, state=state,
                      bd=0)
        b.pack(fill="x", padx=16, pady=3)
        return b

    # ── Display helpers ───────────────────────────────────────────────────────

    def _show_frame(self, frame_bgr: np.ndarray):
        self.current_frame = frame_bgr.copy()
        self._canvas_frame.update_idletasks()
        cw = max(self._canvas_frame.winfo_width(),  10)
        ch = max(self._canvas_frame.winfo_height(), 10)
        fh, fw = frame_bgr.shape[:2]
        scale = min(cw / fw, ch / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (nw, nh)), cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._img_label.configure(image=photo, text="")
        self._img_label.image = photo

    def _update_results(self, class_idx: int, confidence: float, fps: float | None = None):
        severity = SEVERITY[class_idx]
        col = SEV_HEX[severity]
        uncertain = confidence < CONFIG["confidence_threshold"]

        if uncertain:
            self._pred_lbl.configure(text="Uncertain — reposition camera", fg=TXT_MUTED)
            self._badge.configure(text="?", bg=BG_PANEL, fg=TXT_MUTED)
        else:
            self._pred_lbl.configure(text=CLASS_NAMES[class_idx], fg=col)
            self._badge.configure(text=SEV_LABEL[severity], bg=col, fg="white")

        self._conf_lbl.configure(text=f"{confidence*100:.1f}%", fg=col)

        self._conf_bar.update_idletasks()
        bw = self._conf_bar.winfo_width() or 300
        self._conf_bar.delete("all")
        self._conf_bar.create_rectangle(0, 0, bw, 10, fill="#2a2a40", outline="")
        self._conf_bar.create_rectangle(0, 0, int(bw * confidence), 10, fill=col, outline="")

        if fps is not None:
            self._fps_lbl.configure(text=f"FPS  {fps:.1f}")

    # ── Actions ───────────────────────────────────────────────────────────────

    def _load_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Select checkpoint",
            initialdir=CONFIG["checkpoint_dir"] if os.path.isdir(CONFIG["checkpoint_dir"]) else ".",
            filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.predictor = Predictor(path)
            self._status_dot.configure(text="● Model ready", fg=COL_GREEN)
            self._model_lbl.configure(
                text=f"{os.path.basename(path)}\n{CONFIG['backbone']}  ·  {CONFIG['num_classes']} classes")
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def _require_model(self) -> bool:
        if self.predictor is None:
            messagebox.showwarning("No model", "Please load a checkpoint first.")
            return False
        return True

    def _open_image(self):
        if not self._require_model():
            return
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")])
        if not path:
            return
        self._stop()
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Could not read image.")
            return
        class_idx, conf, _ = self.predictor.predict(frame)
        if self._gradcam_var.get():
            frame = self.predictor.get_gradcam(frame, class_idx)
        self._update_results(class_idx, conf)
        self._show_frame(annotate(frame, class_idx, conf))

    def _open_video(self):
        if not self._require_model():
            return
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        if not path:
            return
        self._stop()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file.")
            return
        self._start_loop()

    def _start_webcam(self):
        if not self._require_model():
            return
        self._stop()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.")
            return
        self._start_loop()

    def _start_loop(self):
        self.running = True
        self._prev_t = time.time()
        self._stop_btn.configure(state="normal")
        self._video_loop()

    def _video_loop(self):
        if not self.running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self._stop()
            return

        now = time.time()
        fps = 1.0 / max(now - self._prev_t, 1e-9)
        self._prev_t = now

        class_idx, conf, _ = self.predictor.predict(frame)
        if self._gradcam_var.get():
            frame = self.predictor.get_gradcam(frame, class_idx)
        self._update_results(class_idx, conf, fps)
        self._show_frame(annotate(frame, class_idx, conf))

        self._after_id = self.after(30, self._video_loop)

    def _stop(self):
        self.running = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self._stop_btn.configure(state="disabled")

    def _save_screenshot(self):
        if self.current_frame is None:
            messagebox.showinfo("Nothing to save", "No frame available yet.")
            return
        os.makedirs("results/screenshots", exist_ok=True)
        path = f"results/screenshots/screenshot_{int(time.time())}.jpg"
        cv2.imwrite(path, self.current_frame)
        messagebox.showinfo("Saved", f"Screenshot saved:\n{path}")

    def _on_close(self):
        self._stop()
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
