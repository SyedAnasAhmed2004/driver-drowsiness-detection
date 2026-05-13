# Driver Distraction Detection System
### Deep Learning | MobileNetV2 Transfer Learning | PyTorch + OpenCV

Detects 10 classes of driver behaviour from a dashboard camera feed in real time,
overlays confidence scores and colour-coded severity alerts on the video.

---

## Project Structure

```
driver_distraction/
├── config.py          ← all hyperparameters, paths, class names
├── model.py           ← MobileNetV2 / ResNet50 transfer-learning wrapper
├── train.py           ← training loop, data loading, scheduler
├── detect.py          ← real-time webcam / video inference
├── evaluate.py        ← test-set evaluation + Grad-CAM visualisation
├── utils.py           ← checkpointing, plots, Grad-CAM class
├── requirements.txt
└── README.md
```

---

## 10 Detection Classes (State Farm)

| ID  | Label                     | Severity  |
|-----|---------------------------|-----------|
| c0  | Safe driving              | ✅ Safe   |
| c1  | Texting — right hand      | 🔴 High   |
| c2  | Phone call — right hand   | 🔴 High   |
| c3  | Texting — left hand       | 🔴 High   |
| c4  | Phone call — left hand    | 🔴 High   |
| c5  | Radio / stereo            | 🟠 Mild   |
| c6  | Drinking                  | 🟠 Mild   |
| c7  | Reaching behind           | 🔴 High   |
| c8  | Hair / makeup             | 🟠 Mild   |
| c9  | Talking to passenger      | 🟠 Mild   |

---

## Setup

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Download the dataset

1. Go to https://www.kaggle.com/c/state-farm-distracted-driver-detection
2. Accept the competition rules and download **imgs.zip**
3. Unzip so the folder layout is:

```
data/
└── train/
    ├── c0/   (safe driving images)
    ├── c1/
    ├── ...
    └── c9/
```

---

## Training

```bash
python train.py
```

What happens:
- Loads MobileNetV2 pretrained on ImageNet (frozen backbone)
- Trains only the new classifier head for the first epochs
- Saves the best checkpoint to `checkpoints/best_model.pth`
- Saves loss/accuracy curves to `results/training_history.png`
- Prints a full per-class classification report at the end

**Want higher accuracy?** After the first training run, open `train.py` and add
`model.unfreeze_backbone(20)` then re-run with a lower learning rate (1e-5).
This is called two-stage fine-tuning.

**Switch to ResNet50** (higher accuracy, slower):

```python
# in config.py
"backbone": "resnet50"
```

---

## Real-Time Detection

**Webcam:**
```bash
python detect.py --checkpoint checkpoints/best_model.pth --source 0
```

**Video file:**
```bash
python detect.py --checkpoint checkpoints/best_model.pth --source path/to/video.mp4
```

Controls while running:
- `q` → quit
- `s` → save a screenshot to `results/screenshots/`

---

## Evaluation + Grad-CAM

```bash
# Basic evaluation metrics + confusion matrix
python evaluate.py --checkpoint checkpoints/best_model.pth --split train

# Also generate Grad-CAM heatmaps (shows WHAT the model looks at)
python evaluate.py --checkpoint checkpoints/best_model.pth --gradcam --n_samples 15
```

Grad-CAM output is saved to `results/gradcam_samples.png`.
Green title = correct prediction, red = wrong.

---

## Expected Results

| Backbone    | Val Accuracy  | Inference speed (CPU) |
|-------------|---------------|-----------------------|
| MobileNetV2 | ~96–98 %      | ~15–20 FPS            |
| ResNet50    | ~97–99 %      | ~8–12 FPS             |

*(Results vary with train/val split and augmentation settings.)*

---

## Tips for Your IPCV Report

- **Model choice**: explain why MobileNetV2 suits embedded/real-time use
- **Transfer learning**: discuss why ImageNet weights help even for in-car images
- **Grad-CAM**: use the heatmaps to show the model attends to hands/phone, not background
- **Class imbalance**: check class sizes in the dataset; discuss whether augmentation compensates
- **Severity mapping**: you added domain knowledge on top of the classifier — discuss that design decision
