"""
Central configuration for Driver Distraction Detection project.
Edit these values to match your environment before running.
"""

CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    # Download the State Farm dataset from Kaggle and point this to the folder
    # that contains a 'train/' sub-directory with class sub-folders (c0..c9).
    "data_dir":        "data/",
    "checkpoint_dir":  "checkpoints/",
    "results_dir":     "results/",

    # ── Model ──────────────────────────────────────────────────────────────
    "num_classes":     10,          # c0 (safe) … c9 (10 distraction classes)
    "img_size":        224,         # MobileNetV2 / ResNet standard input size
    "backbone":        "mobilenetv2",  # choices: mobilenetv2 | resnet50

    # ── Training ───────────────────────────────────────────────────────────
    "epochs":          20,
    "batch_size":      32,
    "learning_rate":   1e-4,
    "val_split":       0.2,         # fraction of train set held out for val

    # ── Inference ──────────────────────────────────────────────────────────
    "confidence_threshold": 0.6,    # below this → show "Uncertain" warning
}

# Human-readable labels for all 10 State Farm classes
CLASS_NAMES = [
    "c0: Safe driving",
    "c1: Texting (right hand)",
    "c2: Phone call (right hand)",
    "c3: Texting (left hand)",
    "c4: Phone call (left hand)",
    "c5: Radio / adjusting stereo",
    "c6: Drinking",
    "c7: Reaching behind",
    "c8: Hair / makeup",
    "c9: Talking to passenger",
]

# Alert severity per class (used by the real-time detector)
# 0 = safe  |  1 = mild distraction  |  2 = high distraction
SEVERITY = {
    0: 0,   # safe driving
    1: 2,   # texting  – high risk
    2: 2,   # phone call
    3: 2,   # texting
    4: 2,   # phone call
    5: 1,   # stereo
    6: 1,   # drinking
    7: 2,   # reaching behind
    8: 1,   # grooming
    9: 1,   # talking to passenger
}

SEVERITY_COLOR = {
    0: (0, 220, 80),    # green  – BGR for OpenCV
    1: (0, 165, 255),   # orange
    2: (0, 0, 220),     # red
}
