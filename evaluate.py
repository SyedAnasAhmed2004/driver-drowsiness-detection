"""
Evaluate a saved checkpoint on the test set (or validation set).
Produces: accuracy, per-class report, confusion matrix, Grad-CAM samples.
"""

import os
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import CONFIG, CLASS_NAMES
from model import DistractionDetector
from utils import load_checkpoint, plot_confusion_matrix, GradCAM


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(checkpoint_path: str, data_split: str = "train"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(
        root=os.path.join(CONFIG["data_dir"], data_split),
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False,
                        num_workers=0, pin_memory=False)

    model = DistractionDetector(num_classes=CONFIG["num_classes"])
    model, _, _, _ = load_checkpoint(model, checkpoint_path, device=device)
    model.to(device).eval()

    all_preds, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100.0 * correct / total
    print(f"\n[RESULT] Overall Accuracy: {acc:.2f}%  ({correct}/{total})\n")

    from sklearn.metrics import classification_report
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    plot_confusion_matrix(
        all_labels, all_preds, CLASS_NAMES,
        save_path=os.path.join(CONFIG["results_dir"], "eval_confusion_matrix.png")
    )

    return model, dataset, device


# ─────────────────────────────────────────────
# GRAD-CAM VISUALISATION
# ─────────────────────────────────────────────
def visualise_gradcam(model, dataset, device, n_samples: int = 10):
    """
    Display Grad-CAM heatmaps for n_samples random images from the dataset.
    The heatmap shows which pixels drove the model's prediction.
    """
    from config import CLASS_NAMES
    # Target the last conv layer of MobileNetV2
    target_layer = model.model.features[-1]
    cam = GradCAM(model, target_layer)

    indices = np.random.choice(len(dataset), n_samples, replace=False)

    cols = 5
    rows = (n_samples + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    fig.suptitle("Grad-CAM: regions that influenced the prediction",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)

    for i, idx in enumerate(indices):
        img_tensor, true_label = dataset[idx]

        # De-normalise for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_display = (img_tensor * std + mean).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)

        # Predict
        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0).to(device))
            pred_label = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()

        # Grad-CAM heatmap
        heatmap = cam(img_tensor.to(device), class_idx=pred_label)

        import cv2
        img_uint8 = (img_display * 255).astype(np.uint8)
        overlay = GradCAM.overlay(img_uint8, heatmap, alpha=0.45)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.imshow(overlay_rgb)
        correct = pred_label == true_label
        title_color = "green" if correct else "red"
        short_pred = CLASS_NAMES[pred_label].split(":")[0]
        short_true = CLASS_NAMES[true_label].split(":")[0]
        ax.set_title(f"Pred: {short_pred}\nTrue: {short_true} ({confidence*100:.0f}%)",
                     fontsize=8, color=title_color)
        ax.axis("off")

    save_path = os.path.join(CONFIG["results_dir"], "gradcam_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Grad-CAM visualisation saved -> {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--split", default="train",
                        help="Which data split to evaluate on: train | test")
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate Grad-CAM visualisations after evaluation")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of Grad-CAM samples to display")
    args = parser.parse_args()

    model, dataset, device = evaluate(args.checkpoint, args.split)
    if args.gradcam:
        visualise_gradcam(model, dataset, device, n_samples=args.n_samples)
