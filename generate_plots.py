"""
Re-generates result plots by running a final validation pass on the saved checkpoint.
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report

from config import CONFIG
from model import DistractionDetector
from utils import load_checkpoint, plot_confusion_matrix

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    val_transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(
        root=os.path.join(CONFIG["data_dir"], "train"),
        transform=val_transform
    )
    val_size = int(len(full_dataset) * CONFIG["val_split"])
    train_size = len(full_dataset) - val_size
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=0)

    model = DistractionDetector(num_classes=CONFIG["num_classes"]).to(device)
    ckpt_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
    model, _, epoch, val_acc = load_checkpoint(model, ckpt_path, device=device)
    print(f"[INFO] Loaded best model from epoch {epoch} (Val Acc: {val_acc:.2f}%)")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    class_names = full_dataset.classes
    print("\n[INFO] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    plot_confusion_matrix(
        all_labels, all_preds, class_names,
        save_path=os.path.join(CONFIG["results_dir"], "confusion_matrix.png")
    )
    print("[DONE] Plots saved to results/")

if __name__ == "__main__":
    main()
