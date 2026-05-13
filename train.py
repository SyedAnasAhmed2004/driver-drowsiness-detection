"""
Driver Distraction Detection - Training Script
Dataset: State Farm Distracted Driver Detection (Kaggle)
Model: MobileNetV2 Transfer Learning (fine-tuned)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset

from config import CONFIG
from model import DistractionDetector
from utils import save_checkpoint, plot_training_history, plot_confusion_matrix

# ─────────────────────────────────────────────
# 1. DATA TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def _get_base_dataset(transform):
    return datasets.ImageFolder(
        root=os.path.join(CONFIG["data_dir"], "train"),
        transform=transform
    )


def _sample_balanced_subset(dataset, samples_per_class, seed=42):
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    rng = np.random.RandomState(seed)
    selected_indices = []
    for label, indices in class_to_indices.items():
        if len(indices) < samples_per_class:
            raise ValueError(
                f"Class {dataset.classes[label]} has only {len(indices)} samples, "
                f"cannot select {samples_per_class} without replacement."
            )
        selected_indices.extend(rng.choice(indices, samples_per_class, replace=False).tolist())

    rng.shuffle(selected_indices)
    return Subset(dataset, selected_indices)


# ─────────────────────────────────────────────
# 2. LOAD DATASET
# ─────────────────────────────────────────────
def load_data(use_subset: bool = False, samples_per_class: int = None, seed: int = None):
    train_transform, val_transform = get_transforms()
    seed = seed if seed is not None else CONFIG["random_seed"]
    samples_per_class = samples_per_class if samples_per_class is not None else CONFIG["subset_samples_per_class"]

    full_dataset = _get_base_dataset(train_transform)
    if use_subset:
        full_dataset = _sample_balanced_subset(full_dataset, samples_per_class, seed)

    val_size = int(len(full_dataset) * CONFIG["val_split"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    # Apply val transform to validation subset
    base_dataset = getattr(val_dataset.dataset, "dataset", val_dataset.dataset)
    base_dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    class_names = getattr(full_dataset, "classes", None)
    if class_names is None:
        class_names = full_dataset.dataset.classes

    print(f"[INFO] Classes: {class_names}")
    print(f"[INFO] Train samples: {train_size} | Val samples: {val_size}")
    return train_loader, val_loader, class_names


# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


# ─────────────────────────────────────────────
# 4. MAIN TRAINING ENTRY POINT
# ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    train_loader, val_loader, class_names = load_data(
        use_subset=CONFIG["use_subset"],
        samples_per_class=CONFIG["subset_samples_per_class"],
        seed=CONFIG["random_seed"]
    )

    model = DistractionDetector(num_classes=CONFIG["num_classes"]).to(device)
    print(f"[INFO] Model: {model.__class__.__name__} loaded")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                           weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3,
                                  factor=0.5)

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print("\n[INFO] Starting training...\n")
    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, labels = validate(
            model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{CONFIG['epochs']}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc,
                            path=os.path.join(CONFIG["checkpoint_dir"], "best_model.pth"))
            print(f"  >> Best model saved (Val Acc: {val_acc:.2f}%)")

    print(f"\n[DONE] Best Validation Accuracy: {best_val_acc:.2f}%")

    # Final evaluation report
    _, _, final_preds, final_labels = validate(model, val_loader, criterion, device)
    print("\n[INFO] Classification Report:")
    print(classification_report(final_labels, final_preds,
                                target_names=class_names))

    plot_training_history(history,
                          save_path=os.path.join(CONFIG["results_dir"], "training_history.png"))
    plot_confusion_matrix(final_labels, final_preds, class_names,
                          save_path=os.path.join(CONFIG["results_dir"], "confusion_matrix.png"))


if __name__ == "__main__":
    main()
