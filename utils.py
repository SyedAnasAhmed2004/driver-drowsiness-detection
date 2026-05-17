"""
Utility functions: saving/loading checkpoints, plotting, evaluation helpers.
"""

import os
import numpy as np
import torch


# ─────────────────────────────────────────────
# CHECKPOINTING
# ─────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_acc, path: str):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
    }, path)


def load_checkpoint(model, path: str, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    val_acc = checkpoint.get("val_acc", 0.0)
    print(f"[INFO] Loaded checkpoint from epoch {epoch} (Val Acc: {val_acc:.2f}%)")
    return model, optimizer, epoch, val_acc


# ─────────────────────────────────────────────
# TRAINING HISTORY PLOT
# ─────────────────────────────────────────────

def plot_training_history(history: dict, save_path: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", color="#E8593C", linewidth=2)
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="#3B8BD4", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train Acc", color="#E8593C", linewidth=2)
    axes[1].plot(history["val_acc"],   label="Val Acc",   color="#3B8BD4", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Training history saved -> {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────

def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Shorten labels for readability
    short_names = [c.split(":")[0] for c in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Normalized"],
        ["d", ".2f"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=short_names, yticklabels=short_names,
                    ax=ax, linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Confusion matrix saved -> {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# CLASS ACTIVATION MAP (Grad-CAM)
# ─────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Highlights which regions of an image triggered the prediction.

    Usage:
        cam = GradCAM(model, target_layer=model.model.features[-1])
        heatmap = cam(image_tensor, class_idx)
        overlay = cam.overlay(original_image_np, heatmap)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, image_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0).requires_grad_(True)

        output = self.model(image_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Pool gradients over spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().numpy()

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    @staticmethod
    def overlay(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        import cv2
        heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        heatmap_color = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)
        overlaid = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
        return overlaid
