#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt

from datasets import TreeSubsetDataset, get_transform
from models import get_model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, device):
    """Evaluate model on test data and return loss, accuracy, predictions, and labels."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = criterion(output, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def per_class_accuracy(y_true, y_pred, num_classes):
    """Compute accuracy for each class individually."""
    correct = Counter()
    total = Counter()
    for yt, yp in zip(y_true, y_pred):
        total[yt] += 1
        if yt == yp:
            correct[yt] += 1
    return [correct[i] / total[i] if total[i] > 0 else 0 for i in range(num_classes)]


def compute_prf(y_true, y_pred, num_classes):
    """Compute per-class precision, recall, and F1 scores."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), zero_division=0
    )
    return precision.tolist(), recall.tolist(), f1.tolist()


def save_results(output_dir, name, cm, loss, acc, class_acc, precision, recall, f1):
    """Save metrics to disk and return them as a dictionary."""
    np.save(os.path.join(output_dir, f"confusion_{name}.npy"), cm)
    return {
        f"{name}_loss": float(loss),
        f"{name}_accuracy": float(acc),
        f"{name}_class_accuracy": [float(x) for x in class_acc],
        f"{name}_precision": [float(x) for x in precision],
        f"{name}_recall": [float(x) for x in recall],
        f"{name}_f1": [float(x) for x in f1],
    }


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load experiment metadata
    with open(os.path.join(args.experiment_dir, "metadata.json")) as f:
        meta = json.load(f)

    # Prepare dataset and dataloader
    transform = get_transform(model_type=meta["model"])
    test_ds = TreeSubsetDataset(
        meta["dataset"], root="./data", train=False,
        transform=transform, class_names=meta["classes"]
    )
    test_loader = DataLoader(test_ds, batch_size=32)
    num_classes = len(meta["classes"])

    # -----------------------------------------------------------------------
    # Federated model evaluation
    # -----------------------------------------------------------------------
    fed_model = get_model(meta["model"], num_classes, pretrained=False)
    fed_model.load_state_dict(
        torch.load(os.path.join(args.experiment_dir, "fed_model.pt"), map_location=device)
    )
    fed_model.to(device)

    fed_loss, fed_acc, fed_preds, fed_labels = evaluate(fed_model, test_loader, device)
    fed_class_acc = per_class_accuracy(fed_labels, fed_preds, num_classes)
    cm_fed = confusion_matrix(fed_labels, fed_preds, labels=range(num_classes), normalize="true")
    fed_precision, fed_recall, fed_f1 = compute_prf(fed_labels, fed_preds, num_classes)

    # -----------------------------------------------------------------------
    # Centralized model evaluation
    # -----------------------------------------------------------------------
    central_model = get_model(meta["model"], num_classes, pretrained=False)
    central_model.load_state_dict(
        torch.load(os.path.join(args.experiment_dir, "central_model.pt"), map_location=device)
    )
    central_model.to(device)

    central_loss, central_acc, central_preds, central_labels = evaluate(central_model, test_loader, device)
    central_class_acc = per_class_accuracy(central_labels, central_preds, num_classes)
    cm_central = confusion_matrix(central_labels, central_preds, labels=range(num_classes), normalize="true")
    central_precision, central_recall, central_f1 = compute_prf(central_labels, central_preds, num_classes)

    # -----------------------------------------------------------------------
    # Print summary results
    # -----------------------------------------------------------------------
    print(f"\nFederated   -> Loss: {fed_loss:.4f} | Accuracy: {fed_acc*100:.2f}%")
    print(f"Centralized -> Loss: {central_loss:.4f} | Accuracy: {central_acc*100:.2f}%\n")

    print("Per-Class Accuracy (Fed vs Central):")
    for i, cls in enumerate(meta["classes"]):
        print(f"{cls:20s} | Fed: {fed_class_acc[i]*100:6.2f}% | Central: {central_class_acc[i]*100:6.2f}%")

    # -----------------------------------------------------------------------
    # Save all metrics and confusion matrices
    # -----------------------------------------------------------------------
    results = {}
    results.update(save_results(
        args.experiment_dir, "fed", cm_fed, fed_loss, fed_acc, fed_class_acc,
        fed_precision, fed_recall, fed_f1
    ))
    results.update(save_results(
        args.experiment_dir, "central", cm_central, central_loss, central_acc,
        central_class_acc, central_precision, central_recall, central_f1
    ))

    with open(os.path.join(args.experiment_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to:", os.path.join(args.experiment_dir, "results.json"))

    # -----------------------------------------------------------------------
    # Generate plots (if requested)
    # -----------------------------------------------------------------------
    if args.plot:
        plots_dir = os.path.join(args.experiment_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Plot class-wise accuracy comparison
        x = np.arange(num_classes)
        width = 0.35
        plt.figure(figsize=(8, 5))
        plt.bar(x - width/2, fed_class_acc, width, label="Federated")
        plt.bar(x + width/2, central_class_acc, width, label="Centralized")
        plt.xticks(x, meta["classes"], rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Class-wise Accuracy")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "class_accuracy.png"))
        plt.close()

        # Plot confusion matrix for federated model
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_fed, display_labels=meta["classes"])
        disp1.plot(ax=ax1, xticks_rotation=45, cmap="Blues")
        plt.title("Federated Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "confusion_fed.png"))
        plt.close(fig1)

        # Plot confusion matrix for centralized model
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_central, display_labels=meta["classes"])
        disp2.plot(ax=ax2, xticks_rotation=45, cmap="Oranges")
        plt.title("Centralized Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "confusion_central.png"))
        plt.close(fig2)

        print(f"Plots saved to: {plots_dir}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    main(args)
