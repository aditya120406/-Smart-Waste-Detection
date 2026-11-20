"""
Model Evaluation Script for Smart Waste Classification System.
Calculates Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

Usage:
    python evaluate.py
    python evaluate.py --data-dir data/dataset-resized --model models/waste_classifier.keras
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import IMAGE_SIZE


def plot_confusion_matrix(cm, class_names, save_path: str):
    """Generates and saves a clean confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Waste Classification - Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    # Label each cell with numeric value
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(
                j, i, f"{val}",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val > thresh else "black",
                fontweight='bold'
            )

    plt.tight_layout()
    plt.ylabel('Ground Truth Label', fontsize=11, fontweight='bold')
    plt.xlabel('Model Predicted Label', fontsize=11, fontweight='bold')
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Export] Saved confusion matrix visualization to {save_path}")


def evaluate(data_dir: str, model_path: str, models_dir: str):
    print("=" * 70)
    print(" Smart Waste Classification System - Evaluation Pipeline")
    print("=" * 70)

    if not os.path.exists(model_path):
        print(f"[Status] Trained model weights not found at: {model_path}")
        print("Model evaluation cannot be executed without trained weights.")
        print("Performance Metrics: [To be measured]")
        print("\nTo train the model:")
        print("  1. Prepare dataset in 'data/dataset-resized'")
        print("  2. Run 'python train.py'")
        print("  3. Re-run 'python evaluate.py'")
        sys.exit(0)

    if not os.path.exists(data_dir):
        print(f"[Error] Dataset directory not found at: {data_dir}")
        print("Performance Metrics: [To be measured]")
        sys.exit(1)

    import tensorflow as tf

    print(f"[Load] Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

    print(f"[Dataset] Loading evaluation data from {data_dir}...")
    # Load validation split for evaluation with identical seed and shuffle for balanced split
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=32,
        label_mode="categorical",
        shuffle=True
    )
    class_names = val_ds.class_names

    # Normalize pixels
    rescale_layer = tf.keras.layers.Rescaling(1.0 / 255.0)
    val_ds_normalized = val_ds.map(lambda x, y: (rescale_layer(x), y))

    print("[Inference] Generating predictions across evaluation samples...")
    y_true_indices = []
    y_pred_indices = []

    for images, labels in val_ds_normalized:
        preds = model.predict(images, verbose=0)
        y_true_indices.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_indices.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true_indices)
    y_pred = np.array(y_pred_indices)

    all_labels = list(range(len(class_names)))

    # Compute classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\n" + "=" * 70)
    print(" CLASSIFICATION PERFORMANCE METRICS")
    print("=" * 70)
    print(report_text)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    cm_path = os.path.join(models_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)

    # Export metrics JSON
    metrics_path = os.path.join(models_dir, "evaluation_report.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"[Export] Saved evaluation metrics JSON to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained CNN model on test/val dataset")
    parser.add_argument("--data-dir", type=str, default=os.path.join(BASE_DIR, "data", "dataset-resized"),
                        help="Path to TrashNet dataset folder")
    parser.add_argument("--model", type=str, default=os.path.join(BASE_DIR, "models", "waste_classifier.keras"),
                        help="Path to trained model weights")
    parser.add_argument("--models-dir", type=str, default=os.path.join(BASE_DIR, "models"),
                        help="Folder where evaluation artifacts will be written")
    args = parser.parse_args()

    evaluate(data_dir=args.data_dir, model_path=args.model, models_dir=args.models_dir)
