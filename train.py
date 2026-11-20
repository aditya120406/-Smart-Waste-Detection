"""
Model Training Script for Smart Waste Classification System.
Trains a high-accuracy neural network on the waste dataset.
Uses a two-stage training strategy:
  Stage 1: Transfer Learning (frozen MobileNetV2 base + classification head)
  Stage 2: Fine-Tuning (unfreezes top convolutional blocks with low learning rate)

Usage:
    python train.py
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import create_dataset_pipeline, count_dataset_classes, IMAGE_SIZE
from src.model import build_waste_cnn, compile_model, unfreeze_top_layers


def plot_combined_history(history1, history2, save_path: str):
    """Plots combined accuracy and loss curves across Stage 1 and Stage 2."""
    acc = history1.history.get('accuracy', [])
    val_acc = history1.history.get('val_accuracy', [])
    loss = history1.history.get('loss', [])
    val_loss = history1.history.get('val_loss', [])

    if history2:
        acc = acc + history2.history.get('accuracy', [])
        val_acc = val_acc + history2.history.get('val_accuracy', [])
        loss = loss + history2.history.get('loss', [])
        val_loss = val_loss + history2.history.get('val_loss', [])

    epochs_range = range(1, len(acc) + 1)
    split_epoch = len(history1.history.get('accuracy', []))

    plt.figure(figsize=(13, 5))

    # Subplot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='#2e7d32', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#1565c0', linewidth=2, linestyle='--')
    if history2:
        plt.axvline(x=split_epoch, color='#e65100', linestyle=':', label='Fine-Tuning Start')
    plt.title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')

    # Subplot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='#c62828', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#e65100', linewidth=2, linestyle='--')
    if history2:
        plt.axvline(x=split_epoch, color='#e65100', linestyle=':', label='Fine-Tuning Start')
    plt.title('Training & Validation Loss', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Categorical Crossentropy)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Export] Saved training history plot to {save_path}")


def train(data_dir: str, stage1_epochs: int, stage2_epochs: int, batch_size: int, models_dir: str):
    print("=" * 70)
    print(" Smart Waste Classification System - High-Accuracy Training Pipeline")
    print(" Two-Stage Optimization: Feature Extraction + Top-Layer Fine Tuning")
    print("=" * 70)

    if not os.path.exists(data_dir):
        print(f"[Error] Dataset directory not found at: {data_dir}")
        sys.exit(1)

    class_counts = count_dataset_classes(data_dir)
    print(f"\n[Dataset Verified] Found {len(class_counts)} classes:")
    total_samples = sum(class_counts.values())
    for cls_name, cnt in class_counts.items():
        print(f"  - {cls_name:<12}: {cnt:>4} images")
    print(f"  Total samples: {total_samples}")

    os.makedirs(models_dir, exist_ok=True)

    print(f"\n[Pipeline] Constructing balanced dataset pipeline (image size={IMAGE_SIZE})...")
    train_ds, val_ds, class_names = create_dataset_pipeline(
        data_dir=data_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        validation_split=0.2
    )

    # Save class index mapping
    class_mapping = {idx: name for idx, name in enumerate(class_names)}
    mapping_path = os.path.join(models_dir, "class_indices.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"[Export] Saved class index mapping ({len(class_names)} classes) to {mapping_path}")

    # Compute balanced class weights to give minority classes (e.g. trash) equal learning priority
    class_weights = {}
    for idx, cls_name in enumerate(class_names):
        cnt = class_counts.get(cls_name, 1)
        class_weights[idx] = round(total_samples / (len(class_names) * cnt), 2)
    print("[Class Weights Computed for Balance]:", class_weights)

    import tensorflow as tf
    best_model_path = os.path.join(models_dir, "waste_classifier.keras")

    # =========================================================================
    # STAGE 1: Train Classification Head (Base Model Frozen)
    # =========================================================================
    print("\n" + "=" * 60)
    print(f" STAGE 1: Training Classification Head ({stage1_epochs} Epochs, LR=1e-3)")
    print("=" * 60)
    model = build_waste_cnn(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=len(class_names),
        dropout_rate=0.3,
        use_transfer_learning=True
    )
    compile_model(model, learning_rate=1e-3, label_smoothing=0.05)

    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        class_weight=class_weights,
        callbacks=callbacks_stage1
    )

    # =========================================================================
    # STAGE 2: Fine-Tuning Top Convolutional Layers (Low Learning Rate)
    # =========================================================================
    history2 = None
    if stage2_epochs > 0:
        print("\n" + "=" * 60)
        print(f" STAGE 2: Fine-Tuning Top 30 Layers ({stage2_epochs} Epochs, LR=5e-5)")
        print("=" * 60)
        unfreeze_top_layers(model, num_layers=30)
        compile_model(model, learning_rate=5e-5, label_smoothing=0.05)

        callbacks_stage2 = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
        ]

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=stage2_epochs,
            class_weight=class_weights,
            callbacks=callbacks_stage2
        )

    # Export training curves
    plot_path = os.path.join(models_dir, "training_history.png")
    plot_combined_history(history1, history2, plot_path)

    print("\n" + "=" * 70)
    print(f" High-accuracy training complete! Best model stored at: {best_model_path}")
    print(" Run 'python evaluate.py' to generate the updated evaluation report.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-accuracy waste classification training")
    parser.add_argument("--data-dir", type=str, default=os.path.join(BASE_DIR, "data", "dataset-resized"))
    parser.add_argument("--stage1-epochs", type=int, default=5, help="Epochs for feature extraction stage")
    parser.add_argument("--stage2-epochs", type=int, default=3, help="Epochs for top-layer fine-tuning stage")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--models-dir", type=str, default=os.path.join(BASE_DIR, "models"))
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        models_dir=args.models_dir
    )
