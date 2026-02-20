#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Concrete crack segmentation (U-Net + ResNet34 backbone) using patch-based training.

This script:
  - Loads RGB images and corresponding binary masks
  - Center-crops to multiples of patch size
  - Patchifies into non-overlapping 512×512 tiles
  - (Optional) keeps only tiles that contain at least N crack pixels
  - Trains a U-Net (segmentation_models) with BCE+Jaccard loss
  - Saves best checkpoint, SavedModel, .h5, and a small config.json
  - Includes two small demos:
      * predict on one random test patch
      * predict on an arbitrary full-size image (center-cropped to multiples)
"""

import os
import glob
import json
import datetime
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import normalize  # unused (kept for parity)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from patchify import patchify, unpatchify

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm  # noqa: E402

sm.set_framework("tf.keras")  # safety


# -----------------------------------------------------------------------------
# GPU setup (safe for mixed systems)
# -----------------------------------------------------------------------------

def _enable_tf_memory_growth():
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def lower_multiple(x: int, k: int) -> int:
    return (x // k) * k


def center_crop_to_multiple(img: np.ndarray, k: int) -> np.ndarray:
    """Center-crop image to a size where both H and W are multiples of k."""
    h, w = img.shape[:2]
    hc, wc = lower_multiple(h, k), lower_multiple(w, k)
    if hc == 0 or wc == 0:
        raise ValueError(f"Too small to crop to multiple of {k}: got {h}x{w}")
    top, left = (h - hc) // 2, (w - wc) // 2
    return img[top : top + hc, left : left + wc]


def filter_crack_patches(
    img_patches: np.ndarray,
    mask_patches: np.ndarray,
    min_pos_px: int = 1,
    thresh: float = 0.5,
    verbose: bool = True,
):
    """
    Keep only (img, mask) patches where the mask contains at least `min_pos_px`
    crack pixels.

    Works with masks stored as {0,255} or {0,1}. Returns (imgs, masks, kept_idx).
    """
    if mask_patches.ndim == 4 and mask_patches.shape[-1] == 1:
        masks = mask_patches[..., 0]
    else:
        masks = mask_patches

    if masks.max() > 1.0:
        masks01 = (masks > 127).astype(np.uint8)
    else:
        masks01 = (masks > thresh).astype(np.uint8)

    pos_counts = masks01.reshape(masks01.shape[0], -1).sum(axis=1)
    keep = pos_counts >= int(min_pos_px)

    if verbose:
        kept = int(keep.sum())
        total = int(masks01.shape[0])
        pct = (kept / total * 100.0) if total else 0.0
        print(
            f"[filter_crack_patches] kept {kept}/{total} patches ({pct:.1f}%) "
            f"with ≥{int(min_pos_px)} crack px"
        )

    return img_patches[keep], mask_patches[keep], np.where(keep)[0]


def predict_full_image(
    img_path: str,
    model_path: str,
    backbone: str,
    size_x: int = 512,
    size_y: int = 512,
    thresh: float = 0.5,
    save_png: str | None = None,
) -> np.ndarray:
    """
    Full-image inference (arbitrary input resolution):
      - Reads image (BGR->RGB)
      - Center-crops to multiples of (size_x,size_y)
      - Patchify -> preprocess -> predict -> threshold -> unpatchify
      - Places cropped mask back into original canvas; returns uint8 mask (H,W) in {0,255}
    """
    model = tf.keras.models.load_model(model_path, compile=False)
    preprocess = sm.get_preprocessing(backbone)

    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(img_path)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    hc, wc = lower_multiple(h, size_x), lower_multiple(w, size_y)
    top, left = (h - hc) // 2, (w - wc) // 2
    rgb_c = rgb[top : top + hc, left : left + wc]

    patches = patchify(rgb_c, (size_x, size_y, 3), step=size_x)
    ph, pw = patches.shape[:2]
    batch = np.stack([patches[i, j, 0] for i in range(ph) for j in range(pw)], axis=0)

    batch_p = preprocess(batch)
    preds = model.predict(batch_p, batch_size=4, verbose=0)  # (N, H, W, 1)

    preds_bin = (preds[..., 0] > thresh).astype(np.uint8) * 255
    preds_grid = preds_bin.reshape(ph, pw, size_x, size_y)

    mask_c = unpatchify(preds_grid, (hc, wc)).astype(np.uint8)
    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[top : top + hc, left : left + wc] = mask_c

    if save_png is not None:
        cv2.imwrite(save_png, mask_full)

    return mask_full


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SIZE_X = 512
SIZE_Y = 512
BACKBONE = "resnet34"
THRESH = 0.5

SAVE_DIR = os.path.join("checkpoints", "unet_resnet34_crack")
os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Data loading (patchify)
# -----------------------------------------------------------------------------

def load_image_patches(images_root: str, size_x: int, size_y: int) -> np.ndarray:
    patches_out = []
    for directory_path in glob.glob(images_root):
        for img_path in glob.glob(os.path.join(directory_path, "*.[jJ][pP][gG]")):
            if not os.path.isfile(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = center_crop_to_multiple(img, size_x)

            patches = patchify(img, (size_x, size_y, 3), step=size_x)
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patches_out.append(patches[i, j, 0])
    return np.array(patches_out)


def load_mask_patches(masks_root: str, size_x: int, size_y: int) -> np.ndarray:
    patches_out = []
    for directory_path in glob.glob(masks_root):
        for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
            if not os.path.isfile(mask_path):
                continue
            mask = cv2.imread(mask_path, 0)
            if mask is None:
                continue

            mask = ((mask > 127).astype(np.uint8)) * 255  # binarize
            mask = center_crop_to_multiple(mask, size_x)

            mask_patches = patchify(mask, (size_x, size_y), step=size_x)
            for i in range(mask_patches.shape[0]):
                for j in range(mask_patches.shape[1]):
                    p = mask_patches[i, j, :, :]
                    p = (p.astype("float32")) / 255.0  # -> {0,1}
                    patches_out.append(p)
    return np.array(patches_out)


def main():
    _enable_tf_memory_growth()
    print("TF:", tf.__version__, "| sm:", sm.__version__)
    print("GPUs in script:", tf.config.list_physical_devices("GPU"))

    # ----- Load & patchify -----
    train_images = load_image_patches("images/sample/", SIZE_X, SIZE_Y)
    print("train_images", train_images.shape, f"{train_images.nbytes/1e9:.2f} GB")

    train_masks = load_mask_patches("mask/sample/", SIZE_X, SIZE_Y)
    print("train_masks ", train_masks.shape, f"{train_masks.nbytes/1e9:.2f} GB")

    # ----- Optional: keep only patches that contain cracks -----
    train_images, train_masks, kept_idx = filter_crack_patches(
        train_images, train_masks, min_pos_px=1, thresh=0.5, verbose=True
    )
    print("train_images", train_images.shape, f"{train_images.nbytes/1e9:.2f} GB")
    print("train_masks ", train_masks.shape, f"{train_masks.nbytes/1e9:.2f} GB")

    # ----- Encode masks (kept for parity) -----
    labelencoder = LabelEncoder()
    n, h, w = train_masks.shape
    train_masks_reshaped = train_masks.reshape(-1,)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3).astype("float32")

    # ----- Split -----
    X_train, X_test, y_train, y_test = train_test_split(
        train_images, train_masks_input, test_size=0.25, random_state=42
    )
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # ----- Sanity plot -----
    img_idx = random.randint(0, len(X_train) - 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("Train RGB patch")
    plt.imshow(X_train[img_idx].astype(np.uint8))
    plt.axis("off")
    plt.subplot(122)
    plt.title("Train GT mask")
    plt.imshow(y_train[img_idx, ..., 0], cmap="gray")
    plt.axis("off")
    plt.show()

    # ----- Preprocess according to backbone -----
    preprocess_input = sm.get_preprocessing(BACKBONE)
    X_train1 = preprocess_input(X_train)
    X_test1 = preprocess_input(X_test)

    # ----- Class imbalance weighting (per-pixel) -----
    pos_frac = float(y_train.mean())
    pos_w = (1.0 - pos_frac) / (pos_frac + 1e-6)
    sw_train = (1.0 + (pos_w - 1.0) * y_train).astype("float32")

    pos_frac_val = float(y_test.mean())
    pos_w_val = (1.0 - pos_frac_val) / (pos_frac_val + 1e-6)
    sw_val = (1.0 + (pos_w_val - 1.0) * y_test).astype("float32")

    # ----- Model -----
    model = sm.Unet(BACKBONE, encoder_weights="imagenet")
    model.compile(
        "Adam",
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score, sm.metrics.FScore(beta=1.0)],
    )
    model.summary()

    # ----- Callbacks -----
    logdir = os.path.join("tb_logs", "unet_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)

    ckpt_path = os.path.join(SAVE_DIR, "best_val_iou.h5")
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor="val_iou_score",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    # ----- Train -----
    history = model.fit(
        X_train1,
        y_train,
        sample_weight=sw_train,
        validation_data=(X_test1, y_test, sw_val),
        batch_size=16,
        epochs=50,
        callbacks=[tb_cb, ckpt_cb],
    )

    # ----- Curves -----
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, "y", label="train")
    plt.plot(epochs, val_loss, "r", label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    acc = history.history["iou_score"]
    val_acc = history.history["val_iou_score"]
    plt.figure()
    plt.plot(epochs, acc, "y", label="train")
    plt.plot(epochs, val_acc, "r", label="val")
    plt.title("IoU")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    # ----- Quick IoU on test set (threshold 0.5) -----
    y_pred = model.predict(X_test1, verbose=0)
    y_pred_th = (y_pred > THRESH).astype(np.uint8)
    intersection = np.logical_and(y_test, y_pred_th).sum()
    union = np.logical_or(y_test, y_pred_th).sum()
    print("Test-set IoU (global):", float(intersection) / float(union + 1e-6))

    # ----- Save model + config -----
    model.save(os.path.join(SAVE_DIR, "savedmodel"), include_optimizer=False)
    model.save(os.path.join(SAVE_DIR, "unet_resnet34_crack.h5"), include_optimizer=False)

    cfg = {"BACKBONE": BACKBONE, "SIZE_X": SIZE_X, "SIZE_Y": SIZE_Y, "THRESH": THRESH}
    with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Saved to: {SAVE_DIR}")

    # ----- Demo: predict on one test patch -----
    test_idx = random.randint(0, len(X_test) - 1)
    test_img = X_test[test_idx]
    gt = y_test[test_idx, ..., 0]
    pred = model.predict(np.expand_dims(preprocess_input(test_img), 0), verbose=0)[0, ..., 0]

    plt.figure(figsize=(16, 6))
    plt.subplot(131)
    plt.title("Test patch")
    plt.imshow(test_img.astype(np.uint8))
    plt.axis("off")
    plt.subplot(132)
    plt.title("GT")
    plt.imshow(gt, cmap="gray")
    plt.axis("off")
    plt.subplot(133)
    plt.title("Pred")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")
    plt.show()

    # ----- Demo: full-image inference (optional) -----
    demo_img_path = "R5.JPG"  # replace with your image
    if os.path.isfile(demo_img_path):
        save_png = os.path.join(SAVE_DIR, "pred_mask_R.png")
        mask_full = predict_full_image(
            img_path=demo_img_path,
            model_path=os.path.join(SAVE_DIR, "savedmodel"),
            backbone=BACKBONE,
            size_x=SIZE_X,
            size_y=SIZE_Y,
            thresh=THRESH,
            save_png=save_png,
        )

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title("Input")
        plt.imshow(cv2.cvtColor(cv2.imread(demo_img_path), cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Predicted mask")
        plt.imshow(mask_full, cmap="gray")
        plt.axis("off")
        plt.show()
    else:
        print(f"[info] Demo image not found: {demo_img_path}")


if __name__ == "__main__":
    main()
