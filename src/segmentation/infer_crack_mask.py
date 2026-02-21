"""
concrete_crack_inference.py

Purpose
-------
Run full-resolution crack mask inference using a trained U-Net+ResNet34 model.

Key Idea
--------
The input image is center-cropped to a size divisible by the patch size, then patchified,
predicted tile-by-tile, thresholded, and finally stitched back via unpatchify.

Inputs
------
- --image : RGB image path
- --model : SavedModel directory or .h5
- --backbone : segmentation_models backbone name (default: resnet34)
- --patch : patch size (default: 512)
- --thresh : sigmoid threshold (default: 0.5)

Output
------
- Optional binary mask PNG saved via --out (uint8 {0,255})
"""


import os
import argparse

import numpy as np
import cv2
import tensorflow as tf
from patchify import patchify, unpatchify

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm  # noqa: E402

sm.set_framework("tf.keras")


def lower_multiple(x: int, k: int) -> int:
    return (x // k) * k


def predict_full_image(
    img_path: str,
    model_path: str,
    backbone: str,
    size_x: int = 512,
    size_y: int = 512,
    thresh: float = 0.5,
    save_png: str | None = None,
) -> np.ndarray:
    model = tf.keras.models.load_model(model_path, compile=False)
    preprocess = sm.get_preprocessing(backbone)

    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Not found: {img_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    hc, wc = lower_multiple(h, size_x), lower_multiple(w, size_y)
    top, left = (h - hc) // 2, (w - wc) // 2
    rgb_c = rgb[top : top + hc, left : left + wc]

    patches = patchify(rgb_c, (size_x, size_y, 3), step=size_x)
    ph, pw = patches.shape[:2]
    batch = np.stack([patches[i, j, 0] for i in range(ph) for j in range(pw)], axis=0)

    batch = preprocess(batch)
    preds = model.predict(batch, batch_size=4, verbose=0)  # (N, H, W, 1)

    preds_bin = (preds[..., 0] > thresh).astype(np.uint8) * 255
    preds_grid = preds_bin.reshape(ph, pw, size_x, size_y)
    mask_c = unpatchify(preds_grid, (hc, wc)).astype(np.uint8)

    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[top : top + hc, left : left + wc] = mask_c

    if save_png:
        os.makedirs(os.path.dirname(save_png) or ".", exist_ok=True)
        cv2.imwrite(save_png, mask_full)

    return mask_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--model", required=True, help="SavedModel directory or .h5 path")
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--patch", type=int, default=512, help="Patch size (square)")
    ap.add_argument("--thresh", type=float, default=0.5, help="Sigmoid threshold")
    ap.add_argument("--out", default=None, help="Output PNG path (optional)")
    args = ap.parse_args()

    mask = predict_full_image(
        img_path=args.image,
        model_path=args.model,
        backbone=args.backbone,
        size_x=args.patch,
        size_y=args.patch,
        thresh=args.thresh,
        save_png=args.out,
    )

    if args.out:
        print("Saved:", args.out)
    else:
        print("Done. Mask shape:", mask.shape)


if __name__ == "__main__":
    main()
