#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crack_depth_local_plane.py

Python port of your MATLAB "Algorithm A: skeleton-guided local plane baseline" depth drop computation.

Inputs:
  - depth map (mm) from your Python SGBM pipeline (either .mat containing 'depth_mm' or a .npy)
  - rectified crack mask (binary PNG)

Core steps (matches MATLAB):
  1) Load depth + mask, build valid mask
  2) Front-face ROI cropping (fractions)
  3) Crack mask cleaning: open -> area open -> keep N largest components
  4) Build intact eligibility mask: valid & ~crack & ~buffer-dilated-crack; optional border crop; optional depth gradient cull
  5) Skeletonize + spur pruning (approximate)
  6) Subsample skeleton points
  7) For each skeleton point:
        - intact annulus sampling
        - crack disk sampling constrained by crack band
        - local RANSAC plane fit z = a*x + b*y + c (vertical residual threshold in mm)
        - drop = z_base(skel_pt) - median(z_crack_local)
  8) Robust summary stats + histogram
  9) Save CSV outputs

Dependencies:
  pip install numpy scipy opencv-python scikit-image matplotlib

Example:
  python crack_depth_local_plane.py \
    --mask-rect Ls4f35_mask_rect.png \
    --depth-mat depth_mm_rect.mat \
    --outdir results_depthA \
    --roi-left 0.3 --roi-right 0.05 --roi-top 0.1 --roi-bottom 0.2 \
    --min-blob 200 --keepN 1 \
    --use-skeleton 1 --crack-band-half 3 --skel-step 5 \
    --crack-buffer 15 --border-crop 50 \
    --r-out 60 --r-in 18 \
    --ransac-iters 200 --ransac-thr 2.0 \
    --min-intact 300 --min-crack 30 \
    --use-grad-cull 1 --grad-thr 3.0 \
    --show 0
"""

import argparse
import os
from pathlib import Path
import math
import numpy as np
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize, remove_small_objects, disk
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, binary_erosion, binary_dilation


# ------------------------------ IO ------------------------------

def load_depth(depth_path: str) -> np.ndarray:
    """
    Loads depth in mm.
    - .mat: expects 'depth_mm' key (like your pipeline saves)
    - .npy: loads directly
    """
    p = str(depth_path)
    if p.lower().endswith(".npy"):
        D = np.load(p).astype(np.float32)
        return D
    if p.lower().endswith(".mat"):
        S = sio.loadmat(p)
        if "depth_mm" in S:
            D = S["depth_mm"]
        elif "Drect" in S:
            D = S["Drect"]
        else:
            raise KeyError(f"Could not find 'depth_mm' in MAT file keys: {list(S.keys())}")
        return np.array(D, dtype=np.float32)
    raise ValueError("depth file must be .mat or .npy")


def load_mask(mask_path: str) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(mask_path)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    # imbinarize equivalent: threshold at 0 for a predicted mask, but use 127 for safety
    mb = (m > 127).astype(bool)
    return mb


# ------------------------------ Morphology helpers ------------------------------

def keep_n_largest_components(binmask: np.ndarray, keepN: int = 1) -> np.ndarray:
    """
    MATLAB bwareafilt/bwpropfilt equivalent to keep N largest by area.
    """
    lab = label(binmask, connectivity=2)
    if lab.max() == 0:
        return np.zeros_like(binmask, dtype=bool)
    props = regionprops(lab)
    props_sorted = sorted(props, key=lambda r: r.area, reverse=True)
    keep_labels = set([props_sorted[i].label for i in range(min(keepN, len(props_sorted)))])
    out = np.isin(lab, list(keep_labels))
    return out.astype(bool)


def spur_prune_skeleton(skel: np.ndarray, iters: int = 10) -> np.ndarray:
    """
    Approximate MATLAB bwmorph('spur', N):
    iteratively remove endpoints.
    """
    sk = skel.copy().astype(np.uint8)

    # 8-neighborhood endpoint detection: endpoint has exactly one neighbor
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]], dtype=np.uint8)

    for _ in range(int(iters)):
        # count neighbors
        neigh = cv2.filter2D(sk, -1, np.ones((3,3), np.uint8), borderType=cv2.BORDER_CONSTANT)
        # neigh includes the center pixel, so endpoint criteria:
        # for a skeleton pixel (sk==1): neigh == 2 (1 itself + 1 neighbor)
        endpoints = (sk == 1) & (neigh == 2)
        if not endpoints.any():
            break
        sk[endpoints] = 0
    return sk.astype(bool)


# ------------------------------ Gradient culling ------------------------------

def depth_grad_mag_mm(D: np.ndarray) -> np.ndarray:
    """
    MATLAB:
      Gx = imfilter(Drect, fspecial('sobel')'/8, 'replicate');
      Gy = imfilter(Drect, fspecial('sobel')/8,  'replicate');
      Gmag = hypot(Gx, Gy);
    """
    Df = D.astype(np.float32)
    # Sobel kernels same scale as MATLAB /8
    Gx = cv2.Sobel(Df, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Gy = cv2.Sobel(Df, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return cv2.magnitude(Gx, Gy)


# ------------------------------ RANSAC plane fit ------------------------------

def fit_plane_ransac_z(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       iters: int = 200, thr: float = 2.0,
                       early_inlier_ratio: float = 0.85,
                       seed: int | None = None):
    """
    RANSAC for plane: z = a*x + b*y + c
    Residual: |z - (a*x + b*y + c)|
    Port of your MATLAB helper.
    Returns: (a, b, c, inlier_mask)
    """
    rng = np.random.default_rng(seed)
    x = x.reshape(-1).astype(np.float32)
    y = y.reshape(-1).astype(np.float32)
    z = z.reshape(-1).astype(np.float32)

    n = z.size
    if n < 3:
        a_best = 0.0
        b_best = 0.0
        c_best = float(np.nanmedian(z)) if n else 0.0
        return a_best, b_best, c_best, np.zeros((n,), dtype=bool)

    a_best, b_best = 0.0, 0.0
    c_best = float(np.nanmedian(z))
    best_count = 0
    inlier_best = np.zeros((n,), dtype=bool)

    # prepack
    ones = np.ones((3,), dtype=np.float32)

    for t in range(int(iters)):
        idx = rng.choice(n, size=3, replace=False)
        A = np.stack([x[idx], y[idx], ones], axis=1)  # (3,3)

        # rank check (MATLAB rank(A) < 3)
        if np.linalg.matrix_rank(A) < 3:
            continue

        p = np.linalg.lstsq(A, z[idx], rcond=None)[0]  # [a,b,c]
        a, b, c = float(p[0]), float(p[1]), float(p[2])

        zhat = a * x + b * y + c
        r = np.abs(z - zhat)
        inl = r <= float(thr)
        cnt = int(inl.sum())

        if cnt > best_count:
            best_count = cnt
            inlier_best = inl
            a_best, b_best, c_best = a, b, c

            if best_count > early_inlier_ratio * n:
                break

    # refine on best inliers
    if best_count >= 3:
        Xi = x[inlier_best]
        Yi = y[inlier_best]
        Zi = z[inlier_best]
        A = np.stack([Xi, Yi, np.ones_like(Xi)], axis=1)
        p = np.linalg.lstsq(A, Zi, rcond=None)[0]
        a_best, b_best, c_best = float(p[0]), float(p[1]), float(p[2])

    return a_best, b_best, c_best, inlier_best


# ------------------------------ Main pipeline ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-rect", required=True, help="Rectified crack mask PNG (0/255 or 0/1)")
    ap.add_argument("--depth-mat", required=True, help="Depth .mat containing depth_mm, or .npy")
    ap.add_argument("--outdir", default="results_depth_algorithmA")
    ap.add_argument("--show", type=int, default=0, help="1 to show preview plots, 0 to save only")

    # ROI crop
    ap.add_argument("--roi-left", type=float, default=0.3)
    ap.add_argument("--roi-right", type=float, default=0.05)
    ap.add_argument("--roi-top", type=float, default=0.1)
    ap.add_argument("--roi-bottom", type=float, default=0.2)

    # crack cleaning
    ap.add_argument("--min-blob", type=int, default=200)
    ap.add_argument("--keepN", type=int, default=1)
    ap.add_argument("--crack-core-erode", type=int, default=2)

    # skeleton + sampling
    ap.add_argument("--use-skeleton", type=int, default=1)
    ap.add_argument("--crack-band-half", type=int, default=3)
    ap.add_argument("--skel-step", type=int, default=5)

    # local intact reference
    ap.add_argument("--crack-buffer", type=int, default=15)
    ap.add_argument("--border-crop", type=int, default=50)
    ap.add_argument("--r-out", type=int, default=60)
    ap.add_argument("--r-in", type=int, default=18)  # default = crackBuffer + 3 in your MATLAB

    # ransac
    ap.add_argument("--ransac-iters", type=int, default=200)
    ap.add_argument("--ransac-thr", type=float, default=2.0)
    ap.add_argument("--min-intact", type=int, default=300)
    ap.add_argument("--min-crack", type=int, default=30)

    # grad cull
    ap.add_argument("--use-grad-cull", type=int, default=1)
    ap.add_argument("--grad-thr", type=float, default=3.0)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------ Load ------------------
    Drect = load_depth(args.depth_mat)
    MM1 = load_mask(args.mask_rect)

    if Drect.ndim != 2:
        Drect = np.squeeze(Drect)
    # if MM1.shape != Drect.shape:
    #     raise ValueError(f"Mask and depth sizes differ: mask={MM1.shape}, depth={Drect.shape}")
    def match_mask_to_depth(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        """
        Make mask match depth size by center-cropping or symmetric padding then cropping.
        This is safe when depth was produced by overlap-ROI crop.
        """
        th, tw = target_hw
        mh, mw = mask.shape[:2]

        # If mask is larger -> center-crop
        if mh > th:
            top = (mh - th) // 2
            mask = mask[top:top + th, :]
        elif mh < th:
            pad_top = (th - mh) // 2
            pad_bot = th - mh - pad_top
            mask = np.pad(mask, ((pad_top, pad_bot), (0, 0)), mode="constant", constant_values=0)

        mh, mw = mask.shape[:2]
        if mw > tw:
            left = (mw - tw) // 2
            mask = mask[:, left:left + tw]
        elif mw < tw:
            pad_left = (tw - mw) // 2
            pad_right = tw - mw - pad_left
            mask = np.pad(mask, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)

        # Final safety
        return mask[:th, :tw]


    # --- after loading Drect and MM1 ---
    H, W = Drect.shape[:2]

    # Convert mask to 2D (if needed)
    if MM1.ndim == 3:
        MM1 = MM1[..., 0]  # or cv2.cvtColor, but this is enough if it's already grayscale-ish

    # Binarize if needed (uint8 0/255 or 0/1)
    if MM1.dtype != np.bool_:
        if MM1.max() > 1:
            MM1 = (MM1 > 127)
        else:
            MM1 = (MM1 > 0)

    # Auto-fix mismatch
    if MM1.shape != (H, W):
        print(f"[warn] Mask/depth size mismatch: mask={MM1.shape}, depth={(H, W)}. Auto-aligning mask to depth...")
        MM1 = match_mask_to_depth(MM1.astype(np.uint8), (H, W)).astype(bool)
        print(f"[info] Mask resized by crop/pad -> {MM1.shape}")

    # Now proceed
    valid = np.isfinite(Drect) & (Drect > 0)


    H, W = Drect.shape
    valid = np.isfinite(Drect) & (Drect > 0)

    # ------------------ ROI ------------------
    cropLeftPx  = int(round(args.roi_left   * W))
    cropRightPx = int(round(args.roi_right  * W))
    cropTopPx   = int(round(args.roi_top    * H))
    cropBotPx   = int(round(args.roi_bottom * H))

    ROI = np.zeros((H, W), dtype=bool)
    y0 = cropTopPx
    y1 = H - cropBotPx
    x0 = cropLeftPx
    x1 = W - cropRightPx
    ROI[y0:y1, x0:x1] = True

    valid = valid & ROI
    MM1 = MM1 & ROI

    # ------------------ Crack mask clean (imopen + bwareaopen + keepN) ------------------
    rawMask = MM1.astype(bool)

    # imopen with disk(1)
    rawMask = binary_opening(rawMask, footprint=disk(1))
    # bwareaopen(minBlobAreaPx)
    rawMask = remove_small_objects(rawMask, min_size=int(args.min_blob), connectivity=2)
    rawMask = rawMask.astype(bool)

    crackMaskFull = keep_n_largest_components(rawMask, keepN=int(args.keepN))
    crackMaskFull = crackMaskFull & valid

    if not crackMaskFull.any():
        raise RuntimeError("No crack pixels after cleaning. Relax --min-blob/--keepN/ROI.")

    # ------------------ Intact eligibility mask ------------------
    crackBandGlobal = binary_dilation(crackMaskFull, footprint=disk(int(args.crack_buffer)))
    intactMaskGlobal = valid & (~crackMaskFull) & (~crackBandGlobal)

    # border crop
    if int(args.border_crop) > 0:
        bc = int(args.border_crop)
        borderKeep = np.zeros((H, W), dtype=bool)
        rr0, rr1 = 0 + bc, H - bc
        cc0, cc1 = 0 + bc, W - bc
        if rr1 > rr0 and cc1 > cc0:
            borderKeep[rr0:rr1, cc0:cc1] = True
        intactMaskGlobal = intactMaskGlobal & borderKeep

    # bwareaopen(200)
    intactMaskGlobal = remove_small_objects(intactMaskGlobal, min_size=200, connectivity=2).astype(bool)

    # gradient culling
    if int(args.use_grad_cull) == 1:
        Gmag = depth_grad_mag_mm(Drect)
        intactMaskGlobal = intactMaskGlobal & (Gmag <= float(args.grad_thr))

    if not intactMaskGlobal.any():
        raise RuntimeError("No intact pixels after constraints. Relax border/grad/buffer.")

    # ------------------ Skeleton ------------------
    if int(args.use_skeleton) == 1:
        skel = skeletonize(crackMaskFull).astype(bool)
    else:
        # fallback: erode then skeletonize
        er = int(args.crack_core_erode)
        crackCore = binary_erosion(crackMaskFull, footprint=disk(er)) if er > 0 else crackMaskFull.copy()
        skel = skeletonize(crackCore).astype(bool)

    # spur prune
    skel = spur_prune_skeleton(skel, iters=10)
    skel = skel & valid

    sy, sx = np.where(skel)
    if sx.size == 0:
        raise RuntimeError("Skeleton is empty. Check crack mask cleaning/erosion.")

    # subsample skeleton points
    step = max(1, int(args.skel_step))
    idx = np.arange(0, sx.size, step)
    sx = sx[idx].astype(np.int32)
    sy = sy[idx].astype(np.int32)

    # ------------------ Crack sampling band ------------------
    crackBandLocalAll = binary_dilation(skel, footprint=disk(int(args.crack_band_half))).astype(bool)
    crackDepthMaskAll = crackBandLocalAll & crackMaskFull & valid

    # Save mask previews
    cv2.imwrite(str(outdir / "mask_crack_clean.png"), (crackMaskFull.astype(np.uint8) * 255))
    cv2.imwrite(str(outdir / "mask_skeleton.png"), (skel.astype(np.uint8) * 255))
    cv2.imwrite(str(outdir / "mask_crack_band.png"), (crackDepthMaskAll.astype(np.uint8) * 255))
    cv2.imwrite(str(outdir / "mask_intact_global.png"), (intactMaskGlobal.astype(np.uint8) * 255))
    cv2.imwrite(str(outdir / "mask_roi.png"), (ROI.astype(np.uint8) * 255))

    # ------------------ Precompute grids ------------------
    # MATLAB meshgrid(1:W, 1:H) => x in [1..W], y in [1..H]
    # We'll keep the same 1-based coords for parity.
    Xgrid, Ygrid = np.meshgrid(np.arange(1, W + 1, dtype=np.float32),
                               np.arange(1, H + 1, dtype=np.float32))

    # ------------------ Algorithm A loop ------------------
    drop_local = np.full((sx.size,), np.nan, dtype=np.float32)
    n_inliers  = np.zeros((sx.size,), dtype=np.int32)
    n_intact   = np.zeros((sx.size,), dtype=np.int32)
    n_crack    = np.zeros((sx.size,), dtype=np.int32)

    r_out = float(args.r_out)
    r_in  = float(args.r_in)
    r_out2 = r_out * r_out
    r_in2  = r_in * r_in

    # For speed, you can optionally crop computation to a bounding box around each point.
    for k in range(sx.size):
        x0 = int(sx[k])   # 0-based index into arrays, but coords in Xgrid are 1-based
        y0 = int(sy[k])

        # skeleton point in 1-based coordinate system
        x0_1 = float(x0 + 1)
        y0_1 = float(y0 + 1)

        # Compute local radius mask
        dx = Xgrid - x0_1
        dy = Ygrid - y0_1
        r2 = dx * dx + dy * dy

        neigh_out = r2 <= r_out2
        neigh_in  = r2 <= r_in2
        annulus   = neigh_out & (~neigh_in)

        intactLoc = intactMaskGlobal & annulus

        # local crack sampling: within r_out AND within crack band AND crack mask AND valid
        crackLoc = crackMaskFull & neigh_out & crackBandLocalAll & valid

        Zi = Drect[intactLoc]
        Zc = Drect[crackLoc]

        # finite
        Zi = Zi[np.isfinite(Zi)]
        Zc = Zc[np.isfinite(Zc)]

        n_intact[k] = int(Zi.size)
        n_crack[k]  = int(Zc.size)

        if Zi.size < int(args.min_intact) or Zc.size < int(args.min_crack):
            continue

        # coords for intact points
        xi = Xgrid[intactLoc]
        yi = Ygrid[intactLoc]
        zi = Drect[intactLoc]

        fin = np.isfinite(zi)
        xi = xi[fin]; yi = yi[fin]; zi = zi[fin]

        if zi.size < 3:
            continue

        a, b, c, inlierMask = fit_plane_ransac_z(
            xi, yi, zi,
            iters=int(args.ransac_iters),
            thr=float(args.ransac_thr),
            seed=12345 + k  # deterministic
        )

        nin = int(inlierMask.sum())
        n_inliers[k] = nin

        if nin < 0.5 * int(args.min_intact):
            continue

        # baseline at skeleton point
        z_base = a * x0_1 + b * y0_1 + c

        # robust crack depth near skeleton point
        z_crack_med = float(np.median(Zc))

        drop_local[k] = float(z_base - z_crack_med)

    # remove invalid
    drop_valid = drop_local[np.isfinite(drop_local)]
    if drop_valid.size == 0:
        raise RuntimeError("No valid local depth estimates. Relax thresholds or increase radii.")

    # robust stats
    drop_med = float(np.median(drop_valid))
    q25 = float(np.percentile(drop_valid, 25))
    q75 = float(np.percentile(drop_valid, 75))
    drop_iqr = float(q75 - q25)
    p10 = float(np.percentile(drop_valid, 10))
    p90 = float(np.percentile(drop_valid, 90))

    print("\n[Algorithm A] Local plane baseline (RANSAC) depth drop")
    print(f"  N valid skeleton samples = {drop_valid.size}")
    print(f"  Median ΔZ      = {drop_med:.2f} mm")
    print(f"  IQR (Q75-Q25)  = {drop_iqr:.2f} mm (Q25={q25:.2f}, Q75={q75:.2f})")
    print(f"  P90 - P10      = {(p90 - p10):.2f} mm (P10={p10:.2f}, P90={p90:.2f})")

    # ------------------ Save per-sample CSV ------------------
    samples_csv = outdir / "crack_depth_local_samples.csv"
    with open(samples_csv, "w", newline="") as f:
        f.write("k,sx,sy,n_intact,n_crack,n_inliers,drop_mm\n")
        for k in range(sx.size):
            f.write(f"{k},{sx[k]},{sy[k]},{n_intact[k]},{n_crack[k]},{n_inliers[k]},"
                    f"{drop_local[k] if np.isfinite(drop_local[k]) else ''}\n")
    print(f"[saved] {samples_csv}")

    # ------------------ Save summary CSV ------------------
    summary_csv = outdir / "crack_depth_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        f.write("metric,value\n")
        f.write(f"N_valid,{drop_valid.size}\n")
        f.write(f"median_mm,{drop_med:.6f}\n")
        f.write(f"q25_mm,{q25:.6f}\n")
        f.write(f"q75_mm,{q75:.6f}\n")
        f.write(f"iqr_mm,{drop_iqr:.6f}\n")
        f.write(f"p10_mm,{p10:.6f}\n")
        f.write(f"p90_mm,{p90:.6f}\n")
        f.write(f"p90_minus_p10_mm,{(p90-p10):.6f}\n")
    print(f"[saved] {summary_csv}")

    # ------------------ Histogram figure ------------------
    fig_path = outdir / "crack_depth_histogram.png"
    plt.figure(figsize=(8, 4.6))
    plt.hist(drop_valid, bins=60)
    plt.xlabel("ΔZ (mm): local intact plane baseline - local crack median")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.35)
    plt.axvline(drop_med, linewidth=1.5)
    plt.axvline(q25, linestyle="--", linewidth=1.2)
    plt.axvline(q75, linestyle="--", linewidth=1.2)
    plt.title(f"Median={drop_med:.2f} mm, IQR={drop_iqr:.2f} mm")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    if int(args.show) == 1:
        plt.show()
    plt.close()
    print(f"[saved] {fig_path}")

    # ------------------ Optional preview panel (like MATLAB) ------------------
    if int(args.show) == 1:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.title("Crack mask (clean)"); plt.imshow(crackMaskFull, cmap="gray"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.title("Skeleton"); plt.imshow(skel, cmap="gray"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.title("Crack band mask"); plt.imshow(crackDepthMaskAll, cmap="gray"); plt.axis("off")
        plt.tight_layout()
        plt.show()

    print(f"[done] Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

