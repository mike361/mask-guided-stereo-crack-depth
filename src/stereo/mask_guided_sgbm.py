"""
mask_guided_sgbm.py

Purpose
-------
Compute a metric depth map for concrete crack scenes using a mask-guided two-pass
SGBM strategy on rectified stereo pairs.

Inputs
------
- Rectified left/right images (RGB or grayscale)
- Rectified left/right crack masks (binary)
- Stereo calibration metadata (fx, baseline, etc.)

Outputs
-------
- Disparity maps (pass1/pass2/fused)
- Validity masks (ROI-valid / LRC-valid)
- Metric depth map (mm)
- Optional diagnostic figures and exported arrays (.mat/.npy)

Core Ideas
----------
1) Two-pass SGBM:
   - crack-focused pass: tuned to preserve discontinuities near crack pixels
   - background-focused pass: tuned for smooth intact regions
2) Mask-derived fusion:
   - use crack mask / soft weights to blend disparities from the two passes
3) Validity enforcement:
   - ROI + left-right consistency (LRC) + optional filtering
4) Metric depth:
   - depth_mm = (fx_px * baseline_mm) / disparity_px

Example
-------
python crack_depth_local_plane.py   --mask-rect "left/Ls4f35_mask_rect.png"   --depth-mat "results_S4f35_metrics3/depth_mm_rect.mat"
--outdir "results_S4f35_metrics3/results_depthA_s4f35_fix"   --roi-left 0.3 --roi-right 0.05 --roi-top 0.1 --roi-bottom 0.2   
--min-blob 200 --keepN 1 --use-skeleton 1   --crack-band-half 2 --skel-step 5   --crack-buffer 20 --border-crop 80   --r-out 60 
--r-in 23   --ransac-iters 200 --ransac-thr 1.5   --min-intact 250 --min-crack 20   --use-grad-cull 1 --grad-thr 3.0   --show 0
"""

import argparse
import gc
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


# =============================================================================
# Helpers: band selection + plotting
# =============================================================================

def compute_band_from_z(fx_px: float, baseline_mm: float, zmin: float, zmax: float):
    """Return (min_disp, num_disp, (dmin, dmax)) from depth band and geometry."""
    dmin = fx_px * baseline_mm / zmax  # far
    dmax = fx_px * baseline_mm / zmin  # near

    # Pad a little and ensure num_disp is positive.
    min_disp = int(max(0, math.floor(dmin) - 16))
    max_disp = int(math.ceil(dmax) + 16)
    num_disp = max(16, max_disp - min_disp)

    return min_disp, num_disp, (dmin, dmax)


def safe_colorbar(
    arr2d: np.ndarray,
    out_png: str,
    title: str,
    unit: str,
    vmin=None,
    vmax=None,
    cmap="viridis",
):
    plt.figure(figsize=(6.8, 5.2))
    im = plt.imshow(arr2d, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(unit)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _downsample_for_viz(arr, max_pixels=4_000_000):
    """Downsample arr (optionally) to keep plotting manageable."""
    h, w = arr.shape[:2]
    total = h * w
    if total <= max_pixels:
        return arr

    scale = math.sqrt(max_pixels / float(total))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # Prefer OpenCV for speed; preserve NaNs by resizing a mask.
    try:
        tmp = arr.astype(np.float32)
        nan_mask = ~np.isfinite(tmp)

        tmp2 = tmp.copy()
        tmp2[nan_mask] = 0.0

        ds = cv2.resize(tmp2, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        if nan_mask.any():
            m = (
                cv2.resize(
                    nan_mask.astype(np.uint8),
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            )
            ds[m] = np.nan
        return ds
    except Exception:
        # Simple stride fallback (keeps NaNs but not anti-aliased).
        s = max(1, int(1 / scale))
        return arr[::s, ::s]


def save_colorbar_png_agg(
    arr2d,
    out_png,
    title,
    unit,
    vmin=None,
    vmax=None,
    cmap="viridis",
    dpi=120,
    max_pixels=4_000_000,
):
    """Memory-safe save: Agg backend + optional downsampling + explicit GC."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    A = _downsample_for_viz(arr2d, max_pixels=max_pixels).astype(np.float32)
    h, w = A.shape

    fig_w = 6.8
    fig_h = max(3.6, fig_w * (h / max(w, 1)))

    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi)
    FigureCanvas(fig)

    ax = fig.add_axes([0.06, 0.06, 0.78, 0.88])
    im = ax.imshow(A, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    cax = fig.add_axes([0.87, 0.12, 0.03, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(unit)

    fig.savefig(out_png)

    del fig, ax, cax, im, cb, A
    gc.collect()


def save_hist_png_agg(values, out_png, lo, hi, dpi=120):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6.8, 5.2), dpi=dpi)
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    if values.size:
        v = values[np.isfinite(values)]
        # FILTER into the plotting window (do NOT clip)
        v = v[(v > lo) & (v < hi)]

        if v.size:
            ax.hist(v, bins=200, range=(lo, hi))
            ax.set_title("Depth Histogram (mm)")
            ax.set_xlabel("Depth (mm)")
            ax.set_ylabel("Count")
        else:
            ax.axis("off")
            ax.text(
                0.5, 0.5, "No values within plot range",
                ha="center", va="center", fontsize=14
            )
    else:
        ax.axis("off")
        ax.text(
            0.5, 0.5, "No finite depth values",
            ha="center", va="center", fontsize=14
        )

    fig.savefig(out_png)
    del fig, ax
    import gc as _gc
    _gc.collect()


# =============================================================================
# IO helpers
# =============================================================================

def imread_color(p: str) -> np.ndarray:
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return img


def imread_mask(p: str, target_shape=None) -> np.ndarray:
    """Read mask -> float32 in [0,1]. Accepts 0/1 or 0/255 images."""
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(p)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if target_shape is not None and m.shape[:2] != target_shape[:2]:
        raise ValueError(f"Mask shape {m.shape} != image shape {target_shape}")
    m = m.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    return np.clip(m, 0.0, 1.0)


def load_meta(meta_path, fx_cli, baseline_cli):
    fx = fx_cli
    B = baseline_cli
    cx = cy = None

    if meta_path:
        m = sio.loadmat(meta_path, squeeze_me=False, struct_as_record=False)
        if fx is None and "fx" in m:
            fx = float(np.array(m["fx"]).ravel()[0])
        if B is None and "baseline_mm" in m:
            B = float(np.array(m["baseline_mm"]).ravel()[0])
        if "cx" in m:
            cx = float(np.array(m["cx"]).ravel()[0])
        if "cy" in m:
            cy = float(np.array(m["cy"]).ravel()[0])

    if fx is None or B is None:
        raise ValueError(
            "Need fx (pixels) and baseline (mm). Provide --meta rect_meta.mat or --fx/--baseline-mm."
        )
    return fx, B, cx, cy


def compute_display_range(arr: np.ndarray, pct_low=2, pct_high=98, fallback=(0.0, 1.0)):
    """Return (vmin, vmax) based on percentiles of finite values."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return fallback
    vmin, vmax = np.nanpercentile(finite, [pct_low, pct_high])
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def save_colorbar_image(
    arr2d: np.ndarray,
    out_png: str,
    title: str,
    unit: str,
    vmin=None,
    vmax=None,
    cmap="viridis",
):
    """Matplotlib save with colorbar; NaNs render as transparent background."""
    plt.figure(figsize=(6.8, 5.2))
    im = plt.imshow(arr2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(unit)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =============================================================================
# Local stats / cues
# =============================================================================

def local_mean_sigma(img, k):
    f = img.astype(np.float32)
    mu = cv2.boxFilter(f, ddepth=-1, ksize=(k, k), normalize=True)
    mu2 = cv2.boxFilter(f * f, ddepth=-1, ksize=(k, k), normalize=True)
    var = np.maximum(mu2 - mu * mu, 1e-6)
    return mu, np.sqrt(var)


def to_zsad(img, k):
    """Zero-mean image for ZSAD; rescale back to 8-bit."""
    mu, _ = local_mean_sigma(img, k)
    z = img.astype(np.float32) - mu
    z = np.clip(z + 128.0, 0, 255).astype(np.uint8)
    return z


def to_ncc_like(img, k):
    """Zero-mean, unit-variance; rescale to 8-bit."""
    mu, sigma = local_mean_sigma(img, k)
    z = (img.astype(np.float32) - mu) / sigma
    z = np.clip(z, -3.0, 3.0)
    z = ((z + 3.0) * (255.0 / 6.0)).astype(np.uint8)
    return z


def to_gradient_mag(img, k_sobel=3, blur_ks=0):
    gX = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=k_sobel)
    gY = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=k_sobel)
    mag = cv2.magnitude(gX, gY)
    if blur_ks > 1:
        mag = cv2.GaussianBlur(mag, (blur_ks, blur_ks), 0)

    mmin, mmax = float(np.min(mag)), float(np.max(mag))
    if mmax > mmin:
        mag = ((mag - mmin) * (255.0 / (mmax - mmin))).astype(np.uint8)
    else:
        mag = np.zeros_like(img, dtype=np.uint8)
    return mag


# =============================================================================
# Mask -> soft weights W
# =============================================================================

def make_soft_weights(mask01: np.ndarray, band_px: int = 15, gamma: float = 0.75) -> np.ndarray:
    """Broaden crack influence into a soft band so fusion favors crack pass near the line."""
    m = np.clip(mask01.astype(np.float32), 0, 1)
    k = max(3, 2 * max(1, band_px // 6) + 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.dilate(m, se, iterations=1)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=band_px / 3.0, sigmaY=band_px / 3.0)
    if m.max() > 0:
        m = m / m.max()
    if gamma != 1.0:
        m = np.power(m, gamma)
    return np.clip(m, 0, 1).astype(np.float32)


# =============================================================================
# SGBM core
# =============================================================================

def create_sgbm(
    min_disp,
    num_disp,
    block,
    cn,
    P1_scale,
    P2_scale,
    uniq,
    speckle_ws,
    speckle_range,
    mode_3way,
):
    if num_disp % 16 != 0:
        num_disp = int(np.ceil(num_disp / 16.0) * 16)
        print(f"[info] Rounded numDisparities -> {num_disp}")

    mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY if mode_3way else cv2.STEREO_SGBM_MODE_SGBM
    P1 = int(8 * cn * (block ** 2) * P1_scale)
    P2 = int(32 * cn * (block ** 2) * P2_scale)  # typically P2 ≈ 4–8·P1 (scales tune it)

    return cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=uniq,
        speckleWindowSize=speckle_ws,
        speckleRange=speckle_range,
        preFilterCap=63,
        mode=mode,
    )


def compute_two_pass(
    grayL,
    grayR,
    min_disp,
    num_disp,
    crack_block=3,
    bg_block=9,
    crack_uniq=6,
    bg_uniq=12,
    crack_P1_scale=0.5,
    crack_P2_scale=0.5,
    bg_P1_scale=1.0,
    bg_P2_scale=1.0,
    mode_3way=True,
    crack_L=None,
    crack_R=None,
):
    cn = 1  # grayscale

    # Pass A (crack-optimized)
    sA = create_sgbm(
        min_disp,
        num_disp,
        block=crack_block,
        cn=cn,
        P1_scale=crack_P1_scale,
        P2_scale=crack_P2_scale,
        uniq=crack_uniq,
        speckle_ws=0,
        speckle_range=0,
        mode_3way=mode_3way,
    )

    # Pass B (background-robust)
    sB = create_sgbm(
        min_disp,
        num_disp,
        block=bg_block,
        cn=cn,
        P1_scale=bg_P1_scale,
        P2_scale=bg_P2_scale,
        uniq=bg_uniq,
        speckle_ws=0,
        speckle_range=1,
        mode_3way=mode_3way,
    )

    crackL = crack_L if crack_L is not None else grayL
    crackR = crack_R if crack_R is not None else grayR

    dispA = sA.compute(crackL, crackR).astype(np.float32) / 16.0
    dispB = sB.compute(grayL, grayR).astype(np.float32) / 16.0

    return dispA, dispB


def left_right_consistency(dispL: np.ndarray, dispR: np.ndarray, thresh: float = 1.0) -> np.ndarray:
    """Mask where |dL - dR(y, x - dL)| <= thresh. NaN-safe."""
    h, w = dispL.shape
    mask = np.zeros_like(dispL, dtype=bool)

    finiteL = np.isfinite(dispL)
    if not np.any(finiteL):
        return mask

    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    xr = np.full_like(xs, -1, dtype=np.int32)

    xr[finiteL] = np.rint(xs[finiteL] - dispL[finiteL]).astype(np.int32)

    valid = finiteL & (xr >= 0) & (xr < w) & np.isfinite(dispR)

    rows = np.repeat(np.arange(h)[:, None], w, axis=1)
    dR_sampl = np.full_like(dispL, np.nan, dtype=np.float32)
    dR_sampl[valid] = dispR[rows[valid], xr[valid]]

    ok = valid & np.isfinite(dR_sampl)
    mask[ok] = (np.abs(dispL[ok] - dR_sampl[ok]) <= float(thresh))
    return mask


def compute_validity_metrics(
    disp: np.ndarray,
    lrc_mask: np.ndarray,
    overlap_mask: np.ndarray,
    crack_band_mask: np.ndarray,
    disp_min_abs: float,
    disp_max_abs: float,
):
    """
    Returns:
      - valid_ratio_* : fraction of pixels that are finite and in disparity band (and in overlap)
      - lrc_pass_rate_* : among candidate pixels (finite+band+overlap), fraction passing LRC
    """
    candidate = (
        overlap_mask
        & np.isfinite(disp)
        & (disp >= disp_min_abs)
        & (disp <= disp_max_abs)
    )

    valid = candidate.copy()
    lrc_pass = candidate & lrc_mask

    # Regions
    crack = candidate & crack_band_mask
    bg = candidate & (~crack_band_mask)

    def frac(a: np.ndarray, b: np.ndarray) -> float:
        den = float(b.sum())
        return float(a.sum()) / den if den > 0 else 0.0

    metrics = {
        "valid_ratio_all": float(valid.mean()),
        "lrc_pass_rate_all": frac(lrc_pass, candidate),

        "valid_ratio_crack": frac(valid & crack_band_mask, crack_band_mask & overlap_mask),
        "valid_ratio_bg": frac(valid & (~crack_band_mask), (~crack_band_mask) & overlap_mask),

        "lrc_pass_rate_crack": frac(lrc_pass & crack_band_mask, candidate & crack_band_mask),
        "lrc_pass_rate_bg": frac(lrc_pass & (~crack_band_mask), candidate & (~crack_band_mask)),
    }
    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("left_rect")
    ap.add_argument("right_rect")
    ap.add_argument(
        "--left-mask-rect",
        default=None,
        help="Rectified left crack mask (binary 0/255 or 0/1)",
    )
    ap.add_argument("--meta", default=None, help="rect_meta.mat (fx, baseline_mm[, cx, cy])")
    ap.add_argument("--fx", type=float, default=None, help="Focal length in pixels (overrides meta)")
    ap.add_argument("--baseline-mm", type=float, default=None, help="Baseline in millimetres (overrides meta)")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--show", action="store_true")

    # crack-pass cost cue
    ap.add_argument(
        "--crack-cost",
        type=str,
        default="grad",
        choices=["grad", "intensity", "zsad", "ncc"],
        help="Cost cue for crack pass (preprocess before SGBM)",
    )
    ap.add_argument("--crack-k", type=int, default=7, help="Window for ZSAD/NCC local stats (odd)")
    ap.add_argument("--sobel-k", type=int, default=3, help="Sobel kernel (3 or 5)")

    # P1/P2 scales per pass
    ap.add_argument("--crack-P1-scale", type=float, default=0.5)
    ap.add_argument("--crack-P2-scale", type=float, default=0.5)
    ap.add_argument("--bg-P1-scale", type=float, default=1.0)
    ap.add_argument("--bg-P2-scale", type=float, default=1.0)

    # disparity config (manual)
    ap.add_argument("--min-disp", type=int, default=0)
    ap.add_argument("--num-disp", type=int, default=256)
    ap.add_argument("--use-3way", action="store_true", help="Use SGBM 3WAY mode")

    # two-pass params
    ap.add_argument("--crack-block", type=int, default=3)
    ap.add_argument("--bg-block", type=int, default=9)
    ap.add_argument("--crack-uniq", type=int, default=6)
    ap.add_argument("--bg-uniq", type=int, default=12)

    # auto-band switches
    ap.add_argument("--auto-band", action="store_true")
    ap.add_argument("--zmin-mm", type=float, default=700.0)
    ap.add_argument("--zmax-mm", type=float, default=1000.0)

    # mask -> W
    ap.add_argument("--mask-dilate", type=int, default=1)
    ap.add_argument("--w-gamma", type=float, default=1.0, help="Gamma for W (<1 spreads influence)")

    # LR consistency
    ap.add_argument("--lrc-thresh", type=float, default=1.0)

    # viz
    ap.add_argument("--max-range-mm", type=float, default=900.0, help="Only for visualization scaling")
    ap.add_argument("--min-depth-mm", type=float, default=700.0, help="Only for visualization clipping")
    ap.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for depth (viridis/jet/turbo/...)",
    )

    args = ap.parse_args()

    def round_to_16(n: int) -> int:
        return int((n + 15) // 16 * 16)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    L = imread_color(args.left_rect)
    R = imread_color(args.right_rect)

    L_rgb_src = L.copy()
    if L.shape[:2] != R.shape[:2]:
        raise ValueError(f"Rectified images must be same size. Got {L.shape} vs {R.shape}")

    # ------------------ Diagnostics: overlap ------------------
    non_black_L = np.any(L > 0, axis=2)
    non_black_R = np.any(R > 0, axis=2)
    overlap_mask = non_black_L & non_black_R
    cv2.imwrite(str(out / "overlap_mask.png"), (overlap_mask.astype(np.uint8) * 255))

    concat_lr = np.hstack([L, R])
    cv2.imwrite(str(out / "rectified_pair_concat.png"), concat_lr)
    print("[dbg] Saved overlap_mask.png and rectified_pair_concat.png.")

    fx, baseline_mm, cx, cy = load_meta(args.meta, args.fx, args.baseline_mm)
    print(f"fx={fx:.3f} px, baseline={baseline_mm:.3f} mm")

    # ------------------ Grayscale + contrast boost ------------------
    grayL = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    grayL = clahe.apply(grayL)
    grayR = clahe.apply(grayR)

    # ------------------ Crack-focused inputs ------------------
    crackL = crackR = None
    if args.crack_cost == "grad":
        crackL = to_gradient_mag(grayL, k_sobel=args.sobel_k, blur_ks=0)
        crackR = to_gradient_mag(grayR, k_sobel=args.sobel_k, blur_ks=0)
    elif args.crack_cost == "zsad":
        crackL = to_zsad(grayL, k=max(args.crack_k, 3))
        crackR = to_zsad(grayR, k=max(args.crack_k, 3))
    elif args.crack_cost == "ncc":
        crackL = to_ncc_like(grayL, k=max(args.crack_k, 3))
        crackR = to_ncc_like(grayR, k=max(args.crack_k, 3))
    elif args.crack_cost == "intensity":
        pass

    # ------------------ Choose disparity band (auto-band or manual) ------------------
    if args.auto_band:
        min_disp_auto, num_disp_auto, (dmin, dmax) = compute_band_from_z(
            fx, baseline_mm, args.zmin_mm, args.zmax_mm
        )
        min_disp = int(min_disp_auto)
        num_disp_rounded = round_to_16(int(num_disp_auto))
        print(
            f"[auto-band] Z [{args.zmin_mm:.0f},{args.zmax_mm:.0f}] mm -> "
            f"disp [{dmin:.1f},{dmax:.1f}] px; min_disp={min_disp}, num_disp={num_disp_rounded}"
        )
    else:
        min_disp = int(args.min_disp)
        num_disp_rounded = round_to_16(args.num_disp)
        print(f"[manual-band] min_disp={min_disp}, num_disp={num_disp_rounded}")

    # ------------------ Two-pass SGBM ------------------
    dispA, dispB = compute_two_pass(
        grayL,
        grayR,
        min_disp=min_disp,
        num_disp=num_disp_rounded,
        crack_block=args.crack_block,
        bg_block=args.bg_block,
        crack_uniq=args.crack_uniq,
        bg_uniq=args.bg_uniq,
        crack_P1_scale=args.crack_P1_scale,
        crack_P2_scale=args.crack_P2_scale,
        bg_P1_scale=args.bg_P1_scale,
        bg_P2_scale=args.bg_P2_scale,
        mode_3way=args.use_3way,
        crack_L=crackL,
        crack_R=crackR,
    )

    # ------------------ Soft weights W ------------------
    if args.left_mask_rect is None:
        print("[warn] --left-mask-rect not provided; using background disparity only.")
        W = np.zeros_like(dispA, dtype=np.float32)
    else:
        M_full = imread_mask(args.left_mask_rect, target_shape=None)
        M = M_full

        if M.shape[:2] != grayL.shape[:2]:
            raise ValueError(f"Cropped mask shape {M.shape} != cropped image shape {grayL.shape}")

        band_px = max(3, int(args.mask_dilate))
        m = (M >= 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3).astype(np.float32)
        sigma = max(1.0, band_px / 2.355)
        W = np.exp(-(dist * dist) / (2.0 * sigma * sigma))
        W = cv2.GaussianBlur(W, (0, 0), sigmaX=sigma / 2, sigmaY=sigma / 2)

        if W.max() > 0:
            W = W / W.max()
        if args.w_gamma != 1.0:
            W = np.power(W, args.w_gamma)

        floor_w = 0.05
        W = floor_w + (1.0 - floor_w) * W
        W = np.clip(W, 0.0, 1.0).astype(np.float32)

        cv2.imwrite(str(out / "W_soft.png"), (np.clip(W, 0, 1) * 255).astype(np.uint8))

    # ------------------ Crack-aware fusion (VALIDITY-AWARE) ------------------
    disp_min_abs = float(min_disp)
    disp_max_abs = float(min_disp + num_disp_rounded - 1)

    validA = np.isfinite(dispA) & (dispA >= disp_min_abs) & (dispA <= disp_max_abs)
    validB = np.isfinite(dispB) & (dispB >= disp_min_abs) & (dispB <= disp_max_abs)

    disp_fused = dispB.copy()

    crack_core = (W >= 0.85)
    crack_band = (W >= 0.50) & (W < 0.85)

    useA_core = crack_core & validA
    disp_fused[useA_core] = dispA[useA_core]

    both_valid = crack_band & validA & validB
    if np.any(both_valid):
        alpha = np.clip((W[both_valid] - 0.50) / (0.85 - 0.50), 0.0, 1.0).astype(np.float32)
        disp_fused[both_valid] = alpha * dispA[both_valid] + (1.0 - alpha) * dispB[both_valid]

    onlyA = crack_band & validA & (~validB)
    onlyB = crack_band & validB & (~validA)
    disp_fused[onlyA] = dispA[onlyA]
    disp_fused[onlyB] = dispB[onlyB]

    neither = crack_band & (~validA) & (~validB)
    disp_fused[neither] = np.nan

    # --- HARD STOP: never allow disparities outside true stereo overlap ---
    disp_fused[~overlap_mask] = np.nan
    dispA[~overlap_mask] = np.nan
    dispB[~overlap_mask] = np.nan

    # ------------------ Early trusted validity mask ------------------
    disp_min_abs = float(min_disp)
    disp_max_abs = float(min_disp + num_disp_rounded - 1)
    eps_disp = 1e-3

    valid_disp_early = (
        np.isfinite(disp_fused)
        & (disp_fused >= disp_min_abs + eps_disp)
        & (disp_fused <= disp_max_abs)
    )
    valid_disp_early = valid_disp_early & overlap_mask

    # ------------------ Left–right consistency ------------------
    sR = create_sgbm(
        min_disp,
        num_disp_rounded,
        block=args.bg_block,
        cn=1,
        P1_scale=args.bg_P1_scale,
        P2_scale=args.bg_P2_scale,
        uniq=args.bg_uniq,
        speckle_ws=0,
        speckle_range=1,
        mode_3way=args.use_3way,
    )
    dispRL = sR.compute(grayR, grayL).astype(np.float32) / 16.0
    lrc_mask = left_right_consistency(disp_fused, dispRL, thresh=args.lrc_thresh)

    fallback_bg = (~lrc_mask) & (W < 0.5)
    disp_fused[fallback_bg] = dispB[fallback_bg]

    fallback_crack = (~lrc_mask) & (W >= 0.5)
    if np.any(fallback_crack):
        disp_med = cv2.medianBlur(np.nan_to_num(disp_fused, nan=0.0).astype(np.float32), 3)
        disp_fused[fallback_crack] = disp_med[fallback_crack]

    # Optional guided refinement only inside crack band
    try:
        import cv2.ximgproc as xip

        rad = 45
        eps = 1e-2
        alpha = 0.3

        disp_g = xip.guidedFilter(
            guide=grayL.astype(np.uint8),
            src=disp_fused.astype(np.float32),
            radius=rad,
            eps=eps,
        )

        crack_mask = (W > 0.8) & np.isfinite(disp_fused)
        disp_fused[crack_mask] = (1.0 - alpha) * disp_fused[crack_mask] + alpha * disp_g[crack_mask]

        print("[info] Applied ximgproc.guidedFilter inside crack band (rad=3, eps=0.05, alpha=0.1).")
    except Exception as e:
        print("[warn] guidedFilter failed:", e)

    # Optional guided refinement ONLY on background
    try:
        import cv2.ximgproc as xip

        rad, eps = 100, 2e-1

        tmp = disp_fused.copy()
        med_val = np.nanmedian(tmp[valid_disp_early]) if np.any(valid_disp_early) else (disp_min_abs + 1.0)
        tmp[~valid_disp_early] = med_val

        disp_g = xip.guidedFilter(
            guide=grayL.astype(np.uint8),
            src=tmp.astype(np.float32),
            radius=rad,
            eps=eps,
        )

        bg_mask = (W < 0.5) & valid_disp_early
        disp_fused[bg_mask] = disp_g[bg_mask]

        print("[info] Applied WEAK background guidedFilter (rad=45, eps=1e-3) on valid background only.")
    except Exception:
        print("[info] cv2.ximgproc not available; skipping background guidedFilter.")
        pass

    # ------------------ Enforce band & validity ------------------
    disp_min_abs = float(min_disp)
    disp_max_abs = float(min_disp + num_disp_rounded - 1)
    eps = 1e-3

    bad = (~np.isfinite(disp_fused)) | (disp_fused < disp_min_abs) | (disp_fused > disp_max_abs)
    disp_fused[~bad] = np.maximum(disp_fused[~bad], disp_min_abs + eps)
    disp_fused[bad] = np.nan

    overlap_mask2 = np.any(L > 0, axis=2) & np.any(R > 0, axis=2)
    valid = np.isfinite(disp_fused) & (disp_fused >= disp_min_abs + eps)

    # ================= metrics for paper comparison =================
    crack_band_metrics = (W >= 0.5)
    overlap_for_metrics = overlap_mask2

    metrics = compute_validity_metrics(
        disp=disp_fused,
        lrc_mask=lrc_mask,
        overlap_mask=overlap_for_metrics,
        crack_band_mask=crack_band_metrics,
        disp_min_abs=disp_min_abs,
        disp_max_abs=disp_max_abs,
    )

    metrics_csv = out / "metrics_mask_guided.csv"
    with open(metrics_csv, "w", newline="") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            f.write(f"{k},{v:.6f}\n")

    print("[metrics] Mask-guided SGBM metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print(f"[saved] {metrics_csv}")
    # ===============================================================

    # ------------------ Diagnostics ------------------
    validA = np.isfinite(dispA) & (dispA > 0)
    validB = np.isfinite(dispB) & (dispB > 0)

    in_band = (
        np.isfinite(disp_fused)
        & (disp_fused >= disp_min_abs)
        & (disp_fused <= disp_max_abs)
    )

    crack_band = W >= 0.5
    bg_band = ~crack_band

    after_lrc = in_band & lrc_mask

    # NOTE: original logic kept as-is
    final_valid = (crack_band & in_band) | (bg_band & in_band)

    overlap_mask2 = np.any(L > 0, axis=2) & np.any(R > 0, axis=2)

    dispA[~overlap_mask2] = np.nan
    dispB[~overlap_mask2] = np.nan
    disp_fused[~overlap_mask2] = np.nan

    def save_mask(name, m):
        cv2.imwrite(str(out / f"{name}.png"), (m.astype(np.uint8) * 255))

    save_mask("validA_crackpass", validA)
    save_mask("validB_bgpass", validB)
    save_mask("crack_band", crack_band)
    save_mask("in_band", in_band)
    save_mask("lrc_mask", lrc_mask)
    save_mask("final_valid", final_valid)
    save_mask("overlap_mask", overlap_mask2)

    blocked_by_overlap = (~overlap_mask2) & (~final_valid)
    save_mask("blocked_by_overlap", blocked_by_overlap)

    # ------------------ Disparity viz ------------------
    def disp_to_u8(d):
        d8 = np.zeros_like(d, dtype=np.uint8)
        if np.isfinite(d).any():
            m = (np.isfinite(d) & (d > 0)).astype(np.uint8)
            if m.any():
                d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U, mask=m)
                d8[m == 0] = 0
        return d8

    cv2.imwrite(str(out / "dispA_crack.png"), cv2.applyColorMap(disp_to_u8(dispA), cv2.COLORMAP_JET))
    cv2.imwrite(str(out / "dispB_bg.png"), cv2.applyColorMap(disp_to_u8(dispB), cv2.COLORMAP_JET))
    cv2.imwrite(str(out / "disp_fused.png"), cv2.applyColorMap(disp_to_u8(disp_fused), cv2.COLORMAP_JET))

    # ------------------ Colorbar PNG for disparity ------------------
    if np.isfinite(disp_fused).any():
        dmin_show, dmax_show = compute_display_range(
            disp_fused,
            pct_low=2,
            pct_high=98,
            fallback=(disp_min_abs, disp_max_abs),
        )
    else:
        dmin_show, dmax_show = disp_min_abs, disp_max_abs

    disp_cb_path = out / "disp_with_colorbar.png"
    save_colorbar_png_agg(
        disp_fused,
        str(disp_cb_path),
        "",
        "px",
        vmin=dmin_show,
        vmax=dmax_show,
        cmap=args.cmap,
        dpi=120,
        max_pixels=4_000_000,
    )
    print(f"[saved] {disp_cb_path}")

    # ------------------ Save raw disparities ------------------
    dispA_path = out / "dispA_raw.npy"
    dispB_path = out / "dispB_raw.npy"
    dispFused_path = out / "disp_fused_raw.npy"
    mat_path = out / "disp_raw.mat"

    np.save(dispA_path, dispA.astype(np.float32))
    np.save(dispB_path, dispB.astype(np.float32))
    np.save(dispFused_path, disp_fused.astype(np.float32))

    sio.savemat(
        str(mat_path),
        {
            "dispA": dispA.astype(np.float32),
            "dispB": dispB.astype(np.float32),
            "disp_fused": disp_fused.astype(np.float32),
            "min_disp": float(min_disp),
            "num_disp": float(num_disp_rounded),
            "fx_px": float(fx),
            "baseline_mm": float(baseline_mm),
        },
    )

    print(
        "[save] Raw disparities saved to:\n"
        f"       {dispA_path}\n"
        f"       {dispB_path}\n"
        f"       {dispFused_path}\n"
        f"       {mat_path}"
    )

    # --------- Kill background disparity spikes ----------
    bg_mask = (W < 0.5) & np.isfinite(disp_fused) & overlap_mask2

    bg_vals = disp_fused[bg_mask]
    if bg_vals.size > 1000:
        lo, hi = np.percentile(bg_vals, [0.5, 99.5])
        spike = bg_mask & ((disp_fused < lo) | (disp_fused > hi))
        # disp_fused[spike] = np.nan
        print(f"[info] Removed disparity spikes: {int(spike.sum())} px (bg p1={lo:.1f}, p99={hi:.1f})")

    cr = (W >= 0.8) & np.isfinite(disp_fused) & overlap_mask2
    vals = disp_fused[cr]
    if vals.size > 1000:
        lo, hi = np.percentile(vals, [29, 71])
        spike = cr & ((disp_fused < lo) | (disp_fused > hi))
        disp_fused[spike] = np.nan

    valid = np.isfinite(disp_fused) & (disp_fused >= disp_min_abs + eps) & overlap_mask

    # ------------------ Depth (mm) ------------------
    depth_mm = np.full(disp_fused.shape, np.nan, dtype=np.float32)
    depth_mm[valid] = (fx * baseline_mm) / disp_fused[valid]

    non_black = np.any(L > 0, axis=2)
    scene_mask = valid & non_black

    depth_scene = np.full_like(depth_mm, np.nan)
    depth_scene[scene_mask] = depth_mm[scene_mask]

    depth_full = depth_scene
    L_full = L

    # ------------------ Relative depth via plane fit ------------------
    H, Wimg = depth_full.shape

    if args.left_mask_rect is not None:
        M_pc_full = imread_mask(args.left_mask_rect, target_shape=None)
        M_pc = M_pc_full

        if M_pc.shape[:2] != grayL.shape[:2]:
            raise ValueError(f"Cropped mask shape {M_pc.shape} != cropped image shape {grayL.shape}")

        crack_mask_pc = M_pc >= 0.5
    else:
        crack_mask_pc = np.zeros_like(depth_full, dtype=bool)

    cv2.imwrite(str(out / "mask_rect_cropped.png"), (np.clip(M_pc, 0, 1) * 255).astype(np.uint8))

    Z_abs = depth_full.astype(np.float32)
    valid_depth = np.isfinite(Z_abs) & (Z_abs > 0)
    non_black_pc = np.any(L_rgb_src > 0, axis=2)

    intact_mask_pc = valid_depth & non_black_pc & (~crack_mask_pc)

    cx_use = float(cx) if cx is not None else (Wimg - 1) / 2.0
    cy_use = float(cy) if cy is not None else (H - 1) / 2.0

    ys, xs = np.indices((H, Wimg))
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    X_abs = (xs - cx_use) * Z_abs / float(fx)
    Y_abs = (ys - cy_use) * Z_abs / float(fx)

    if np.any(intact_mask_pc):
        Xi = X_abs[intact_mask_pc].reshape(-1, 1)
        Yi = Y_abs[intact_mask_pc].reshape(-1, 1)
        Zi = Z_abs[intact_mask_pc].reshape(-1, 1)

        A = np.concatenate([Xi, Yi, np.ones_like(Xi)], axis=1)
        coeffs, _, _, _ = np.linalg.lstsq(A, Zi, rcond=None)
        a, b, c_plane = coeffs.ravel()
        print(f"[plane] Fitted plane: Z = {a:.6f}*X + {b:.6f}*Y + {c_plane:.3f}")
    else:
        Zi_all = Z_abs[valid_depth].reshape(-1, 1)
        a = b = 0.0
        c_plane = float(np.nanmedian(Zi_all))
        print(f"[plane] No intact pixels; using flat plane Z = {c_plane:.3f} mm")

    Z_plane = a * X_abs + b * Y_abs + c_plane
    rel = (Z_abs - Z_plane).astype(np.float32)

    depth_plane_rel = np.full_like(Z_abs, np.nan, dtype=np.float32)

    keep_mask = valid_depth & np.isfinite(rel)
    remove_crack_above = keep_mask & crack_mask_pc & (rel > 0.0)

    keep_final = keep_mask & (~remove_crack_above)
    depth_plane_rel[keep_final] = rel[keep_final]

    print(
        f"[rel-depth] valid={int(keep_mask.sum())}, "
        f"removed_crack_above_plane={int(remove_crack_above.sum())}, "
        f"kept={int(keep_final.sum())}"
    )

    bg_clip_mm = 15.0
    W_soft = W
    bg_mask = (W_soft < 0.5) & np.isfinite(depth_plane_rel)
    depth_plane_rel[bg_mask & (np.abs(depth_plane_rel) > bg_clip_mm)] = np.nan

    sio.savemat(str(out / "depth_plane_rel_mm_rect.mat"), {"depth_plane_rel_mm": depth_plane_rel})

    # ------------------ Point clouds (THREE exports) ------------------
    Z_abs_mm = Z_abs.astype(np.float32)
    Z_plane_mm_map = Z_plane.astype(np.float32)
    rel_mm = depth_plane_rel.astype(np.float32)

    overlap_mask_strict = np.any(L > 0, axis=2) & np.any(R > 0, axis=2)

    valid_xyz = np.isfinite(Z_abs_mm) & (Z_abs_mm > 0) & overlap_mask2

    if np.any(valid_xyz):
        xs_v = xs[valid_xyz].astype(np.float32)
        ys_v = ys[valid_xyz].astype(np.float32)

        Z_abs_m_v = (Z_abs_mm[valid_xyz] * 1e-3).astype(np.float32)
        Z_plane_m_v = (Z_plane_mm_map[valid_xyz] * 1e-3).astype(np.float32)

        X_m_v = (xs_v - float(cx_use)) * Z_plane_m_v / float(fx)
        Y_m_v = (ys_v - float(cy_use)) * Z_plane_m_v / float(fx)

        Y_m_v = -Y_m_v

        colors_bgr = L_rgb_src[valid_xyz]
        B_v = colors_bgr[:, 0].astype(np.uint8)
        G_v = colors_bgr[:, 1].astype(np.uint8)
        R_v = colors_bgr[:, 2].astype(np.uint8)

        label_v = crack_mask_pc[valid_xyz].astype(np.uint8)

        pc_abs = np.column_stack([X_m_v, Y_m_v, -Z_abs_m_v, R_v, G_v, B_v, label_v])
        abs_path = out / "pointcloud_abs_xyzrgb_label.txt"
        np.savetxt(abs_path, pc_abs, fmt="%.6f %.6f %.6f %d %d %d %d")
        print(f"[save] {abs_path}")

        rel_mm_v = rel_mm[valid_xyz]
        rel_mm_v = np.nan_to_num(rel_mm_v, nan=0.0).astype(np.float32)

        Z_delta_m_v = np.zeros_like(Z_abs_m_v, dtype=np.float32)
        crack_pts = label_v == 1
        Z_delta_m_v[crack_pts] = -(rel_mm_v[crack_pts] * 1e-3)

        pc_delta = np.column_stack([X_m_v, Y_m_v, -Z_delta_m_v, R_v, G_v, B_v, label_v])
        delta_path = out / "pointcloud_delta_xyzrgb_label.txt"
        np.savetxt(delta_path, pc_delta, fmt="%.6f %.6f %.6f %d %d %d %d")
        print(f"[save] {delta_path}  (background Z=0, crack Z=-rel)")

        Z_flat_m_v = Z_plane_m_v.copy()
        Z_flat_m_v[crack_pts] = Z_abs_m_v[crack_pts]

        pc_flat = np.column_stack([X_m_v, Y_m_v, -Z_flat_m_v, R_v, G_v, B_v, label_v])
        flat_path = out / "pointcloud_flat_plane_xyzrgb_label.txt"
        np.savetxt(flat_path, pc_flat, fmt="%.6f %.6f %.6f %d %d %d %d")
        print(f"[save] {flat_path}  (background=plane, crack=absolute)")
    else:
        print("[save] No valid depth values – point clouds not written.")

    # ------------------ Save depth products ------------------
    sio.savemat(str(out / "depth_mm_rect.mat"), {"depth_mm": depth_full})
    np.save(out / "depth_mm_rect.npy", depth_full)
    cv2.imwrite(str(out / "depth_mm_float32.tiff"), depth_full.astype(np.float32))

    scale = 10.0
    depth_u16 = np.clip(np.nan_to_num(depth_full, nan=0.0) * scale, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(out / "depth_mm_x0p1mm_uint16.png"), depth_u16)

    invalid_mask = ~np.isfinite(depth_full)
    cv2.imwrite(str(out / "depth_invalid_mask.png"), (invalid_mask * 255).astype(np.uint8))

    # ----- colorized depth (OpenCV quick view) -----
    vmin = args.min_depth_mm
    vmax = args.max_range_mm

    depth_show = np.nan_to_num(depth_full, nan=0.0, posinf=0.0, neginf=0.0)
    depth_show = np.clip(depth_show, vmin, vmax)

    cv_cmap = getattr(cv2, f"COLORMAP_{args.cmap.upper()}", cv2.COLORMAP_VIRIDIS)
    depth_u8 = ((depth_show - vmin) / max(vmax - vmin, 1e-6) * 255.0).astype(np.uint8)

    depth_u8[~overlap_mask2] = 0
    depth_color = cv2.applyColorMap(depth_u8, cv_cmap)
    depth_color[~overlap_mask2] = 0

    cv2.imwrite(str(out / "depth_color.png"), depth_color)

    # ----- colorbar for depth -----
    depth_cb_path = out / "depth_with_colorbar.png"
    finite_count = int(np.isfinite(depth_full).sum())

    if finite_count > 0:
        zmin_show, zmax_show = compute_display_range(
            depth_full,
            pct_low=2,
            pct_high=98,
            fallback=(args.min_depth_mm, args.max_range_mm),
        )
    else:
        zmin_show, zmax_show = args.min_depth_mm, args.max_range_mm

    print(f"[depth cb] finite={finite_count}, vmin={zmin_show:.3f}, vmax={zmax_show:.3f}, save={depth_cb_path}")
    save_colorbar_png_agg(
        depth_full,
        str(depth_cb_path),
        "",
        "mm",
        vmin=zmin_show,
        vmax=zmax_show,
        cmap=args.cmap,
        dpi=120,
        max_pixels=4_000_000,
    )
    print(f"[saved] {depth_cb_path}")

    # ----- depth histogram (Agg) -----
    hist_path = out / "depth_hist.png"
    vd = depth_mm[np.isfinite(depth_mm)]
    save_hist_png_agg(vd, str(hist_path), zmin_show, zmax_show, dpi=120)
    print(f"[saved] {hist_path}")

    # ----- Disparity + Depth stats -----
    vd = depth_mm[np.isfinite(depth_mm)]

    dv = disp_fused[np.isfinite(disp_fused)]
    if dv.size:
        p1d, p50d, p99d = np.percentile(dv, [1, 50, 99])
        Z1 = (fx * baseline_mm) / p1d
        Z50 = (fx * baseline_mm) / p50d
        Z99 = (fx * baseline_mm) / p99d
        print(f"[disp] p1={p1d:.1f}px  median={p50d:.1f}px  p99={p99d:.1f}px")
        print(f"[disp→Z] p1={Z1:.1f}mm  median={Z50:.1f}mm  p99={Z99:.1f}mm\n")
    else:
        print("[disp] no valid disparities")

    if vd.size:
        p1, p50, p99 = np.percentile(vd, [1, 50, 99])
        print(f"[depth] p1={p1:.2f}mm, median={p50:.2f}mm, p99={p99:.2f}mm, max={np.nanmax(vd):.2f}mm")
    else:
        print("[depth] no finite depth values")

    Z_truth = 600.0
    d_expected = (fx * baseline_mm) / Z_truth
    print(f"[check] For Z=600 mm, expected disparity ≈ {d_expected:.1f} px")

    d_min = float(min_disp)
    d_max = float(min_disp + num_disp_rounded - 1)
    Z_at_dmin = (fx * baseline_mm) / d_min if d_min > 0 else float("inf")
    Z_at_dmax = (fx * baseline_mm) / d_max if d_max > 0 else float("inf")
    print(f"[band] disp ∈ [{d_min:.1f}, {d_max:.1f}] px  →  Z ∈ [{Z_at_dmax:.1f}, {Z_at_dmin:.1f}] mm")

    overlay = cv2.addWeighted(L_full, 0.5, depth_color, 0.5, 0.0)
    cv2.imwrite(str(out / "overlay_left_depth.png"), overlay)

    # ---------- Interactive preview ----------
    import matplotlib as mpl

    backend = mpl.get_backend().lower()
    interactive_backends = ["qt5agg", "tkagg", "macosx", "gtk3agg", "wxagg", "webagg"]
    can_show = any(b in backend for b in interactive_backends)

    if args.show and can_show:
        plt.figure(figsize=(10, 5))
        plt.suptitle("Rectified pair")
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(L, cv2.COLOR_BGR2RGB))
        plt.title("Left")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(R, cv2.COLOR_BGR2RGB))
        plt.title("Right")
        plt.axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.figure(figsize=(6.8, 5.2))
        im0 = plt.imshow(disp_fused, vmin=dmin_show, vmax=dmax_show, cmap=args.cmap)
        plt.title("Disparity (px) – fused")
        plt.axis("off")
        plt.colorbar(im0, fraction=0.046, pad=0.04).set_label("px")

        plt.figure(figsize=(6.8, 5.2))
        im1 = plt.imshow(depth_full, vmin=zmin_show, vmax=zmax_show, cmap=args.cmap)
        plt.title("Depth (mm)")
        plt.axis("off")
        plt.colorbar(im1, fraction=0.046, pad=0.04).set_label("mm")

        plt.show()
    else:
        print(f"[info] Figures saved in: {out}")
        plt.close("all")


if __name__ == "__main__":
    main()
