**Primary implementation:** Python (segmentation, SGBM, crack depth estimation)  
**Supporting implementation:** MATLAB (stereo calibration and rectification)

# Mask-Guided Stereo Crack Depth Estimation

This repository implements a mask-guided stereo vision framework for quantitative crack depth estimation on concrete surfaces.  
The pipeline integrates semantic crack segmentation, stereo rectification, region-adaptive SGBM disparity estimation, and skeleton-guided local plane depth quantification.

---

# Overview of the Pipeline

The overall workflow consists of five major stages:

## 1. Crack Segmentation (Training + Inference)

A U-Net model with a ResNet34 backbone is trained on a public crack image dataset.  
The trained model is then used to generate binary crack masks for stereo image pairs (left and right images).

**Output:**
- `left_mask.png`
- `right_mask.png`

---

## 2. Stereo Camera Calibration (MATLAB)

Stereo camera parameters are estimated using MATLAB Stereo Camera Calibrator.

**Outputs include:**
- Intrinsic parameters
- Extrinsic parameters
- Projection matrix `Q`
- Focal length `fx`
- Baseline (converted to mm)

---

## 3. Stereo Rectification (MATLAB)

Using the estimated stereo parameters:

- Left and right RGB images are rectified
- Corresponding crack masks are rectified using nearest-neighbor interpolation
- Rectification metadata (fx, baseline_mm) is exported

**Outputs:**
- `*_rect.png` (rectified RGB images)
- `*_mask_rect.png` (rectified masks)
- `rect_meta_data.mat` (fx, baseline_mm, image sizes)

---

## 4. Mask-Guided SGBM Disparity & Metric Depth

Using:

- Rectified stereo images
- Rectified crack masks
- Calibration metadata

A region-adaptive two-pass SGBM algorithm is applied:

- Auto or manual disparity band selection
- Overlap-validity constraint
- Left-right consistency (LRC)
- Region-aware matching cost and smoothness
- Disparity fusion
- Metric depth recovery (mm)

**Outputs:**
- Fused disparity map
- Depth map (`depth_mm`)
- Validity masks
- Diagnostic visualizations

---

## 5. Crack Depth Estimation (Local Plane Method)

Using:

- Rectified crack mask
- Metric depth map

A skeleton-guided local plane fitting method is applied:

- Skeleton extraction
- Local intact annulus sampling
- RANSAC plane fitting
- Vertical crack depth drop estimation (mm)
- Robust statistics (Median, IQR, P90–P10)

**Outputs:**
- Crack depth samples (CSV)
- Summary statistics (CSV)
- Histogram plot
- Diagnostic masks

---

# Repository Structure

