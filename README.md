# Mask-Guided Semi Global Block Matching Depth Estimation   
## Mask-Guided Stereo Vision Technique for Crack Depth Estimation of Concrete Bridge Deck

Michael Bekele Maru, AHM Muntasir Billaha∗  

---

## Abstract

Accurate quantification of crack depth on concrete surfaces remains challenging due to the limitations of purely image-based and purely geometric approaches. Conventional 2D crack detection methods provide only surface-level information, while standard stereo reconstruction algorithms often degrade near sharp depth discontinuities such as crack boundaries.

This repository presents a mask-guided stereo vision framework for quantitative crack depth estimation on concrete surfaces. The framework integrates semantic crack segmentation, stereo calibration and rectification, region-adaptive Semi-Global Block Matching (SGBM), and skeleton-guided local plane depth quantification.

A two-pass disparity estimation strategy is employed, combining crack-focused and background-focused SGBM with mask-based fusion and left-right consistency validation. The resulting metric depth map is processed using a local intact plane fitting strategy to compute physically interpretable crack depth in millimeters.

The proposed pipeline enables non-contact, image-based crack depth measurement suitable for structural health monitoring and automated infrastructure inspection.

---
## Key Contributions

- Mask-guided two-pass region-adaptive SGBM formulation
- Disparity fusion via crack-aware weighting
- Skeleton-guided local plane crack depth quantification
- Field validation on real concrete crack defects

# System Requirements

All experiments were conducted under the following software configuration.

## Software Environment

- Python 3.9 – 3.11
- OpenCV (contrib build required for SGBM) 
- NumPy, SciPy, scikit-image
- Matplotlib
- TensorFlow (segmentation training)     
- MATLAB R2022a or later (Stereo Camera Calibrator + Computer Vision Toolbox)  
Note: Stereo rectification must be performed beforehand in MATLAB (Computer Vision Toolbox). This repository assumes rectified stereo pairs and rectified crack masks are provided as inputs.  
---

# 1. Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/mask-guided-stereo-crack-depth.git
cd mask-guided-stereo-crack-depth
```
---

**Environment Setup**

The framework uses two independent Python environments:

1. Segmentation environment (U-Net training and inference)  
2. Stereo and depth environment (Mask-guided SGBM and crack depth estimation)

---

**Segmentation Environment**

```bash
pip install -r requirements/requirements_segmentation.txt
```

---

**Stereo/depth environment**
```bash
pip install -r requirements/requirements_stereo.txt
```
---

**MATLAB Requirements**

Stereo calibration and rectification are performed using:

- MATLAB Stereo Camera Calibrator
- Computer Vision Toolbox
  
---

# 2. Processing Pipeline

The overall workflow consists of five major stages.

**Step 1 — Crack Segmentation**
```bash
python src/segmentation/train_crack_segmentation.py
```
- Patch-based U-Net training (512×512 tiles)
- Optional crack-patch filtering
- BCE + Jaccard loss
- IoU-based checkpoint saving

For inference:
```bash
python src/segmentation/infer_crack_mask.py 
```

**Outputs**

- left_mask.png
- right_mask.png

---

**Step 2 — Stereo Calibration (MATLAB)**

Using MATLAB Stereo Camera Calibrator:

- Estimate intrinsic and extrinsic parameters
- Generate projection matrix Q
- Export calibrationSession.mat

---

**Step 3 — Stereo Rectification (MATLAB)**

Execute:
```bash
matlab/rectify_stereo_pairs.m
```
**Outputs**

- Rectified left RGB image: *_rect.png
- Rectified right RGB image: *_rect.png
- Rectified left crack mask: *_mask_rect.png
- Rectified right crack mask: *_mask_rect.png
- Rectification metadata file: rect_meta_data.mat
  (containing fx, baseline_mm, orig_width, rect_width)

---

**Step 4 — Mask-Guided SGBM & Metric Depth Recovery**

Execute:
```bash
src/stereo/mask_guided_sgbm.py
```
**Core components:**

- Automatic or manual disparity band selection
- Two-pass SGBM (crack-focused + background-focused)
- Mask-derived soft weighting
- Left-right consistency validation
- Region-aware disparity fusion
- Metric depth conversion (mm)

**Outputs**

- Fused disparity map
- Validity masks
- depth_mm
- Diagnostic visualizations

---

**Step 5 — Skeleton-Guided Crack Depth Estimation**

Execute:
```bash
src/depth/crack_depth_local_plane.py
```
**Method:**

1. Skeleton extraction from crack mask
2. Local intact annulus sampling
3. RANSAC plane fitting
4. Vertical depth drop computation
5. Robust statistics (Median, IQR, P90–P10)

**Outputs**

- Crack depth samples (CSV)
- Summary statistics (CSV)
- Histogram plots
- Diagnostic masks

---

# Repository Structure
```
mask-guided-stereo-crack-depth/
│
├── src/
│   ├── segmentation/
│   ├── stereo/
│   ├── depth/
│
├── matlab/
│   └── rectify_stereo_pairs.m
│
├── requirements/
│
├── docs/
│   └── figures/
│
└── examples/
```

# Citation

If you use this code, please cite:

@article{maru2026maskguided,
  title={Mask-Guided Stereo Vision Technique for Crack Depth Estimation of Concrete Bridge Deck},
  author={Maru Michael Bekele, AHM Muntasir Billaha*},
  journal={TBD},
  year={2026}
}







