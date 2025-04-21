# Stereo Reconstruction Project

This project implements stereo vision techniques to reconstruct 3D scenes from pairs of 2D images. It includes camera calibration, stereo rectification, feature matching, and 3D point cloud generation.

## Theory

### Stereo Vision Basics
Stereo vision works by:
1. Capturing two images from slightly different viewpoints
2. Finding corresponding points between images (feature matching)
3. Calculating depth from disparity (difference in x-coordinates) by Semi Global Block Matching for dense reconstruction.
4. Triangulating 3D points for sparse reconstruction.

### Key Mathematical Concepts
1. **Camera Calibration**:
   - Finds intrinsic (focal length, principal point) and extrinsic (rotation, translation) parameters
   - Uses chessboard pattern and OpenCV's `calibrateCamera()`

2. **Stereo Rectification**:
   - Aligns images to make epipolar lines horizontal
   - Uses `stereoRectify()` to compute rectification transforms

3. **Feature Matching**:
   - SIFT and XFeat for feature detection
   - FLANN-based matcher for correspondence finding
   - Lowe's ratio test for robust matching

4. **3D Reconstruction**:
   - Triangulation using `triangulatePoints()`
   - Point cloud visualization with Open3D

## Requirements

### Python Packages
- OpenCV (with contrib modules)
- NumPy
- Matplotlib
- Open3D
- Tkinter
- XFeat (for accelerated feature matching and experimentation)
- Torch (required for XFeat)

Install with conda by using the environment.yml file:
```bash
conda env create -f environment.yml
```

insure you have cloned the xfeat repo with this repo side by side for proper execution:
https://github.com/verlab/accelerated_features

### Data Requirements
- Calibration images (chessboard pattern)
- Stereo image pairs for reconstruction

## GUI Walkthrough

The repository includes two versions of the GUI:

1. **Single-camera (static scene with different views):**  
   - Available on branch: `xfeat_integ`

2. **Stereo-camera (simultaneous capture from two cameras):**  
   - Available on branch: `triangulation_3dreconstruction`

---

### GUI for `triangulation_3dreconstruction` branch

**Data:**  
Calibration data:  
[Google Drive Folder](https://drive.google.com/drive/folders/191MXhfmMoBgT7wTdzjUvxnEg0y1pAmsM?usp=sharing)  
- Use `stereo_calib_data_v2` or `stereo_calib_data_v3` (PNG-heavy)

Testing data:  
- Use `dataset/stereov2/d*` with `stereo_calib_data_v2`  
- Use `dataset/stereov3/d*` with `stereo_calib_data_v3`

**Workflow:**
- **Calibrate:** Use **Tab 1** with multiple calibration pairs  
- **Rectify:** Use **Tab 2** with a single pair to analyze, based on results from Tab 1  
- **Dense Reconstruction:**  
  - Use **Tab 6** ("Run Disparity") with Tab 2's output  
  - Then **Tab 6** ("Visualize 3D Point Cloud") with results from Tab 6 and Tab 2  
- **Sparse Reconstruction:**  
  - Use **Tab 3** ("Run Detection & Matching") on the same pair as Tab 2  
  - Use **Tab 5** ("Run Triangulation") with results from Tab 1 and Tab 3  
- **Geometry Estimation:** Use **Tab 4** to estimate F/E/R/T from a single stereo pair  

---

### GUI for `xfeat_integ` branch

**Data:**  
Calibration:  
- `calibration_data_logitech_1280x720`  
- `calibration_data_logitech_3840x2160`

Testing:
- `d8` + calibration from `1280x720`  
- `d6-7` + calibration from `3840x2160`

**Workflow:**
- **Dense Reconstruction:**  
  - **Tab 1** (Calibrate Camera) → **Tab 2** (Rectify Pair) → **Tab 6** (Run Disparity) → **Tab 6** (Visualize Dense 3D)

- **Sparse Reconstruction (Standard Features):**  
  - **Tab 1** → **Tab 3** (Detect/Match & Get R, T) → **Tab 5** (Triangulate & Visualize Sparse 3D)

- **Sparse Reconstruction (XFeat Features):**  
  - **Tab 1** → **Tab 3** → **Tab 7** (Run XFeat Matching) → **Tab 7** (Run XFeat Reconstruction & Visualize)

- **Geometry Estimation:**  
  - **Tab 1** → **Tab 4** (Estimate Geometry, using K)  
  - OR directly use **Tab 4**