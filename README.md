# Stereo Reconstruction Project

This project implements stereo vision techniques to reconstruct 3D scenes from pairs of 2D images. It includes camera calibration, stereo rectification, feature matching, and 3D point cloud generation.

## Theory

### Stereo Vision Basics
Stereo vision works by:
1. Capturing two images from slightly different viewpoints
2. Finding corresponding points between images (feature matching)
3. Calculating depth from disparity (difference in x-coordinates)
4. Triangulating 3D points

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
- XFeat (for accelerated feature matching)

Install with:
```bash
pip install opencv-contrib-python numpy matplotlib open3d
```

### Data Requirements
- Calibration images (chessboard pattern)
- Stereo image pairs for reconstruction

## Usage

1. **Camera Calibration**:
```python
cal_data = StereoCalibration(calibration_data_path)
```

2. **Stereo Rectification**:
```python
R1, R2, P1, P2, Q = stereo_rectify(K0, dist_coeffs0, K1, dist_coeffs1, R, T, image_size)
```

3. **Feature Matching**:
```python
# Using SIFT
imgL_with_kp, imgR_with_kp, matches = feat_detect_match(imgL, imgR)

# Using XFeat
mkpts_0, mkpts_1 = xfeat.match_xfeat_star(imgL, imgR)
```

4. **3D Reconstruction**:
```python
# Triangulation
points_4d = cv2.triangulatePoints(P1, P2, pts_left.T, pts_right.T)
points_3d = points_4d[:3] / points_4d[3]

# Visualization
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d.T)
o3d.visualization.draw_geometries([pcd])
```

## File Structure

Key files:
- `main.ipynb`: Main notebook with all processing steps
- `gui.py`: Optional GUI interface
- `dataset/`: Sample stereo image pairs
- `calib_data_name`: Sample data images for camera calibration
- `checkerboardsize`: (9, 7)

Output files:
- Rectified images (`*_rectified.jpg`)
- Feature matches visualizations (`*_matches.jpg`)
- Point clouds (`*.ply`)
