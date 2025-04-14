import os
import numpy as np
import cv2
import glob
import open3d as o3d
import torch
from modules.xfeat import XFeat

# --- Camera Calibration Function ---
def cam_calib(base_path, cancel_event=None):
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    checkerboardsize = (9, 7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((checkerboardsize[0] * checkerboardsize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboardsize[0],
                           0:checkerboardsize[1]].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    gray_shape = None
    images = glob.glob(os.path.join(base_path, "*.jpg"))
    if not images:
        return {"Error": f"No JPG images found in '{base_path}'."}
    images = sorted(images)
    print(f"Found {len(images)} images for calibration.")
    processed_count = 0
    for fname in images:
        if cancel_event and cancel_event.is_set():
            return {"Error": "Task canceled by user."}
        
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Could not read image {fname}. Skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_shape is None:
            gray_shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, checkerboardsize, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_FILTER_QUADS + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            print(f"Found corners in {os.path.basename(fname)}")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            processed_count += 1
        else:
            print(f"Could not find corners in {os.path.basename(fname)}")
    if not objpoints or not imgpoints:
        return {"Error": "Could not find chessboard corners in any suitable images."}
    if gray_shape is None:
        return {"Error": "Could not read shape from any image."}
    print(
        f"Calibrating using {len(objpoints)} images (out of {processed_count} processed) with shape {gray_shape}.")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    try:
        ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray_shape, None, None, flags=cv2.CALIB_RATIONAL_MODEL)
        if not ret:
            return {"Error": "Camera calibration failed (cv2.calibrateCamera returned False)."}
    except Exception as e:
        return {"Error": f"Camera calibration threw an exception: {e}"}

    reprojection_error = 0
    for i in range(len(objpoints)):
        if cancel_event and cancel_event.is_set():
            return {"Error": "Task canceled by user."}
        
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2,
                         cv2.NORM_L2) / len(imgpoints2)
        reprojection_error += error
    mean_error = reprojection_error / len(objpoints)
    print(f"Calibration successful. Mean Reprojection Error: {mean_error:.4f}")
    return {"Camera Matrix": cameraMatrix, "Distortion Parameters": dist, "Reprojection Error": mean_error}

# --- Draw Epipolar Lines Function ---
def draw_epilines(img1, img2, lines, pts1, pts2, cancel_event=None):
    if cancel_event and cancel_event.is_set():
        return img1, img2  # Return unmodified images if canceled
    
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2
    r, c = img1_gray.shape
    img1_color = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2BGR)
    pts1 = np.int32(pts1.reshape(-1, 2))
    pts2 = np.int32(pts2.reshape(-1, 2))
    lines = lines.reshape(-1, 3)
    for r_line, pt1, pt2 in zip(lines, pts1, pts2):
        if cancel_event and cancel_event.is_set():
            return img1_color, img2_color
        
        color = tuple(np.random.randint(0, 255, 3).tolist())
        if abs(r_line[1]) > 1e-6:
            y0 = -r_line[2] / r_line[1]
            y1 = -(r_line[2] + r_line[0] * c) / r_line[1]
            x0, x1 = 0, c
        elif abs(r_line[0]) > 1e-6:
            x0 = -r_line[2] / r_line[0]
            x1 = -(r_line[2] + r_line[1] * r) / \
                r_line[0]
            y0, y1 = 0, r
        else:
            print("Warning: Degenerate epipolar line detected. Skipping draw.")
            continue

        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])

        try:
            ret, p1_clip, p2_clip = cv2.clipLine(
                (0, 0, c, r), (x0, y0), (x1, y1))
            if ret:
                img1_color = cv2.line(img1_color, p1_clip, p2_clip, color, 1)
        except OverflowError:
            print(
                f"Warning: Overflow drawing line segment ({x0},{y0}) to ({x1},{y1}). Skipping.")
            continue
        
        try:
            img1_color = cv2.circle(
                img1_color, tuple(np.int32(pt1)), 5, color, -1)
            img2_color = cv2.circle(
                img2_color, tuple(np.int32(pt2)), 5, color, -1)
        except Exception as e:
            print(f"Warning: Error drawing point circle: {e}")
    return img1_color, img2_color

# --- Stereo Rectification Function ---
def stereo_rect(stereo_path, cameraMatrix=None, distCoeffs=None, cancel_event=None):
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    print("--- Stereo Rectification ---")
    if cameraMatrix is None:
        cameraMatrix = np.array(
            [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float32)
        print("Warning: Using default camera matrix.")
    if distCoeffs is None:
        distCoeffs = np.zeros(5, dtype=np.float32)
        print("Warning: Assuming zero distortion.")
    
    left_image_path = os.path.join(stereo_path, "img1.jpg")
    right_image_path = os.path.join(stereo_path, "img2.jpg")
    
    if not os.path.exists(left_image_path):
        return {"Error": f"Missing {left_image_path}"}
    if not os.path.exists(right_image_path):
        return {"Error": f"Missing {right_image_path}"}
    
    imgL_color = cv2.imread(left_image_path)
    imgR_color = cv2.imread(right_image_path)
    
    if imgL_color is None or imgR_color is None:
        return {"Error": "Failed to load images."}
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    imgL = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR_color, cv2.COLOR_BGR2GRAY)
    h, w = imgL.shape[:2]
    image_size = (w, h)
    K = np.array(cameraMatrix)
    sift = cv2.SIFT_create()
    keypoints_left, descriptors_left = sift.detectAndCompute(imgL, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(imgR, None)
    
    if descriptors_left is None or descriptors_right is None or len(descriptors_left) < 8 or len(descriptors_right) < 8:
        return {"Error": "Not enough keypoints found (need at least 8)."}
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)
    good_matches = []
    
    if matches is not None:
        for pair in matches:
            if cancel_event and cancel_event.is_set():
                return {"Error": "Task canceled by user."}
            if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance:
                good_matches.append(pair[0])
    if len(good_matches) < 8:
        return {"Error": f"Not enough good matches found ({len(good_matches)}, need at least 8)."}
    
    pts_left = np.float32(
        [keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_right = np.float32(
        [keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    E, mask_e = cv2.findEssentialMat(
        pts_left, pts_right, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    if E is None:
        return {"Error": "Could not compute Essential Matrix."}
    if mask_e is None:
        return {"Error": "Essential matrix mask is None."}
    
    pts_left_e = pts_left[mask_e.ravel() == 1]
    pts_right_e = pts_right[mask_e.ravel() == 1]
    
    if len(pts_left_e) < 5:
        return {"Error": f"Not enough inliers for recoverPose ({len(pts_left_e)})."}
    _, R_rect, T_vec, mask_pose = cv2.recoverPose(
        E, pts_left_e, pts_right_e, K)
    if R_rect is None or T_vec is None:
        return {"Error": "Could not recover pose (R, T) from Essential Matrix."}
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    F, mask_f = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_LMEDS)
    imgL_before_rec = imgL_color.copy()
    imgR_before_rec = imgR_color.copy()
    
    if F is not None and mask_f is not None:
        pts_left_in_f = pts_left.reshape(-1, 2)[mask_f.ravel() == 1]
        pts_right_in_f = pts_right.reshape(-1, 2)[mask_f.ravel() == 1]
        if len(pts_left_in_f) > 0:
            try:
                lines_left = cv2.computeCorrespondEpilines(
                    pts_right_in_f.reshape(-1, 1, 2), 2, F)
                lines_left = lines_left.reshape(-1, 3)
                imgL_before_rec, _ = draw_epilines(
                    imgL_before_rec, imgR_before_rec, lines_left, pts_left_in_f, pts_right_in_f, cancel_event)
                lines_right = cv2.computeCorrespondEpilines(
                    pts_left_in_f.reshape(-1, 1, 2), 1, F)
                lines_right = lines_right.reshape(-1, 3)
                imgR_before_rec, _ = draw_epilines(
                    imgR_before_rec, imgL_before_rec, lines_right, pts_right_in_f, pts_left_in_f, cancel_event)
            except Exception as e:
                print(
                    f"Warning: Could not draw epilines before rectification: {e}")
    
    distCoeffs_rect = distCoeffs[:5] if distCoeffs is not None and len(
        distCoeffs) > 5 else distCoeffs
    
    try:
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K, distCoeffs_rect, K, distCoeffs_rect, image_size, R_rect, T_vec, alpha=0.0)
        if Q is None:
            return {"Error": "Stereo rectification failed (Q matrix is None)."}
    except Exception as e:
        return {"Error": f"Stereo rectification failed with exception: {e}"}

    print("Q matrix (Disparity-to-depth) calculated.")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    mapL1, mapL2 = cv2.initUndistortRectifyMap(
        K, distCoeffs_rect, R1, P1, image_size, cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(
        K, distCoeffs_rect, R2, P2, image_size, cv2.CV_32FC1)
    
    imgL_rect_color = cv2.remap(
        imgL_color, mapL1, mapL2, interpolation=cv2.INTER_LINEAR)
    imgR_rect_color = cv2.remap(
        imgR_color, mapR1, mapR2, interpolation=cv2.INTER_LINEAR)
    
    imgL_rect = cv2.cvtColor(imgL_rect_color, cv2.COLOR_BGR2GRAY)
    imgR_rect = cv2.cvtColor(imgR_rect_color, cv2.COLOR_BGR2GRAY)
    
    imgL_after_rec = imgL_rect_color.copy()
    imgR_after_rec = imgR_rect_color.copy()
    
    keypoints_left_rect, descriptors_left_rect = sift.detectAndCompute(
        imgL_rect, None)
    keypoints_right_rect, descriptors_right_rect = sift.detectAndCompute(
        imgR_rect, None)
    
    if descriptors_left_rect is not None and descriptors_right_rect is not None and len(descriptors_left_rect) > 8 and len(descriptors_right_rect) > 8:
        matches_rect = flann.knnMatch(
            descriptors_left_rect, descriptors_right_rect, k=2)
        good_matches_rect = []
        if matches_rect:
            for pair in matches_rect:
                if cancel_event and cancel_event.is_set():
                    return {"Error": "Task canceled by user."}
                if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance:
                    good_matches_rect.append(pair[0])
        if len(good_matches_rect) > 8:
            pts_left_rect = np.float32(
                [keypoints_left_rect[m.queryIdx].pt for m in good_matches_rect]).reshape(-1, 1, 2)
            pts_right_rect = np.float32(
                [keypoints_right_rect[m.trainIdx].pt for m in good_matches_rect]).reshape(-1, 1, 2)
            F_rect, mask_f_rect = cv2.findFundamentalMat(
                pts_left_rect, pts_right_rect, cv2.FM_LMEDS)
    
            if F_rect is not None and mask_f_rect is not None:
                pts_left_rect_in = pts_left_rect.reshape(
                    -1, 2)[mask_f_rect.ravel() == 1]
                pts_right_rect_in = pts_right_rect.reshape(
                    -1, 2)[mask_f_rect.ravel() == 1]
    
                if len(pts_left_rect_in) > 0:
                    try:
                        lines_left_rect = cv2.computeCorrespondEpilines(
                            pts_right_rect_in.reshape(-1, 1, 2), 2, F_rect)
                        lines_left_rect = lines_left_rect.reshape(-1, 3)
                        imgL_after_rec, _ = draw_epilines(
                            imgL_after_rec, imgR_after_rec, lines_left_rect, pts_left_rect_in, pts_right_rect_in, cancel_event)
    
                        lines_right_rect = cv2.computeCorrespondEpilines(
                            pts_left_rect_in.reshape(-1, 1, 2), 1, F_rect)
                        lines_right_rect = lines_right_rect.reshape(-1, 3)
                        imgR_after_rec, _ = draw_epilines(
                            imgR_after_rec, imgL_after_rec, lines_right_rect, pts_right_rect_in, pts_left_rect_in, cancel_event)
                    except Exception as e:
                        print(
                            f"Warning: Could not draw epilines after rectification: {e}")
    
    return {"Original Left": imgL_before_rec, "Original Right": imgR_before_rec,
            "Drawn Rectified Left": imgL_after_rec, "Drawn Rectified Right": imgR_after_rec,
            "Rectified Left": imgL_rect, "Rectified Right": imgR_rect,
            "Rectified Color Left": imgL_rect_color, "Rectified Color Right": imgR_rect_color,
            "disp2depth map": Q}

# --- Feature Detection and Matching Function ---
def feat_detect_match(stereo_path, cameraMatrix=None, cancel_event=None):
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    print("--- Feature Detection & Matching ---")
    if cameraMatrix is None:
        cameraMatrix = np.array(
            [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float32)
        print("Warning: Using default camera matrix.")
    sift = cv2.SIFT_create()
    left_image_path = os.path.join(stereo_path, "img1.jpg")
    right_image_path = os.path.join(stereo_path, "img2.jpg")
    if not os.path.exists(left_image_path):
        return {"Error": f"Missing {left_image_path}"}
    if not os.path.exists(right_image_path):
        return {"Error": f"Missing {right_image_path}"}
    imgL_color = cv2.imread(left_image_path)
    imgR_color = cv2.imread(right_image_path)
    if imgL_color is None or imgR_color is None:
        return {"Error": "Failed to load images."}
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    imgL = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR_color, cv2.COLOR_BGR2GRAY)
    keypoints_left, descriptors_left = sift.detectAndCompute(imgL, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(imgR, None)
    num_kp_left = len(keypoints_left) if keypoints_left is not None else 0
    num_kp_right = len(keypoints_right) if keypoints_right is not None else 0
    if descriptors_left is None or descriptors_right is None or num_kp_left < 8 or num_kp_right < 8:
        return {"Error": "Not enough keypoints found (need at least 8)."}
    print(f"SIFT: {num_kp_left} kp left, {num_kp_right} kp right.")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)
    num_raw_matches = len(matches) if matches else 0
    if not matches:
        return {"Error": f"No matches found by FLANN."}
    print(f"FLANN: {num_raw_matches} raw potential matches (knn=2).")
    good_matches = []
    pts1_list = []
    pts2_list = []
    matches_viz = []
    for i, pair in enumerate(matches):
        if cancel_event and cancel_event.is_set():
            return {"Error": "Task canceled by user."}
        if len(pair) == 2:
            m, n = pair
            matches_viz.append([m])
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                pts1_list.append(keypoints_left[m.queryIdx].pt)
                pts2_list.append(keypoints_right[m.trainIdx].pt)
        elif len(pair) == 1:
            matches_viz.append(pair)
    num_good_matches = len(good_matches)
    print(f"{num_good_matches} good matches after ratio test.")
    if num_good_matches < 8:
        return {"Error": f"Not enough good matches ({num_good_matches}, need at least 8)."}
    pts1 = np.float32(pts1_list).reshape(-1, 1, 2)
    pts2 = np.float32(pts2_list).reshape(-1, 1, 2)
    F, mask_f = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    if F is None:
        F = np.eye(3)
        print("Warning: Fundamental Matrix calculation failed.")
    else:
        print(
            f"Fundamental Matrix inliers: {np.sum(mask_f) if mask_f is not None else 'N/A'}")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    K = np.array(cameraMatrix)
    E, mask_e = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    R, t_vec = np.eye(3), np.zeros((3, 1))
    if E is None:
        print("Warning: Essential Matrix calculation failed. Using identity pose.")
    elif mask_e is None:
        print("Warning: Essential Matrix mask is None. Using identity pose.")
    else:
        pts1_e = pts1[mask_e.ravel() == 1]
        pts2_e = pts2[mask_e.ravel() == 1]
        num_inliers_e = len(pts1_e)
        print(f"Essential Matrix inliers: {num_inliers_e}")
        if num_inliers_e < 5:
            print(
                f"Warning: Not enough inliers for recoverPose ({num_inliers_e}). Using identity pose.")
        else:
            try:
                _, R_pose, t_vec_pose, mask_pose = cv2.recoverPose(
                    E, pts1_e, pts2_e, K)
                if R_pose is None or t_vec_pose is None:
                    print("Warning: recoverPose failed. Using identity pose.")
                else:
                    R, t_vec = R_pose, t_vec_pose
                if mask_pose is not None:
                    print(f"recoverPose inliers: {np.sum(mask_pose)}")
                else:
                    print("Warning: recoverPose mask is None.")
            except Exception as e:
                print(
                    f"Warning: recoverPose threw an exception: {e}. Using identity pose.")
                R, t_vec = np.eye(3), np.zeros((3, 1))

    imgL_with_kp = cv2.drawKeypoints(
        imgL_color, keypoints_left, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imgR_with_kp = cv2.drawKeypoints(
        imgR_color, keypoints_right, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_matches_raw = cv2.drawMatchesKnn(imgL_color, keypoints_left, imgR_color, keypoints_right,
                                         matches_viz, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_good_matches = cv2.drawMatches(imgL_color, keypoints_left, imgR_color, keypoints_right,
                                       good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return {"Left Image with Keypoints": imgL_with_kp, "Right Image with Keypoints": imgR_with_kp,
            "Matched Images Before Lowe's Ratio": img_matches_raw, "Matched Images After Lowe's Ratio": img_good_matches,
            "Left Aligned Keypoints": pts1.reshape(-1, 2), "Right Aligned Keypoints": pts2.reshape(-1, 2),
            "Fundamental Matrix": F, "Essential Matrix": E, "Rotation Matrix": R, "Translation Vector": t_vec,
            "Left Color Image": imgL_color,
            "Right Color Image": imgR_color,
            "Num Keypoints Left": num_kp_left, "Num Keypoints Right": num_kp_right,
            "Num Raw Matches": num_raw_matches, "Num Good Matches": num_good_matches
            }

# --- Stereo Geometry Estimation Function ---
def stereo_geometry_estimation(stereo_path, cameraMatrix=None, cancel_event=None):
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    print("--- Stereo Geometry Estimation ---")
    if cameraMatrix is None:
        cameraMatrix = np.array(
            [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float32)
        print("Warning: Using default camera matrix.")
    left_image_path = os.path.join(stereo_path, "img1.jpg")
    right_image_path = os.path.join(stereo_path, "img2.jpg")
    if not os.path.exists(left_image_path) or not os.path.exists(right_image_path):
        return {"Error": "Missing img1.jpg or img2.jpg."}
    imgL = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        return {"Error": "Failed to load images."}
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    sift = cv2.SIFT_create()
    keypoints_left, descriptors_left = sift.detectAndCompute(imgL, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(imgR, None)
    if descriptors_left is None or descriptors_right is None or len(descriptors_left) < 8 or len(descriptors_right) < 8:
        return {"Error": "Not enough keypoints found (need at least 8)."}
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)
    if not matches:
        return {"Error": "No matches found by FLANN."}
    good_matches = []
    pts_left_list = []
    pts_right_list = []
    for pair in matches:
        if cancel_event and cancel_event.is_set():
            return {"Error": "Task canceled by user."}
        if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance:
            good_matches.append(pair[0])
            pts_left_list.append(keypoints_left[pair[0].queryIdx].pt)
            pts_right_list.append(keypoints_right[pair[0].trainIdx].pt)
    if len(good_matches) < 8:
        return {"Error": f"Not enough good matches ({len(good_matches)}, need at least 8)."}
    pts_left = np.float32(pts_left_list).reshape(-1, 1, 2)
    pts_right = np.float32(pts_right_list).reshape(-1, 1, 2)
    F, mask_f = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)
    if F is None:
        print("Warning: Fundamental Matrix calculation failed.")
    K = np.array(cameraMatrix)
    E, mask_e = cv2.findEssentialMat(
        pts_left, pts_right, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    det_E = None
    svd_vals_E = None
    non_zero_svd_E = None
    R = None
    T_vec = None
    if E is None:
        print("Warning: Essential Matrix calculation failed.")
    elif mask_e is None:
        print("Warning: Essential Matrix mask is None.")
    else:
        det_E = np.linalg.det(E)
        try:
            svd_vals_E = np.linalg.svd(E, compute_uv=False)
            non_zero_svd_E = np.sum(svd_vals_E > 1e-6)
        except np.linalg.LinAlgError:
            print("Warning: SVD computation for E failed.")
        pts_left_e = pts_left[mask_e.ravel() == 1]
        pts_right_e = pts_right[mask_e.ravel() == 1]
        num_inliers_e = len(pts_left_e)
        print(f"Essential Matrix inliers: {num_inliers_e}")
        if num_inliers_e < 5:
            print(
                f"Warning: Not enough inliers for recoverPose ({num_inliers_e}).")
        else:
            try:
                _, R_pose, T_vec_pose, mask_pose = cv2.recoverPose(
                    E, pts_left_e, pts_right_e, K)
                if R_pose is None or T_vec_pose is None:
                    print("Warning: recoverPose failed.")
                else:
                    R, T_vec = R_pose, T_vec_pose
                if mask_pose is not None:
                    print(f"recoverPose inliers: {np.sum(mask_pose)}")
                else:
                    print("Warning: recoverPose mask is None.")
            except Exception as e:
                print(f"Warning: recoverPose threw an exception: {e}")
                R, T_vec = None, None

    if R is None:
        R = np.eye(3)
    if T_vec is None:
        T_vec = np.zeros((3, 1))
    return {"Fundamental Matrix": F, "Essential Matrix": E, "Determinant E": det_E,
            "Singular Values E": svd_vals_E, "Non Zero SVD E": non_zero_svd_E,
            "Rotation Matrix": R, "Translation Vector": T_vec}

# --- Triangulation Function (Feature-Based SIFT) ---
def triangulation_and_3D_reconstruction(pts1, pts2, cameraMatrix, rotationMatrix, translationVector, imgL_color, cancel_event=None):
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    if pts1 is None or pts2 is None or len(pts1) == 0 or len(pts2) == 0 or len(pts1) != len(pts2):
        return {"Error": "Invalid or mismatched keypoints."}
    if cameraMatrix is None or rotationMatrix is None or translationVector is None:
        return {"Error": "Missing K, R, or T."}
    if imgL_color is None:
        return {"Error": "Missing left color image."}
    img_height, img_width = imgL_color.shape[:2]
    pts1 = np.float32(pts1).reshape(-1, 2)
    pts2 = np.float32(pts2).reshape(-1, 2)
    P1 = cameraMatrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = cameraMatrix @ np.hstack((rotationMatrix, translationVector))
    print("Triangulating points...")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    try:
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    except Exception as e:
        return {"Error": f"cv2.triangulatePoints failed: {e}"}

    valid_4d_mask = np.abs(points_4d_hom[3]) > 1e-6
    if not np.any(valid_4d_mask):
        return {"Error": "Triangulation resulted in invalid homogeneous coordinates (all W near zero)."}
    points_4d_hom_valid = points_4d_hom[:, valid_4d_mask]
    pts1_valid_4d = pts1[valid_4d_mask]
    points_3d = points_4d_hom_valid[:3] / points_4d_hom_valid[3]
    points_3d = points_3d.T

    points_3d_cam1 = points_3d
    points_3d_cam2 = (rotationMatrix @ points_3d.T + translationVector).T
    filter_z_max = 1000.0 
    valid_mask = (points_3d_cam1[:, 2] > 0) & (points_3d_cam2[:, 2] > 0) & (points_3d_cam1[:, 2] < filter_z_max) & \
                 (pts1_valid_4d[:, 0] >= 0) & (pts1_valid_4d[:, 0] < img_width) & \
                 (pts1_valid_4d[:, 1] >= 0) & (
                     pts1_valid_4d[:, 1] < img_height)

    points_3d_filtered = points_3d[valid_mask]
    pts1_filtered = pts1_valid_4d[valid_mask]
    print(f"Triangulation: {len(points_4d_hom.T)} points initially -> {np.sum(valid_4d_mask)} with valid W -> {len(points_3d_filtered)} points after Z/bounds filtering.")
    if len(points_3d_filtered) == 0:
        return {"Error": "No valid 3D points generated after filtering."}

    colors_bgr = []
    for pt in pts1_filtered:
        if cancel_event and cancel_event.is_set():
            return {"Error": "Task canceled by user."}
        x, y = np.clip(np.int32(np.round(pt)), [0, 0], [
                       img_width - 1, img_height - 1])
        colors_bgr.append(imgL_color[y, x])
    colors_bgr = np.array(colors_bgr)
    if colors_bgr.shape[0] != points_3d_filtered.shape[0]:
        print(
            f"Warning: Color array size mismatch ({colors_bgr.shape[0]}) vs points ({points_3d_filtered.shape[0]}). Check filtering logic.")
        colors_rgb_normalized = np.full(
            (len(points_3d_filtered), 3), 0.5)
    else:
        colors_rgb_normalized = colors_bgr[:, ::-1] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d_filtered)
    if colors_rgb_normalized.shape[0] == points_3d_filtered.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb_normalized)
    else:
        print("Warning: Color array size mismatch persists. Point cloud will be uncolored.")

    print("Displaying *colored* point cloud in Open3D window...")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    try:
        o3d.visualization.draw_geometries(
            [pcd], window_name="Triangulated Point Cloud (SIFT/ORB etc.)")
        print("Open3D window closed.")
    except Exception as e:
        print(f"Error displaying Open3D window: {e}")
        return {"Error": f"Open3D display failed: {e}"}

    return {"3D Points": points_3d_filtered}

# --- Disparity Calculation Function ---
def disparity_calculation(imgL_rect, imgR_rect, guide_image=None, cancel_event=None):
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    print("--- Disparity Calculation using SGBM + WLS ---")
    if imgL_rect is None or imgR_rect is None:
        return {"Error": "Rectified images not provided."}
    if len(imgL_rect.shape) == 3:
        left_gray = cv2.cvtColor(imgL_rect, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = imgL_rect.copy()
    if len(imgR_rect.shape) == 3:
        right_gray = cv2.cvtColor(imgR_rect, cv2.COLOR_BGR2GRAY)
    else:
        right_gray = imgR_rect.copy()
    if left_gray.dtype != np.uint8:
        left_gray = cv2.convertScaleAbs(left_gray)
    if right_gray.dtype != np.uint8:
        right_gray = cv2.convertScaleAbs(right_gray)
    if left_gray.shape != right_gray.shape:
        return {"Error": f"Rectified images have different shapes: {left_gray.shape} vs {right_gray.shape}."}
    if guide_image is None:
        guide_image_wls = imgL_rect
    else:
        guide_image_wls = guide_image
    if len(guide_image_wls.shape) == 2:
        print("Warning: WLS guide image is grayscale. Color is often preferred.")
        guide_for_filter = guide_image_wls
    elif guide_image_wls.shape[2] == 3:
        guide_for_filter = guide_image_wls
    else:
        print(
            f"Warning: Unexpected guide image format: {guide_image_wls.shape}. Using left grayscale image as guide.")
        guide_for_filter = left_gray
    img_width = left_gray.shape[1]
    num_disparities = ((270 // 16) + 1) * 16 
    if num_disparities > img_width:
        print(
            f"Warning: Calculated numDisparities ({num_disparities}) > image width ({img_width}). Clamping.")
        num_disparities = max(16, (img_width // 16) * 16)
    min_disparity = 0
    block_size = 13
    P1 = 8 * 3 * 10**2
    P2 = 32 * 3 * 10**2
    disp12_max_diff = 1
    uniqueness_ratio = 15
    speckle_window_size = 200
    speckle_range = 2
    mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

    print(f"SGBM Params: minD={min_disparity}, numD={num_disparities}, blockS={block_size}, P1={P1}, P2={P2}, uniqueR={uniqueness_ratio}, speckleWin={speckle_window_size}, speckleRng={speckle_range}, disp12MaxD={disp12_max_diff}, mode={mode}")
    stereo_left = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, blockSize=block_size, P1=P1, P2=P2,
                                        disp12MaxDiff=disp12_max_diff, uniquenessRatio=uniqueness_ratio, speckleWindowSize=speckle_window_size, speckleRange=speckle_range, mode=mode)
    print("Computing left disparity (SGBM)...")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    disparity_left_raw_scaled = stereo_left.compute(left_gray, right_gray)
    if disparity_left_raw_scaled is None:
        return {"Error": "SGBM computation failed (returned None)."}
    min_raw_s, max_raw_s, _, _ = cv2.minMaxLoc(disparity_left_raw_scaled)
    
    print(
        f"Raw SGBM disparity range (scaled int16): min={min_raw_s}, max={max_raw_s}")
    print("Computing right disparity for WLS...")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    try:
        stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)
        disparity_right_raw_scaled = stereo_right.compute(
            right_gray, left_gray)
    except AttributeError:
        return {"Error": "cv2.ximgproc not available. Install opencv-contrib-python."}
    except Exception as e:
        return {"Error": f"Failed to create right matcher or compute right disparity: {e}"}
    if disparity_right_raw_scaled is None:
        return {"Error": "Right disparity computation failed (returned None)."}
    lambda_wls = 1000000
    sigma_color_wls = 3
    
    print(f"WLS Parameters: lambda={lambda_wls}, sigmaColor={sigma_color_wls}")
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(
        matcher_left=stereo_left)
    wls_filter.setLambda(lambda_wls)
    wls_filter.setSigmaColor(sigma_color_wls)
    print("Applying WLS filter...")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    try:
        filtered_disparity_scaled = wls_filter.filter(
            disparity_left_raw_scaled, guide_for_filter, disparity_map_right=disparity_right_raw_scaled)
    except cv2.error as e:
        return {"Error": f"WLS filtering failed: {e}. Check image types/channels?"}
    except Exception as e:
        return {"Error": f"An unexpected error occurred during WLS filtering: {e}"}
    min_filt_s, max_filt_s, _, _ = cv2.minMaxLoc(filtered_disparity_scaled)
    print(
        f"Filtered disparity range (scaled int16): min={min_filt_s}, max={max_filt_s}")
    filtered_disparity_float = filtered_disparity_scaled.astype(
        np.float32) / 16.0

    valid_disp_mask = (filtered_disparity_float > min_disparity) & np.isfinite(
        filtered_disparity_float)
    if np.any(valid_disp_mask):
        min_vis_float = np.min(filtered_disparity_float[valid_disp_mask])
        max_vis_float = np.max(filtered_disparity_float[valid_disp_mask])
        print(
            f"Normalization range (float, valid > minD): min={min_vis_float:.2f}, max={max_vis_float:.2f}")
        if max_vis_float > min_vis_float:
            norm_image = np.zeros_like(filtered_disparity_float)
            cv2.normalize(src=filtered_disparity_float, dst=norm_image, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, mask=valid_disp_mask.astype(np.uint8))
            filtered_disparity_vis_gray = np.uint8(norm_image)
        else:
            print("Warning: All valid filtered disparities have the same value.")
            filtered_disparity_vis_gray = np.full_like(
                left_gray, 128, dtype=np.uint8)
            filtered_disparity_vis_gray[~valid_disp_mask] = 0
    else:
        print("Warning: No valid disparities (> min_disparity) found after filtering.")
        filtered_disparity_vis_gray = np.zeros_like(left_gray, dtype=np.uint8)

    filtered_disparity_vis_color = cv2.applyColorMap(
        filtered_disparity_vis_gray, cv2.COLORMAP_JET)
    filtered_disparity_vis_color[~valid_disp_mask] = [
        0, 0, 0]

    print("Disparity calculation and filtering complete.")
    return {
        "Disparity Map": filtered_disparity_vis_gray,
        "Disparity Map Color": filtered_disparity_vis_color,
        "Raw Disparity": filtered_disparity_float
    }

# --- Visualize Point Cloud from Disparity ---
def visualize_point_cloud_disparity(raw_disparity_map, Q, colors, cancel_event=None):
    if cancel_event and cancel_event.is_set():
        return "Task canceled by user."
    
    if raw_disparity_map is None or Q is None or colors is None:
        return "Error: Missing raw disparity, Q, or color image for 3D visualization."
    print("Reprojecting image to 3D...")
    
    try:
        if raw_disparity_map.dtype != np.float32:
            print("Warning: Input disparity map is not float32. Converting.")
            raw_disparity_map = raw_disparity_map.astype(np.float32)
        raw_disparity_map[raw_disparity_map == np.inf] = 0
        raw_disparity_map[raw_disparity_map == -np.inf] = 0
        points_3D = cv2.reprojectImageTo3D(
            raw_disparity_map, Q, handleMissingValues=True)
    except Exception as e:
        return f"Error in reprojectImageTo3D: {e}"
    h, w = raw_disparity_map.shape
    if points_3D.shape[:2] != (h, w) or colors.shape[:2] != (h, w):
        return f"Shape mismatch after reproject: Disp({h}x{w}), Pts({points_3D.shape}), Color({colors.shape})"
    
    if cancel_event and cancel_event.is_set():
        return "Task canceled by user."
    
    try:
        min_disp_val = 0.1
        max_depth_val = 1000.0
        mask = (raw_disparity_map > min_disp_val) & \
            np.isfinite(points_3D).all(axis=2) & \
               (points_3D[:, :, 2] > 0) & \
               (points_3D[:, :, 2] < max_depth_val)

        num_valid_mask = np.sum(mask)
        
        print(
            f"Points after reprojectImageTo3D: {h*w}. Points after masking (disp > {min_disp_val}, 0 < Z < {max_depth_val}, finite): {num_valid_mask}")
        if num_valid_mask == 0:
            return "Warning: No valid points found after re-projection and masking (check disparity range, Q matrix, and depth threshold)."
        valid_points = points_3D[mask]
        if len(colors.shape) != 3 or colors.shape[2] != 3:
            return "Error: Input 'colors' image is not a 3-channel BGR image."
        valid_colors_bgr = colors[mask]
        valid_colors_rgb = valid_colors_bgr[:, ::-1] / 255.0
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(valid_points)
        if valid_colors_rgb.shape[0] == valid_points.shape[0]:
            point_cloud.colors = o3d.utility.Vector3dVector(valid_colors_rgb)
        else:
            print(
                f"Warning: Mismatch between points ({valid_points.shape[0]}) and colors ({valid_colors_rgb.shape[0]}) after masking.")
        print(f"Creating point cloud with {len(valid_points)} points.")
        print("Displaying colored point cloud in Open3D window...")
        
        if cancel_event and cancel_event.is_set():
            return "Task canceled by user."
        
        try:
            o3d.visualization.draw_geometries(
                [point_cloud], window_name="Dense Point Cloud (Disparity)")
            print("Open3D window closed.")
        except Exception as e:
            print(f"Error displaying Open3D window: {e}")
            return f"Open3D display failed: {e}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error during point cloud creation/visualization: {e}"
    return None

# --- XFeat Helper Function (warp_corners_and_draw_matches) ---
def draw_xfeat_matches(ref_points, dst_points, img1, img2, cancel_event=None):
    if cancel_event and cancel_event.is_set():
        return cv2.hconcat([img1, img2])
    
    if ref_points is None or dst_points is None:
        print("Warning: No points provided for drawing matches.")
        return cv2.hconcat([img1, img2])

    if len(ref_points) != len(dst_points):
        print("Warning: Mismatch in number of reference and destination points. Cannot draw matches.")
        return cv2.hconcat([img1, img2])

    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]

    matches = [cv2.DMatch(i, i, 0) for i in range(len(ref_points))]

    img_matches = cv2.drawMatches(
        img1, keypoints1,
        img2, keypoints2,
        matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_DEFAULT
    )

    return img_matches

# --- XFeat Matching Function ---
def xfeat_matching(stereo_path, cancel_event=None):
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    print("--- XFeat Matching (Experimental) ---")

    left_image_path = os.path.join(stereo_path, "img1.jpg")
    right_image_path = os.path.join(stereo_path, "img2.jpg")
    if not os.path.exists(left_image_path) or not os.path.exists(right_image_path):
        return {"Error": "Missing img1.jpg or img2.jpg in the specified folder."}

    print("Loading images...")
    imgL_color = cv2.imread(left_image_path)
    imgR_color = cv2.imread(right_image_path)
    if imgL_color is None or imgR_color is None:
        return {"Error": "Failed to load images."}
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    try:
        print("Initializing XFeat...")
        xfeat = XFeat()
        print("Running XFeat matching...")
        mkpts_0, mkpts_1 = xfeat.match_xfeat(
            imgL_color, imgR_color, top_k=4096)
        num_raw_matches = len(mkpts_0) if mkpts_0 is not None else 0
        print(f"XFeat found {num_raw_matches} raw matches.")
        if mkpts_0 is None or mkpts_1 is None or num_raw_matches < 8:
            return {"Error": f"XFeat matching failed or found too few points ({num_raw_matches})."}
    except ImportError:
        return {"Error": "XFeat or Torch library not found/installed."}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"Error": f"Error during XFeat matching: {e}"}

    print("Drawing matches...")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    canvas = draw_xfeat_matches(mkpts_0, mkpts_1, imgL_color, imgR_color, cancel_event)

    return {
        "mkpts_0": mkpts_0,
        "mkpts_1": mkpts_1,
        "Matched Image": canvas,
        "Left Color Image": imgL_color,
        "Num Matches": num_raw_matches,
    }

# --- XFeat Reconstruction Function ---
def xfeat_reconstruction(pts1, pts2, cameraMatrix, rotationMatrix, translationVector, imgL_color, cancel_event=None):
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    print("--- XFeat 3D Reconstruction (Experimental) ---")

    if pts1 is None or pts2 is None or len(pts1) == 0 or len(pts2) == 0 or len(pts1) != len(pts2):
        return {"Error": "Invalid or mismatched keypoints provided for reconstruction."}
    if cameraMatrix is None or rotationMatrix is None or translationVector is None:
        return {"Error": "Missing Camera Matrix (K), Rotation (R), or Translation (T)."}
    if imgL_color is None:
        return {"Error": "Missing left color image for coloring points."}

    img_height, img_width = imgL_color.shape[:2]
    pts1 = np.float32(pts1).reshape(-1, 2)
    pts2 = np.float32(pts2).reshape(-1, 2)

    K0 = cameraMatrix
    R = rotationMatrix
    T = translationVector
    P1 = K0 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K0 @ np.hstack((R, T))

    print(f"Triangulating {len(pts1)} points using provided K, R, T...")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    try:
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    except Exception as e:
        return {"Error": f"cv2.triangulatePoints failed: {e}"}

    valid_4d_mask = np.abs(points_4d_hom[3]) > 1e-6
    if not np.any(valid_4d_mask):
        return {"Error": "Triangulation resulted in invalid homogeneous coordinates (all W near zero)."}

    points_4d_hom_valid = points_4d_hom[:, valid_4d_mask]
    pts1_valid_4d = pts1[valid_4d_mask]

    points_3d = points_4d_hom_valid[:3] / points_4d_hom_valid[3]
    points_3d = points_3d.T

    points_3d_cam1 = points_3d
    points_3d_cam2 = (R @ points_3d.T + T).T
    filter_z_max = 1000.0
    valid_mask = (points_3d_cam1[:, 2] > 0) & (points_3d_cam2[:, 2] > 0) & (points_3d_cam1[:, 2] < filter_z_max) & \
                 (pts1_valid_4d[:, 0] >= 0) & (pts1_valid_4d[:, 0] < img_width) & \
                 (pts1_valid_4d[:, 1] >= 0) & (
                     pts1_valid_4d[:, 1] < img_height)

    points_3d_filtered = points_3d[valid_mask]
    pts1_filtered = pts1_valid_4d[valid_mask]

    num_3d_points = len(points_3d_filtered)
    print(
        f"Triangulation (XFeat): {len(pts1)} initial points -> {np.sum(valid_4d_mask)} with valid W -> {num_3d_points} points after Z/bounds filtering.")

    if num_3d_points == 0:
        return {"Error": "No valid 3D points generated after filtering."}

    print("Extracting colors for 3D points...")
    colors_bgr = []
    for pt in pts1_filtered:
        if cancel_event and cancel_event.is_set():
            return {"Error": "Task canceled by user."}
        x, y = np.clip(np.int32(np.round(pt)), [0, 0], [
                       img_width - 1, img_height - 1])
        colors_bgr.append(imgL_color[y, x])
    colors_bgr = np.array(colors_bgr)
    if colors_bgr.shape[0] != points_3d_filtered.shape[0]:
        print(
            f"Warning: Color array size mismatch ({colors_bgr.shape[0]}) vs points ({points_3d_filtered.shape[0]}). Using gray.")
        colors_rgb_normalized = np.full((len(points_3d_filtered), 3), 0.5)
    else:
        colors_rgb_normalized = colors_bgr[:, ::-1] / 255.0

    print("Creating Open3D point cloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d_filtered)
    if colors_rgb_normalized.shape[0] == points_3d_filtered.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb_normalized)
    else:
        print("Warning: Color array size mismatch persists. Point cloud will be uncolored.")

    print("Attempting to display point cloud in Open3D window...")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Task canceled by user."}
    
    vis_error = None
    try:
        o3d.visualization.draw_geometries(
            [pcd], window_name="XFeat Sparse Point Cloud")
        print("Open3D window closed.")
    except Exception as e:
        vis_error = f"Open3D display failed: {e}"
        print(f"Error displaying Open3D window: {e}")

    results_dict = {
        "3D Points": points_3d_filtered,
        "Num 3D Points": num_3d_points
    }
    if vis_error:
        results_dict["Visualization Error"] = vis_error

    return results_dict