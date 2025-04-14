import os
import numpy as np
import cv2
import glob
import open3d as o3d
import sys
import re
from collections import defaultdict

# --- Helper Functions for File Handling ---

def _find_files(folder_path, prefixes=('cam1', 'cam2'), extensions=('.jpg', '.png')):
    """Finds files in a folder matching given prefixes and extensions."""
    found_files = []
    
    for ext in extensions:
        for prefix in prefixes:
            pattern_lower = os.path.join(folder_path, f"{prefix}_*{ext.lower()}")
            pattern_upper = os.path.join(folder_path, f"{prefix}_*{ext.upper()}")
            found_files.extend(glob.glob(pattern_lower))
            found_files.extend(glob.glob(pattern_upper))
            pattern_title = os.path.join(folder_path, f"{prefix}_*{ext.title()}")
            found_files.extend(glob.glob(pattern_title))
    
    filtered_files = []
    
    for f in set(found_files):
        basename = os.path.basename(f).lower()
        base_prefix_matches = any(basename.startswith(p.lower() + '_') for p in prefixes)
        base_ext_matches = any(basename.endswith(e.lower()) for e in extensions)
        if base_prefix_matches and base_ext_matches:
            filtered_files.append(f)
    return sorted(filtered_files)

def _parse_timestamp(filename):
    """Extracts timestamp from filenames like cam1_TIMESTAMP.jpg or cam2_TIMESTAMP.png"""
    basename = os.path.basename(filename)
    match = re.match(r"^(?:cam1|cam2)_([^\.]+)\.(?:jpg|png)$", basename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def _find_stereo_pair(folder_path, cancel_event=None):
    """Finds ONE stereo pair (cam1_TS.ext, cam2_TS.ext) in a folder."""
    if cancel_event and cancel_event.is_set():
        return None, None  # Early exit if cancelled

    cam1_files = _find_files(folder_path, prefixes=['cam1'])
    cam2_files = _find_files(folder_path, prefixes=['cam2'])

    if not cam1_files:
        print(f"Warn: No 'cam1' files in {folder_path}")
        return None, None

    if not cam2_files:
        print(f"Warn: No 'cam2' files in {folder_path}")
        return None, None

    cam2_timestamps = {_parse_timestamp(f): f for f in cam2_files if _parse_timestamp(f)}

    for cam1_path in cam1_files:
        if cancel_event and cancel_event.is_set():
            return None, None  # Exit loop if cancelled
        ts1 = _parse_timestamp(cam1_path)
        if ts1 and ts1 in cam2_timestamps:
            cam2_path = cam2_timestamps[ts1]
            print(f"Found pair: L='{os.path.basename(cam1_path)}', R='{os.path.basename(cam2_path)}'")
            return cam1_path, cam2_path

    print(f"Warn: No matching timestamp pair in {folder_path}")
    return None, None

# --- Stereo Calibration Function ---
def stereo_calib(calib_path, cancel_event=None):
    print("--- Stereo Calibration ---")
    checkerboard_size = (9, 7)
    criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objpoints, imgpoints_l, imgpoints_r = [], [], []
    
    all_files = _find_files(calib_path, prefixes=['cam1', 'cam2'], extensions=['.jpg', '.png'])
    
    if not all_files:
        return {"Error": f"No 'cam1'/'cam2' images (.jpg,.png) in '{calib_path}'."}
    
    files_by_timestamp = defaultdict(list)
    for f in all_files:
        if cancel_event and cancel_event.is_set():
            return {"Error": "Cancelled by user"}
        timestamp = _parse_timestamp(f)
        if timestamp:
            files_by_timestamp[timestamp].append(f)
        else:
            print(f"Warn: Skipping unexpected filename: {os.path.basename(f)}")
    
    valid_pairs = []
    
    for timestamp, files in files_by_timestamp.items():
        if cancel_event and cancel_event.is_set():
            return {"Error": "Cancelled by user"}
        if len(files) == 2:
            cam1_path, cam2_path = None, None
            for f_path in files:
                if os.path.basename(f_path).lower().startswith('cam1'):
                    cam1_path = f_path
                elif os.path.basename(f_path).lower().startswith('cam2'):
                    cam2_path = f_path
            if cam1_path and cam2_path:
                valid_pairs.append({'ts': timestamp, 'L': cam1_path, 'R': cam2_path})
            else:
                print(f"Warn: TS {timestamp}, 2 files, but not cam1/cam2: {files}")
        elif len(files) > 2:
            print(f"Warn: TS {timestamp} > 2 files. Skip. Files: {files}")
    
    if not valid_pairs:
        return {"Error": "No valid pairs (cam1_TS, cam2_TS) found."}
    
    print(f"Found {len(valid_pairs)} potential valid pairs.")
    valid_pairs_processed = 0
    img_shape = None
    valid_pairs.sort(key=lambda p: p['ts'])
    
    for i, pair_info in enumerate(valid_pairs):
        if cancel_event and cancel_event.is_set():
            return {"Error": "Cancelled by user"}
        img_l_path = pair_info['L']
        img_r_path = pair_info['R']
    
        print(f"Processing {i+1}/{len(valid_pairs)} (TS: {pair_info['ts']}): L='{os.path.basename(img_l_path)}', R='{os.path.basename(img_r_path)}'")
        sys.stdout.flush()
    
        img_l = cv2.imread(img_l_path)
        img_r = cv2.imread(img_r_path)
    
        if img_l is None or img_r is None:
            print(f"  Warn: Cannot read pair {i+1}. Skip.")
            continue
    
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        current_shape = gray_l.shape[::-1]
    
        if img_shape is None:
            img_shape = current_shape
        elif img_shape != current_shape or img_shape != gray_r.shape[::-1]:
            print(f"  Warn: Shape mismatch. Skip.")
            continue
    
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, checkerboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, checkerboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
        if ret_l and ret_r:
            print(f"  Found corners.")
            valid_pairs_processed += 1
            corners_l_subpix = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria_subpix)
            corners_r_subpix = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria_subpix)
            objpoints.append(objp)
            imgpoints_l.append(corners_l_subpix)
            imgpoints_r.append(corners_r_subpix)
        else:
            print(f"  Corners not found (L:{ret_l}, R:{ret_r}). Skip.")
    
    if valid_pairs_processed == 0:
        return {"Error": "No corners found in any valid pair."}
    
    if img_shape is None:
        return {"Error": "Cannot determine image shape."}
    
    print(f"\nCalibrating: {valid_pairs_processed} pairs, size {img_shape}.")
    sys.stdout.flush()
    
    print("Initial individual calib...")
    sys.stdout.flush()
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}
    
    ret_l, M1_init, d1_init, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
    ret_r, M2_init, d2_init, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)
    
    if not ret_l or not ret_r:
        return {"Error": "Initial calib failed."}
    
    print("Initial calib OK.")
    sys.stdout.flush()
    
    print("\nStereo calib...")
    sys.stdout.flush()
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}
    
    flags = cv2.CALIB_FIX_INTRINSIC
    ret_stereo, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, M1_init, d1_init, M2_init, d2_init, img_shape, criteria=criteria_stereo, flags=flags)
    
    if not ret_stereo:
        return {"Error": "Stereo calib failed."}
    
    print("Stereo calib OK.")
    print("M1:\n", M1)
    print("d1:\n", d1)
    print("M2:\n", M2)
    print("d2:\n", d2)
    print("R:\n", R)
    print("T:\n", T)
    print("E:\n", E)
    print("F:\n", F)
    print(f"Stereo Reprojection Error: {ret_stereo:.5f}")
    sys.stdout.flush()
    camera_model = {
        "M1": M1, "d1": d1, "M2": M2, "d2": d2, "R": R, "T": T,
        "E": E, "F": F, "image_size": img_shape, "Reprojection Error": ret_stereo
    }
    return camera_model

# --- Draw Epipolar Lines Function ---
def draw_epilines(img1, img2, lines, pts1, pts2, cancel_event=None):
    if img1 is None or img2 is None or lines is None or pts1 is None or pts2 is None:
        return img1, img2

    if len(pts1) == 0 or len(pts2) == 0:
        return img1, img2

    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1.copy()

    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2.copy()
    r, c = img1_color.shape[:2]

    pts1 = np.int32(pts1.reshape(-1, 2))
    pts2 = np.int32(pts2.reshape(-1, 2))
    lines = lines.reshape(-1, 3)

    for r_line, pt1_coords, pt2_coords in zip(lines, pts1, pts2):
        if cancel_event and cancel_event.is_set():
            return img1_color, img2_color  # Partial result on cancel
        color = tuple(np.random.randint(0, 255, 3).tolist())
        a, b, c_line = r_line
        try:
            if abs(b) > 1e-6:
                y0 = -c_line / b
                y1 = -(c_line + a * c) / b
                x0, x1 = 0, c
            elif abs(a) > 1e-6:
                x0 = -c_line / a
                x1 = x0
                y0, y1 = 0, r
            else:
                continue
            x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
            ret_clip, p1_clip, p2_clip = cv2.clipLine((0, 0, c, r), (x0, y0), (x1, y1))
            if ret_clip:
                img1_color = cv2.line(img1_color, tuple(p1_clip), tuple(p2_clip), color, 1)
            img1_color = cv2.circle(img1_color, tuple(pt1_coords), 5, color, -1)
            img2_color = cv2.circle(img2_color, tuple(pt2_coords), 5, color, -1)
        except OverflowError:
            print(f"Warn: Overflow drawing line. Skip.")
            continue
        except Exception as e_draw:
            print(f"Warn: Error drawing line/point: {e_draw}. Skip.")
            continue

    return img1_color, img2_color

# --- Stereo Rectification Function ---
def stereo_rect(stereo_path, M1, d1, M2, d2, R, T, F, image_size, cancel_event=None):
    print("--- Stereo Rectification (Using Calibrated Parameters) ---")
    left_image_path, right_image_path = _find_stereo_pair(stereo_path, cancel_event=cancel_event)

    if left_image_path is None or right_image_path is None:
        return {"Error": f"Could not find pair in '{stereo_path}'."}

    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}

    imgL_color = cv2.imread(left_image_path)
    imgR_color = cv2.imread(right_image_path)

    if imgL_color is None or imgR_color is None:
        return {"Error": "Failed to load images."}

    h, w = imgL_color.shape[:2]
    input_image_size = (w, h)

    if input_image_size != tuple(image_size):
        print(f"Warn: Input size {input_image_size} != calib size {tuple(image_size)}.")

    imgL_before_rec_draw = imgL_color.copy()
    imgR_before_rec_draw = imgR_color.copy()

    try:  # Epiline drawing before
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(imgL_color, None)
        kp2, des2 = sift.detectAndCompute(imgR_color, None)

        if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:50]
            if len(matches) > 0:
                pts1_vis = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2_vis = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                if F is not None:
                    print("Drawing epilines on originals using calibrated F...")
                    lines1 = cv2.computeCorrespondEpilines(pts2_vis, 2, F)
                    imgL_before_rec_draw, _ = draw_epilines(
                        imgL_before_rec_draw, imgR_color.copy(), lines1, pts1_vis, pts2_vis, cancel_event=cancel_event)
                    if cancel_event and cancel_event.is_set():
                        return {"Error": "Cancelled by user"}
                    lines2 = cv2.computeCorrespondEpilines(pts1_vis, 1, F)
                    imgR_before_rec_draw, _ = draw_epilines(
                        imgR_before_rec_draw, imgL_color.copy(), lines2, pts2_vis, pts1_vis, cancel_event=cancel_event)
                    if cancel_event and cancel_event.is_set():
                        return {"Error": "Cancelled by user"}
                else:
                    print("Warn: Calibrated F needed for pre-rect epilines.")
            else:
                print("Warn: No matches for pre-rect viz.")
        else:
            print("Warn: No keypoints for pre-rect viz.")

    except Exception as e_epi_pre:
        print(f"Warn: Pre-rect epiline viz error: {e_epi_pre}")

    print(f"Running stereoRectify, size={image_size}, alpha=0.0")
    sys.stdout.flush()

    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}

    try:
        T = T.astype(np.float64)  # Ensure T_ is float64 for cv2.stereoRectify
        R1, R2, _, _, Q, _, _ = cv2.stereoRectify(M1, d1, M2, d2, image_size, R, T, alpha=1)
        # Compute the projection matrices for the two cameras
        P1 = M1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for the first camera
        P2 = M2 @ np.hstack((R, T))  # Projection matrix for the second camera
        
    except Exception as e_rect:
        return {"Error": f"stereoRectify failed: {e_rect}"}

    if Q is None:
        return {"Error": "stereoRectify failed (Q is None)."}
    print("Q matrix OK.")

    sys.stdout.flush()

    try:
        mapL1, mapL2 = cv2.initUndistortRectifyMap(M1, d1, R1, P1, tuple(image_size), cv2.CV_32FC1)
    except Exception as e_mapL:
        return {"Error": f"initMap L failed: {e_mapL}"}

    try:
        mapR1, mapR2 = cv2.initUndistortRectifyMap(M2, d2, R2, P2, tuple(image_size), cv2.CV_32FC1)
    except Exception as e_mapR:
        return {"Error": f"initMap R failed: {e_mapR}"}

    try:
        imgL_rect_color = cv2.remap(imgL_color, mapL1, mapL2, interpolation=cv2.INTER_LANCZOS4)
        imgR_rect_color = cv2.remap(imgR_color, mapR1, mapR2, interpolation=cv2.INTER_LANCZOS4)
    except Exception as e_remap:
        return {"Error": f"remap failed: {e_remap}"}

    imgL_rect_gray = cv2.cvtColor(imgL_rect_color, cv2.COLOR_BGR2GRAY)
    imgR_rect_gray = cv2.cvtColor(imgR_rect_color, cv2.COLOR_BGR2GRAY)

    imgL_after_rec_draw = imgL_rect_color.copy()
    imgR_after_rec_draw = imgR_rect_color.copy()

    try:  # Epiline drawing after
        sift_rect = cv2.SIFT_create()
        kp1_rect, des1_rect = sift_rect.detectAndCompute(imgL_rect_gray, None)
        kp2_rect, des2_rect = sift_rect.detectAndCompute(imgR_rect_gray, None)
        if des1_rect is not None and des2_rect is not None and len(des1_rect) > 1 and len(des2_rect) > 1:
            bf_rect = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches_rect = sorted(bf_rect.match(des1_rect, des2_rect), key=lambda x: x.distance)[:50]
            if len(matches_rect) > 0:
                print("Drawing horizontal epilines on rectified...")
                h_rect, w_rect = imgL_after_rec_draw.shape[:2]
                for m in matches_rect:
                    if cancel_event and cancel_event.is_set():
                        return {"Error": "Cancelled by user"}
                    pt1_rect = tuple(np.int32(kp1_rect[m.queryIdx].pt))
                    pt2_rect = tuple(np.int32(kp2_rect[m.trainIdx].pt))
                    y = pt1_rect[1]
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    cv2.line(imgL_after_rec_draw, (0, y), (w_rect, y), color, 1)
                    cv2.line(imgR_after_rec_draw, (0, y), (w_rect, y), color, 1)
                    cv2.circle(imgL_after_rec_draw, pt1_rect, 5, color, -1)
                    cv2.circle(imgR_after_rec_draw, pt2_rect, 5, color, -1)
            else:
                print("Warn: No matches on rectified.")
        else:
            print("Warn: No keypoints on rectified.")

    except Exception as e_epi_post:
        print(f"Warn: Post-rect epiline viz error: {e_epi_post}")

    return {
        "Original Left": imgL_before_rec_draw,
        "Original Right": imgR_before_rec_draw,
        "Drawn Rectified Left": imgL_after_rec_draw,
        "Drawn Rectified Right": imgR_after_rec_draw,
        "Rectified Left": imgL_rect_gray,
        "Rectified Right": imgR_rect_gray,
        "Rectified Color Left": imgL_rect_color,
        "Rectified Color Right": imgR_rect_color,
        "disp2depth map": Q
    }

# --- Feature Detection and Matching Function --- 
def feat_detect_match(stereo_path, cameraMatrix1=None, cancel_event=None):
    print("--- Feature Detection & Matching (Standalone) ---")
    left_image_path, right_image_path = _find_stereo_pair(stereo_path, cancel_event=cancel_event)

    if left_image_path is None or right_image_path is None:
        return {"Error": f"Could not find pair in '{stereo_path}'."}

    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}

    imgL_color = cv2.imread(left_image_path)
    imgR_color = cv2.imread(right_image_path)

    if imgL_color is None or imgR_color is None:
        return {"Error": "Failed to load images."}

    imgL = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR_color, cv2.COLOR_BGR2GRAY)

    if cameraMatrix1 is None:
        h_img, w_img = imgL.shape[:2]
        f = max(w_img, h_img)
        K = np.array([[f, 0, w_img/2], [0, f, h_img/2], [0, 0, 1]], dtype=np.float32)
        print(f"Warn: No M1. Using default K:\n{K}")
    else:
        K = np.array(cameraMatrix1)
        print("Using M1 for E calculation.")

    sift = cv2.SIFT_create()

    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}

    kp_l, des_l = sift.detectAndCompute(imgL, None)
    kp_r, des_r = sift.detectAndCompute(imgR, None)

    nkp_l = len(kp_l) if kp_l is not None else 0
    nkp_r = len(kp_r) if kp_r is not None else 0

    if des_l is None or des_r is None or nkp_l < 8 or nkp_r < 8:
        return {"Error": f"Not enough KPs (L:{nkp_l}, R:{nkp_r}, need 8)."}

    print(f"SIFT: L:{nkp_l}, R:{nkp_r}.")

    FLANN_IDX = 1
    idx_p = dict(algorithm=FLANN_IDX, trees=5)
    srch_p = dict(checks=50)
    flann = cv2.FlannBasedMatcher(idx_p, srch_p)
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}

    matches = flann.knnMatch(des_l, des_r, k=2)
    n_raw = len(matches) if matches else 0

    if not matches:
        return {"Error": "No FLANN matches."}

    good, pts1l, pts2l, m_viz_raw = [], [], [], []
    
    for i, p in enumerate(matches):
        if cancel_event and cancel_event.is_set():
            return {"Error": "Cancelled by user"}
        if len(p) == 2:
            m, n = p
            m_viz_raw.append([m])
            if m.distance < 0.75 * n.distance:
                good.append(m)
                pts1l.append(kp_l[m.queryIdx].pt)
                pts2l.append(kp_r[m.trainIdx].pt)
        elif len(p) == 1:
            m_viz_raw.append(p)
    
    n_good = len(good)
    
    print(f"{n_good} good after ratio test.")
    
    if n_good < 8:
        return {"Error": f"Not enough good matches ({n_good}, need 8)."}
    
    pts1 = np.float32(pts1l).reshape(-1, 1, 2)
    pts2 = np.float32(pts2l).reshape(-1, 1, 2)
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}
    
    F_feat, mask_f = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    if F_feat is None:
        print("Warn: F calc failed.")
        F_feat = np.eye(3)
        inl_f = 0
    else:
        inl_f = np.sum(mask_f) if mask_f is not None else n_good
        print(f"Feature F inliers: {inl_f}/{n_good}")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}
    
    E_feat, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    R_feat, t_feat = np.eye(3), np.zeros((3, 1))
    
    if E_feat is None:
        print("Warn: E calc failed.")
        inl_e = 0
    elif mask_e is None:
        print("Warn: E mask None.")
        inl_e = n_good
    else:
        pts1e = pts1[mask_e.ravel() == 1]
        pts2e = pts2[mask_e.ravel() == 1]
        inl_e = len(pts1e)
        print(f"Feature E inliers: {inl_e}/{n_good}")
    
    if inl_e < 5:
        print(f"Warn: Need 5+ for recoverPose ({inl_e}).")
    else:
        if cancel_event and cancel_event.is_set():
            return {"Error": "Cancelled by user"}
        try:
            _, R_p, t_p, mask_p = cv2.recoverPose(E_feat, pts1e, pts2e, K)
        except cv2.error as e_rec:
            print(f"Warn: recoverPose failed: {e_rec}.")
        else:
            if R_p is None or t_p is None:
                print("Warn: recoverPose failed.")
            else:
                R_feat, t_feat = R_p, t_p
                inl_p = np.sum(mask_p) if mask_p is not None else inl_e
                print(f"Feature recoverPose inliers: {inl_p}/{inl_e}")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}
    
    imgL_kp = cv2.drawKeypoints(imgL_color, kp_l, None, color=(0, 255, 0), flags=0)
    imgR_kp = cv2.drawKeypoints(imgR_color, kp_r, None, color=(0, 255, 0), flags=0)
    img_raw = cv2.drawMatchesKnn(imgL_color, kp_l, imgR_color, kp_r,
                                 m_viz_raw, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_good = cv2.drawMatches(imgL_color, kp_l, imgR_color, kp_r,
                               good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return {
        "Left Image with Keypoints": imgL_kp,
        "Right Image with Keypoints": imgR_kp,
        "Matched Images Before Lowe's Ratio": img_raw,
        "Matched Images After Lowe's Ratio": img_good,
        "Left Aligned Keypoints": pts1.reshape(-1, 2),
        "Right Aligned Keypoints": pts2.reshape(-1, 2),
        "Fundamental Matrix Feature": F_feat,
        "Essential Matrix Feature": E_feat,
        "Rotation Matrix Feature": R_feat,
        "Translation Vector Feature": t_feat,
        "Left Color Image": imgL_color,
        "Num Keypoints Left": nkp_l,
        "Num Keypoints Right": nkp_r,
        "Num Raw Matches": n_raw,
        "Num Good Matches": n_good
    }

# --- Stereo Geometry Estimation Function ---
def stereo_geometry_estimation(stereo_path, cameraMatrix1=None, cancel_event=None):
    print("--- Stereo Geometry Estimation (Feature-Based) ---")
    left_image_path, right_image_path = _find_stereo_pair(stereo_path, cancel_event=cancel_event)
    
    if left_image_path is None or right_image_path is None:
        return {"Error": f"Could not find pair in '{stereo_path}'."}
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}
    
    imgL = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    
    if imgL is None or imgR is None:
        return {"Error": "Failed load images."}
    
    if cameraMatrix1 is None:
        h_img, w_img = imgL.shape[:2]
        f = max(w_img, h_img)
        K = np.array([[f, 0, w_img/2], [0, f, h_img/2], [0, 0, 1]], dtype=np.float32)
        print(f"Warn: No M1. Using default K:\n{K}")
    else:
        K = np.array(cameraMatrix1)
        print("Using M1 for E calculation.")
    
    sift = cv2.SIFT_create()
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}
    
    kp_l, des_l = sift.detectAndCompute(imgL, None)
    kp_r, des_r = sift.detectAndCompute(imgR, None)
    nkp_l = len(kp_l) if kp_l is not None else 0
    nkp_r = len(kp_r) if kp_r is not None else 0
    
    if des_l is None or des_r is None or nkp_l < 8 or nkp_r < 8:
        return {"Error": f"Not enough KPs (L:{nkp_l}, R:{nkp_r}, need 8)."}
    
    FLANN_IDX = 1
    idx_p = dict(algorithm=FLANN_IDX, trees=5)
    srch_p = dict(checks=50)
    flann = cv2.FlannBasedMatcher(idx_p, srch_p)
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}
    
    matches = flann.knnMatch(des_l, des_r, k=2)
    
    if not matches:
        return {"Error": "No FLANN matches."}
    
    good, pts1l, pts2l = [], [], []
    
    for p in matches:
        if cancel_event and cancel_event.is_set():
            return {"Error": "Cancelled by user"}
        if len(p) == 2:
            m, n = p
            if m.distance < 0.7 * n.distance:
                good.append(m)
                pts1l.append(kp_l[m.queryIdx].pt)
                pts2l.append(kp_r[m.trainIdx].pt)
    
    n_good = len(good)
    
    if n_good < 8:
        return {"Error": f"Not enough good matches ({n_good}, need 8)."}
    
    pts1 = np.float32(pts1l).reshape(-1, 1, 2)
    pts2 = np.float32(pts2l).reshape(-1, 1, 2)
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}
    
    F, mask_f = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    if F is None:
        print("Warn: F calc failed.")
        F = np.eye(3)
    else:
        print(f"F RANSAC inliers: {np.sum(mask_f) if mask_f is not None else 'N/A'}")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}
    
    E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    det_E, svd_E, nz_svd_E = None, None, None
    R, T_v = None, None
    
    if E is None:
        print("Warn: E calc failed.")
    elif mask_e is None:
        print("Warn: E mask None.")
    else:
        det_E = np.linalg.det(E)
    
    try:
        svd_E = np.linalg.svd(E, compute_uv=False)
        nz_svd_E = np.sum(svd_E > 1e-6)
    except np.linalg.LinAlgError:
        print("Warn: SVD for E failed.")
    
    pts1e = pts1[mask_e.ravel() == 1]
    pts2e = pts2[mask_e.ravel() == 1]
    n_inl_e = len(pts1e)
    print(f"E RANSAC inliers: {n_inl_e}")
    
    if n_inl_e < 5:
        print(f"Warn: Need 5+ for recoverPose ({n_inl_e}).")
    else:
        if cancel_event and cancel_event.is_set():
            return {"Error": "Cancelled by user"}
        try:
            _, R_p, T_v_p, mask_p = cv2.recoverPose(E, pts1e, pts2e, K)
        except cv2.error as e_rec:
            print(f"Warn: recoverPose failed: {e_rec}.")
        else:
            if R_p is None or T_v_p is None:
                print("Warn: recoverPose failed.")
            else:
                R, T_v = R_p, T_v_p
                print(f"recoverPose inliers: {np.sum(mask_p) if mask_p is not None else 'N/A'}")
    
    if R is None:
        R = np.eye(3)
    if T_v is None:
        T_v = np.zeros((3, 1))
    
    return {
        "Fundamental Matrix": F,
        "Essential Matrix": E,
        "Determinant E": det_E,
        "Singular Values E": svd_E,
        "Non Zero SVD E": nz_svd_E,
        "Rotation Matrix": R,
        "Translation Vector": T_v
    }

# --- Triangulation Function ---
def triangulation_and_3D_reconstruction(pts1_matched, pts2_matched, cameraMatrix1, rotationMatrix, translationVector, imgL_color, cancel_event=None):
    print("--- Triangulation (Using Matched Points and Calibrated Pose) ---")
    if pts1_matched is None or pts2_matched is None or len(pts1_matched) != len(pts2_matched) or len(pts1_matched) == 0:
        return {"Error": "Invalid/mismatched keypoints."}
    if cameraMatrix1 is None or rotationMatrix is None or translationVector is None:
        return {"Error": "Missing calibrated K (M1), R, or T."}
    if imgL_color is None:
        return {"Error": "Missing left color image."}

    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}

    img_height, img_width = imgL_color.shape[:2]
    pts1 = np.float32(pts1_matched).reshape(-1, 2)
    pts2 = np.float32(pts2_matched).reshape(-1, 2)
    P1 = cameraMatrix1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = cameraMatrix1 @ np.hstack((rotationMatrix, translationVector))
    print(f"Triangulating {len(pts1)} points...")
    sys.stdout.flush()

    try:
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    except Exception as e_tri:
        return {"Error": f"triangulatePoints failed: {e_tri}"}

    # Check for valid homogeneous coordinate (W != 0)
    valid_4d_mask = np.abs(points_4d_hom[3]) > 1e-6
    num_valid_w = np.sum(valid_4d_mask)
    if num_valid_w == 0:
        return {"Error": "Triangulation resulted in invalid W for all points."}

    points_4d_hom_valid = points_4d_hom[:, valid_4d_mask]
    pts1_valid_for_color = pts1[valid_4d_mask]

    # Convert to 3D
    points_3d = (points_4d_hom_valid[:3] / points_4d_hom_valid[3]).T

    # --- Filtering ---
    z_positive_mask = points_3d[:, 2] > 0
    filter_z_max = 10000.0
    max_depth_mask = points_3d[:, 2] < filter_z_max
    valid_mask_combined = z_positive_mask & max_depth_mask

    points_3d_filtered = points_3d[valid_mask_combined]
    pts1_filtered = pts1_valid_for_color[valid_mask_combined]

    print(f"Triangulation filtering:")
    print(f"  Initial points: {len(pts1)}")
    print(f"  Valid W coord: {num_valid_w}")
    print(f"  Positive Z (Cam1): {np.sum(z_positive_mask)}")
    print(f"  Below max depth ({filter_z_max:.1f}): {np.sum(max_depth_mask)}")
    print(f"  Final valid points: {len(points_3d_filtered)}")
    sys.stdout.flush()

    if len(points_3d_filtered) == 0:
        return {"Error": "No valid 3D points after relaxed filtering (check calibration/matches)."}

    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}

    # --- Get Colors ---
    colors_bgr = []
    for pt in pts1_filtered:
        if cancel_event and cancel_event.is_set():
            return {"Error": "Cancelled by user"}
        x, y = np.clip(np.int32(np.round(pt)), [0, 0], [img_width - 1, img_height - 1])
        colors_bgr.append(imgL_color[y, x])
    colors_bgr = np.array(colors_bgr)
    colors_rgb_normalized = colors_bgr[:, ::-1] / 255.0

    # --- Create and Visualize Point Cloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d_filtered)
    
    if colors_rgb_normalized.shape[0] == points_3d_filtered.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb_normalized)
        print("Created colored point cloud.")
    else:
        print("Warn: Color array size mismatch.")

    print("Displaying point cloud...")
    sys.stdout.flush()
    warning_msg = None
    
    try:
        o3d.visualization.draw_geometries([pcd])
        print("Open3D window closed.")
    except Exception as e:
        warning_msg = f"Open3D display failed: {e}"
        print(warning_msg)

    result_dict = {"3D Points": points_3d_filtered}
    if warning_msg:
        result_dict["Warning"] = warning_msg
    return result_dict

# --- Disparity Calculation Function ---
def disparity_calculation(imgL_rect, imgR_rect, guide_image=None, cancel_event=None):
    print("--- Disparity Calculation using SGBM + WLS ---")
    if imgL_rect is None or imgR_rect is None:
        return {"Error": "Rectified images not provided."}
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}

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
        return {"Error": f"Shape mismatch: {left_gray.shape} vs {right_gray.shape}."}
    
    if guide_image is None:
        print("Warn: No guide image. Using left.")
        guide = imgL_rect
    else:
        guide = guide_image
    
    if len(guide.shape) == 2:
        print("Using grayscale guide.")
    elif guide.shape[2] == 3:
        print("Using color guide.")
    else:
        print(f"Warn: Bad guide format {guide.shape}. Using left gray.")
        guide = left_gray

    img_w = left_gray.shape[1]
    num_disp = ((img_w // 8) // 16) * 16
    num_disp = max(16, num_disp)
    min_disp = 0
    block_sz = 5
    P1 = 8 * 1 * block_sz ** 2
    P2 = 32 * 1 * block_sz ** 2
    disp12Max = 1
    unique = 10
    speckleWin = 100
    speckleRng = 2
    mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    
    print(f"SGBM: minD={min_disp}, numD={num_disp}, blockS={block_sz}, P1={P1}, P2={P2}, uniqueR={unique}, speckleWin={speckleWin}, speckleRng={speckleRng}, disp12MaxD={disp12Max}, mode={mode}")
    sys.stdout.flush()
    
    # Note: Using StereoBM instead of StereoSGBM as per original code
    stereo_l = cv2.StereoBM_create()

    lambda_w = 100000.0
    sigma_c = 3
    
    try:
        stereo_r = cv2.ximgproc.createRightMatcher(stereo_l)
    except AttributeError:
        return {"Error": "Need opencv-contrib-python for ximgproc."}
    except Exception as e:
        return {"Error": f"createRightMatcher failed: {e}"}
    
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_l)
    wls.setLambda(lambda_w)
    wls.setSigmaColor(sigma_c)
    
    print(f"WLS: lambda={lambda_w:.1f}, sigma={sigma_c:.1f}")
    sys.stdout.flush()
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}
    
    print("Computing left disp...")
    disp_l_raw = stereo_l.compute(left_gray, right_gray)
    
    if disp_l_raw is None:
        return {"Error": "SGBM left failed."}
    min_r, max_r, _, _ = cv2.minMaxLoc(disp_l_raw)
    
    print(f"Raw SGBM range (scaled): min={min_r}, max={max_r}")
    print("Computing right disp...")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}
    
    try:
        disp_r_raw = stereo_r.compute(right_gray, left_gray)
    except Exception as e:
        return {"Error": f"compute right failed: {e}"}
    
    if disp_r_raw is None:
        return {"Error": "Right disp failed."}
    
    print("Applying WLS filter...")
    
    if cancel_event and cancel_event.is_set():
        return {"Error": "Cancelled by user"}
    
    try:
        if guide.shape[:2] != disp_l_raw.shape[:2]:
            guide_res = cv2.resize(guide, (disp_l_raw.shape[1], disp_l_raw.shape[0]))
        else:
            guide_res = guide
        filtered_disp_s = wls.filter(disp_l_raw, guide_res, disparity_map_right=disp_r_raw)
    except cv2.error as e:
        return {"Error": f"WLS filter failed: {e}."}
    except Exception as e:
        return {"Error": f"WLS filter error: {e}"}
    
    min_f, max_f, _, _ = cv2.minMaxLoc(filtered_disp_s)
    print(f"Filtered range (scaled): min={min_f}, max={max_f}")
    filtered_disp_f = filtered_disp_s.astype(np.float32) / 16.0
    
    valid_disp_mask = (filtered_disp_s > (min_disp * 16)) & np.isfinite(filtered_disp_f)
    filtered_vis_gray = np.zeros_like(left_gray, dtype=np.uint8)
    
    if np.any(valid_disp_mask):
        min_v = np.min(filtered_disp_f[valid_disp_mask])
        max_v = np.max(filtered_disp_f[valid_disp_mask])
        print(f"Normalization range (float, valid): min={min_v:.2f}, max={max_v:.2f}")
        if max_v > min_v:
            norm_image = cv2.normalize(
                src=filtered_disp_f, dst=None, alpha=0, beta=255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=valid_disp_mask.astype(np.uint8)
            )
            np.copyto(filtered_vis_gray, norm_image, where=valid_disp_mask)
        else:
            filtered_vis_gray[valid_disp_mask] = 128
    else:
        print("Warn: No valid disparities found after filter for normalization.")
    
    filtered_vis_color = cv2.applyColorMap(filtered_vis_gray, cv2.COLORMAP_JET)
    filtered_vis_color[~valid_disp_mask] = [0, 0, 0]
    print("Disparity calculation complete.")
    sys.stdout.flush()
    return {
        "Disparity Map": filtered_vis_gray,
        "Disparity Map Color": filtered_vis_color,
        "Raw Disparity": filtered_disp_f
    }

# --- Visualize Point Cloud from Disparity ---
def visualize_point_cloud_disparity(raw_disparity_map, Q, colors, cancel_event=None):
    if raw_disparity_map is None or Q is None or colors is None:
        return "Error: Missing inputs."
    
    if cancel_event and cancel_event.is_set():
        return "Cancelled by user"
    
    print("Reprojecting to 3D...")
    sys.stdout.flush()
    
    try:
        if raw_disparity_map.dtype != np.float32:
            raw_disparity_map = raw_disparity_map.astype(np.float32)
        if not np.all(np.isfinite(Q)):
            return "Error: Q not finite."
        points_3D = cv2.reprojectImageTo3D(raw_disparity_map, Q, handleMissingValues=True)
    except Exception as e:
        return f"reprojectImageTo3D failed: {e}"

    h, w = raw_disparity_map.shape
    if points_3D.shape[:2] != (h, w) or colors.shape[:2] != (h, w):
        return f"Shape mismatch: Disp({h}x{w}), Pts({points_3D.shape}), Color({colors.shape})"

    try:
        min_disp_val = 0.01
        max_depth_val = 10000.0

        print(f"Filtering points: Disp > {min_disp_val}, Depth < {max_depth_val}, IsFinite")
        
        disp_mask = (raw_disparity_map > min_disp_val)
        num_after_disp = np.sum(disp_mask)
        print(f"  Points after min disparity filter: {num_after_disp}")

        finite_mask = np.isfinite(points_3D).all(axis=2)
        num_after_finite = np.sum(disp_mask & finite_mask)
        print(f"  Points after finite check: {num_after_finite}")

        final_mask = disp_mask & finite_mask
        num_valid_final = np.sum(final_mask)
        
        if num_valid_final == 0:
            print("Warn: No valid points remaining after all filters.")
            return "Warning: No valid points after filtering."

        valid_pts = points_3D[final_mask]

        if len(colors.shape) != 3 or colors.shape[2] != 3:
            return "Error: Colors not 3-channel BGR."
        valid_bgr = colors[final_mask]
        valid_rgb = valid_bgr[:, ::-1] / 255.0

        if cancel_event and cancel_event.is_set():
            return "Cancelled by user"

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_pts)
        if valid_rgb.shape[0] == valid_pts.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(valid_rgb)
        else:
            print(f"Warn: Point/color mismatch ({valid_pts.shape[0]} vs {valid_rgb.shape[0]}).")

        print(f"Creating point cloud: {len(valid_pts)} points.")
        print("Displaying...")
        sys.stdout.flush()
        
        warning_msg = None
        try:
            o3d.visualization.draw_geometries([pcd])
            print("Open3D window closed.")
        except Exception as e:
            warning_msg = f"Open3D display failed: {e}"
            print(warning_msg)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Point cloud error: {e}"

    return warning_msg