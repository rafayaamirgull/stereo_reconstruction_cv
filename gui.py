import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import os
import numpy as np
import cv2
import glob

# Camera Calibration Function (unchanged)
def cam_calib(base_path):
    checkerboardsize = (9, 7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((checkerboardsize[0] * checkerboardsize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboardsize[0], 0:checkerboardsize[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob.glob(f"{base_path}/*.jpg")
    images = sorted([os.path.join(base_path, img_path) for img_path in images])

    annotation_dir = "chessboard_corners"
    save_chessboard_corner_ann = False
    if not os.path.exists(annotation_dir) and save_chessboard_corner_ann:
        os.mkdir(annotation_dir)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray,
            checkerboardsize,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            if save_chessboard_corner_ann:
                fname_ = os.path.join(base_path, annotation_dir) + "/" + fname.split("/")[-1].split(".")[0] + "corner_plot.jpg"
                cv2.drawChessboardCorners(img, checkerboardsize, corners2, ret)
                cv2.imwrite(fname_, img)

    ret, cameraMatrix, dist, rvec, tvec = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    reprojection_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvec[i], tvec[i], cameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        reprojection_error += error
    reprojection_error /= len(objpoints)

    return [("Camera Matrix", cameraMatrix), ("Distortion Parameters", dist), ("Reprojection Error", reprojection_error)]

# Draw Epipolar Lines Function (unchanged)
def draw_epilines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

# Updated Stereo Rectification Function
def stereo_rect(stereo_path, baseline=0.1, cameraMatrix=None):
    if cameraMatrix is None:
        cameraMatrix = np.array([[1000, 0, 1920/2], [0, 1000, 1080/2], [0, 0, 1]])  # Default if not provided

    left_image = glob.glob(f"{stereo_path}/img1.jpg")
    right_image = glob.glob(f"{stereo_path}/img2.jpg")

    if not left_image or not right_image:
        return "Error: Missing img1.jpg or img2.jpg in the folder."

    imgL = cv2.imread(left_image[0], cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_image[0], cv2.IMREAD_GRAYSCALE)

    K0 = np.array(cameraMatrix)
    K1 = np.array(cameraMatrix)
    R = np.eye(3)  # Identity matrix (no rotation)
    T = np.array([[baseline], [0], [0]])
    image_size = (int(3840), int(2160))

    # Use SIFT for feature detection and matching
    sift = cv2.SIFT_create()
    keypoints_left, descriptors_left = sift.detectAndCompute(imgL, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(imgR, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Use FLANN-based matcher to find correspondences
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)
    
    # Filter matches using the Lowe's ratio test
    pts_left = []
    pts_right = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            pts_left.append(keypoints_left[m.queryIdx].pt)
            pts_right.append(keypoints_right[m.trainIdx].pt)

    pts_left = np.int32(pts_left)
    pts_right = np.int32(pts_right)
    F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_LMEDS)

    # Select only inlier points
    pts_left = pts_left[mask.ravel() == 1]
    pts_right = pts_right[mask.ravel() == 1]

    # Compute the Essential Matrix
    E, mask = cv2.findEssentialMat(pts_left, pts_right, K0, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Decompose the Essential Matrix to recover R and T
    _, R, T, _ = cv2.recoverPose(E, pts_left, pts_right, K0)

    # Find epilines for original images
    left_before_lines = cv2.computeCorrespondEpilines(pts_right.reshape(-1, 1, 2), 2, F)
    left_before_lines = left_before_lines.reshape(-1, 3)
    imgL_before_rec, _ = draw_epilines(imgL, imgR, left_before_lines, pts_left, pts_right)

    right_before_lines = cv2.computeCorrespondEpilines(pts_left.reshape(-1, 1, 2), 1, F)
    right_before_lines = right_before_lines.reshape(-1, 3)
    imgR_before_rec, _ = draw_epilines(imgR, imgL, right_before_lines, pts_right, pts_left)

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K0, None, K1, None, image_size, R, T, alpha=1.0)

    # Rectify Images
    mapL1, mapL2 = cv2.initUndistortRectifyMap(K0, None, R1, P1, image_size, cv2.CV_32F)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(K1, None, R2, P2, image_size, cv2.CV_32F)

    imgL_rect = cv2.remap(imgL, mapL1, mapL2, interpolation=cv2.INTER_LINEAR)
    imgR_rect = cv2.remap(imgR, mapR1, mapR2, interpolation=cv2.INTER_LINEAR)

    # Feature detection and matching on rectified images
    sift = cv2.SIFT_create()
    keypoints_left_rect, descriptors_left_rect = sift.detectAndCompute(imgL_rect, None)
    keypoints_right_rect, descriptors_right_rect = sift.detectAndCompute(imgR_rect, None)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_left_rect, descriptors_right_rect, k=2)

    pts_left_rect = []
    pts_right_rect = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            pts_left_rect.append(keypoints_left_rect[m.queryIdx].pt)
            pts_right_rect.append(keypoints_right_rect[m.trainIdx].pt)

    pts_left_rect = np.int32(pts_left_rect)
    pts_right_rect = np.int32(pts_right_rect)
    F_rect, mask = cv2.findFundamentalMat(pts_left_rect, pts_right_rect, cv2.FM_LMEDS)

    pts_left_rect = pts_left_rect[mask.ravel() == 1]
    pts_right_rect = pts_right_rect[mask.ravel() == 1]

    # Draw epipolar lines on rectified images
    left_after_lines = cv2.computeCorrespondEpilines(pts_right_rect.reshape(-1, 1, 2), 2, F_rect)
    left_after_lines = left_after_lines.reshape(-1, 3)
    imgL_after_rec, _ = draw_epilines(imgL_rect, imgR_rect, left_after_lines, pts_left_rect, pts_right_rect)

    right_after_lines = cv2.computeCorrespondEpilines(pts_right_rect.reshape(-1, 1, 2), 1, F_rect)
    right_after_lines = right_after_lines.reshape(-1, 3)
    imgR_after_rec, _ = draw_epilines(imgR_rect, imgL_rect, right_after_lines, pts_right_rect, pts_left_rect)

    # Resize images for display
    imgL_before_rec = cv2.resize(imgL_before_rec, (640, 360))
    imgR_before_rec = cv2.resize(imgR_before_rec, (640, 360))
    imgL_after_rec = cv2.resize(imgL_after_rec, (640, 360))
    imgR_after_rec = cv2.resize(imgR_after_rec, (640, 360))

    # Return images for display
    return {
        "Original Left": imgL_before_rec,
        "Original Right": imgR_before_rec,
        "Rectified Left": imgL_after_rec,
        "Rectified Right": imgR_after_rec
    }

# Main GUI class
class NotebookGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Notebook Interface")
        self.root.geometry("1300x700")  # Adjusted size to fit resized images and controls

        # Create tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, expand=True, fill="both")

        self.cam_calib_results = None
        self.stereo_rect_results = None

        # Tab 1: Camera Calibration
        self.create_cam_calib_tab()

        # Tab 2: Stereo Rectification
        self.create_stereo_rect_tab()

    def create_cam_calib_tab(self):
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Camera Calibration")

        ttk.Label(tab1, text="Chesseboard Calibration Patterns Folder Path:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.cam_calib_path_entry = ttk.Entry(tab1, width=50)
        self.cam_calib_path_entry.grid(row=0, column=1, padx=5, pady=5)

        browse_btn = ttk.Button(tab1, text="Browse", command=self.cam_calib_browse_folder)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)

        run_btn = ttk.Button(tab1, text="Run", command=self.run_cam_calib)
        run_btn.grid(row=1, column=1, pady=10)

        self.cam_calib_output_text = scrolledtext.ScrolledText(tab1, width=70, height=20, wrap=tk.WORD)
        self.cam_calib_output_text.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

    def run_cam_calib(self):
        folder_path = self.cam_calib_path_entry.get()
        if not folder_path:
            self.cam_calib_output_text.delete(1.0, tk.END)
            self.cam_calib_output_text.insert(tk.END, "Please provide a folder path.\n")
            return
        
        self.cam_calib_results = cam_calib(folder_path)
        self.cam_calib_output_text.delete(1.0, tk.END)
        self.cam_calib_output_text.insert(tk.END, "Camera Calibrated:\n")
        for result in self.cam_calib_results:
            self.cam_calib_output_text.insert(tk.END, str(result[0]) + "\n" + str(result[1]) + "\n")

    def cam_calib_browse_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Chesseboard Calibration Patterns")
        if folder:
            self.cam_calib_path_entry.delete(0, tk.END)
            self.cam_calib_path_entry.insert(0, folder)

    def create_stereo_rect_tab(self):
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="Stereo Rectification")

        # Controls frame
        controls_frame = ttk.Frame(tab2)
        controls_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        ttk.Label(controls_frame, text="Stereo Image Pair Folder Path:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.stereo_rect_path_entry = ttk.Entry(controls_frame, width=50)
        self.stereo_rect_path_entry.grid(row=0, column=1, padx=5, pady=5)

        browse_btn = ttk.Button(controls_frame, text="Browse", command=self.stereo_rect_browse_folder)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(controls_frame, text="Baseline (meters):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.baseline_entry = ttk.Entry(controls_frame, width=10)
        self.baseline_entry.grid(row=1, column=1, padx=5, pady=5)
        self.baseline_entry.insert(0, "0.1")  # Default value

        run_btn = ttk.Button(controls_frame, text="Run", command=self.run_stereo_rect)
        run_btn.grid(row=2, column=1, pady=10)

        # Image display labels
        self.img_labels = {
            "Original Left": tk.Label(tab2),
            "Original Right": tk.Label(tab2),
            "Rectified Left": tk.Label(tab2),
            "Rectified Right": tk.Label(tab2)
        }
        for i, (title, label) in enumerate(self.img_labels.items()):
            ttk.Label(tab2, text=title).grid(row=3 + 2 * (i // 2), column=i % 2, pady=(0, 2), sticky="n")
            label.grid(row=4 + 2 * (i // 2), column=i % 2, padx=5, pady=5)

    def run_stereo_rect(self):
        folder_path = self.stereo_rect_path_entry.get()
        if not folder_path:
            for label in self.img_labels.values():
                label.config(image='')
                tk.Label(label.master, text="Please provide a folder path.").grid(row=7, column=0, columnspan=2)
            return
        
        # Get and validate baseline input
        baseline_str = self.baseline_entry.get().strip()
        try:
            baseline = float(baseline_str)
            if baseline <= 0:
                raise ValueError("Baseline must be positive.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid baseline value: {e}. Using default (0.1).")
            baseline = 0.1

        self.stereo_rect_results = stereo_rect(folder_path, baseline=baseline, cameraMatrix=self.cam_calib_results[0][1] if self.cam_calib_results else None)
        
        if isinstance(self.stereo_rect_results, str):  # Error case
            for label in self.img_labels.values():
                label.config(image='')
                tk.Label(label.master, text=self.stereo_rect_results).grid(row=7, column=0, columnspan=2)
            return

        # Display images
        for title, img in self.stereo_rect_results.items():
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = tk.PhotoImage(data=cv2.imencode('.png', img_rgb)[1].tobytes())
            self.img_labels[title].config(image=img_pil)
            self.img_labels[title].image = img_pil  # Keep a reference to avoid garbage collection

    def stereo_rect_browse_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Stereo Image Pairs")
        if folder:
            self.stereo_rect_path_entry.delete(0, tk.END)
            self.stereo_rect_path_entry.insert(0, folder)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = NotebookGUI(root)
    root.mainloop()