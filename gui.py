import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import os
import numpy as np
import cv2
import glob
import os


def cam_calib(base_path):
    checkerboardsize = (9, 7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((checkerboardsize[0] * checkerboardsize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : checkerboardsize[0], 0 : checkerboardsize[1]].T.reshape(
        -1, 2
    )

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
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            if save_chessboard_corner_ann:
                fname_ = (
                    os.path.join(base_path, annotation_dir)
                    + "/"
                    + fname.split("/")[-1].split(".")[0]
                    + "corner_plot.jpg"
                )
                cv2.drawChessboardCorners(img, checkerboardsize, corners2, ret)
                cv2.imwrite(fname_, img)

    ret, cameraMatrix, dist, rvec, tvec = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Compute reprojection error
    reprojection_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvec[i], tvec[i], cameraMatrix, dist
        )
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        reprojection_error += error
    reprojection_error /= len(objpoints)

    return [("Camera Matrix",cameraMatrix), ("Distortion Parameters",dist), ("Reprojection Error",reprojection_error)]
    # print("Reprojection error:", reprojection_error)
    # print("Camera Calibrated:   ", ret)
    # print("Camera Matrix:   ", cameraMatrix)
    # print("Distortion Parameters:   ", dist)
    # print("Rotation Vector: ", rvec)
    # print("Translation Vector: ", tvec)


# Main GUI class
class NotebookGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Notebook Interface")
        self.root.geometry("600x400")

        # Create tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, expand=True, fill="both")

        # Tab 1: Camera Calibration
        self.create_cam_calib_tab()

        # Let Tkinter auto-size automatically adjust according to items in the previous tab
        self.root.geometry("")  
        self.root.update_idletasks()  
        width = self.root.winfo_reqwidth() + 50  
        height = self.root.winfo_reqheight() + 100
        self.root.geometry(f"{width}x{height}")

        # Add more tabs here as functionality grows

    def create_cam_calib_tab(self):
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Camera Calibration")

        # Folder path entry
        ttk.Label(tab1, text="Chesseboard Calibration Patterns Folder Path:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.path_entry = ttk.Entry(tab1, width=50)
        self.path_entry.grid(row=0, column=1, padx=5, pady=5)

        # Browse button
        browse_btn = ttk.Button(tab1, text="Browse", command=self.browse_folder)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)

        # Run button
        run_btn = ttk.Button(tab1, text="Run", command=self.run_cam_calib)
        run_btn.grid(row=1, column=1, pady=10)

        # Output display
        self.output_text = scrolledtext.ScrolledText(tab1, width=70, height=20, wrap=tk.WORD)
        self.output_text.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Chesseboard Calibration Patterns")
        if folder:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, folder)

    def run_cam_calib(self):
        folder_path = self.path_entry.get()
        if not folder_path:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Please provide a folder path. \n")
            return
        
        results = cam_calib(folder_path)
        
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Camera Calibrated: \n")
        for result in results:
            self.output_text.insert(tk.END, str(result[0]) + "\n" + str(result[1]) + "\n")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = NotebookGUI(root)
    root.mainloop()