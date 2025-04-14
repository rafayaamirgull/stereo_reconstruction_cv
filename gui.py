import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import os
import numpy as np
import cv2
from PIL import Image, ImageTk
from io import StringIO
import sys
import backend  # Ensure backend is imported
import threading
import queue
import numpy.core.arrayprint as arrayprint

# Custom output redirection class
class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.output = StringIO()
        self.stdout_orig = sys.stdout
        self.stderr_orig = sys.stderr

    def write(self, string):
        self.output.write(string)
        try:
            if self.text_widget and self.text_widget.winfo_exists():
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
        except tk.TclError:
            pass
        except Exception as e:
            print(f"RedirectText error: {e}")

    def flush(self):
        pass

    def start_redirect(self):
        if not hasattr(self, 'stdout_orig') or self.stdout_orig is None:
            self.stdout_orig = sys.stdout
        if not hasattr(self, 'stderr_orig') or self.stderr_orig is None:
            self.stderr_orig = sys.stderr
        sys.stdout = self

    def stop_redirect(self):
        if isinstance(sys.stdout, RedirectText) and sys.stdout is self:
            if hasattr(self, 'stdout_orig') and self.stdout_orig:
                sys.stdout = self.stdout_orig
            else:
                sys.stdout = sys.__stdout__

    def get_output(self):
        return self.output.getvalue()

# Main GUI class
class NotebookGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stereo Vision Toolkit GUI")
        self.root.geometry("1500x1000")

        # --- Scrolling Setup ---
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        # --- Notebook ---
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.pack(pady=10, padx=10, expand=True, fill="both")

        # --- State & Image Refs ---
        self.cam_calib_results = None
        self.stereo_rect_results = None
        self.feat_detect_match_results = None
        self.stereo_geometry_results = None
        self.triangulation_results = None
        self.disparity_calculation_results = None
        self.sparse_xfeat_results = {}
        self.image_references = {}

        # --- Loading Indicator State ---
        self.loading_window = None
        self.task_queue = queue.Queue()
        self.cancel_event = None  # NEW: To signal task cancellation
        self.current_thread = None  # NEW: Track the current worker thread

        # --- Path Entries ---
        self.cam_calib_path_entry = None
        self.stereo_rect_path_entry = None
        self.feat_detect_match_path_entry = None
        self.stereo_geometry_estimation_path_entry = None
        self.sparse_xfeat_path_entry = None

        # --- Image Labels ---
        self.stereo_img_labels = {}
        self.feat_img_labels = {}
        self.disparity_img_label = None
        self.sparse_xfeat_match_label = None

        # --- Output Texts ---
        self.cam_calib_output_text = None
        self.feat_detect_match_output_text = None
        self.stereo_geometry_output_text = None
        self.triangulation_output_text = None
        self.disparity_output_text = None
        self.sparse_xfeat_output_text = None

        # --- Create Tabs ---
        self.create_cam_calib_tab()
        self.create_stereo_rect_tab()
        self.create_feat_detect_match_tab()
        self.create_stereo_geometry_tab()
        self.create_triangulation_and_reconstruction_tab()
        self.create_disparity_calculation_tab()
        self.create_sparse_xfeat_tab()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- Scroll Frame Configuration ---
    def _on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    # --- Mouse Wheel Binding/Unbinding for Scrolling ---
    def _bind_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        scroll_amount = 0
        if hasattr(event, 'delta') and event.delta != 0:
            scroll_amount = -1 * int(event.delta / abs(event.delta))
        elif hasattr(event, 'num'):
            if event.num == 4:
                scroll_amount = -1
            elif event.num == 5:
                scroll_amount = 1
        if scroll_amount != 0:
            self.canvas.yview_scroll(scroll_amount, "units")

    # --- Loading Indicator Methods ---
    def _show_loading(self, message="Processing..."):
        if self.loading_window is not None and self.loading_window.winfo_exists():
            self.loading_window.destroy()
        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.transient(self.root)
        self.loading_window.title("Working")
        self.loading_window.resizable(False, False)
        self.loading_window.grab_set()
        # NEW: Allow closing the window to cancel the task
        self.loading_window.protocol("WM_DELETE_WINDOW", self._cancel_task)
        ttk.Label(self.loading_window, text=message, font=("Helvetica", 12)).pack(pady=10, padx=20)
        pb = ttk.Progressbar(self.loading_window, orient='horizontal', mode='indeterminate', length=200)
        pb.pack(pady=(0, 15), padx=20, fill='x', expand=True)
        pb.start(15)
        self.loading_window.update_idletasks()
        root_x, root_y = self.root.winfo_x(), self.root.winfo_y()
        root_w, root_h = self.root.winfo_width(), self.root.winfo_height()
        load_w, load_h = self.loading_window.winfo_width(), self.loading_window.winfo_height()
        x = root_x + (root_w // 2) - (load_w // 2)
        y = root_y + (root_h // 2) - (load_h // 2)
        self.loading_window.geometry(f"+{x}+{y}")
        self.loading_window.lift()

    def _hide_loading(self):
        if self.loading_window is not None and self.loading_window.winfo_exists():
            self.loading_window.grab_release()
            self.loading_window.destroy()
            self.loading_window = None

    # NEW: Cancel the current task
    def _cancel_task(self):
        if self.cancel_event:
            print("Cancellation requested by user.")
            self.cancel_event.set()  # Signal the worker thread to stop
            self.task_queue.put(("CANCELED", {"Error": "Task canceled by user."}))
            self._hide_loading()

    # --- Threading and Queue Handling ---
    def _run_task_with_loading(self, task_func, args=(), kwargs={}, loading_msg="Processing...", result_handler=None):
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
        self.cancel_event = threading.Event()  # NEW: Create a cancellation event
        self._show_loading(loading_msg)

        def worker():
            try:
                if self.cancel_event.is_set():
                    return  # Exit early if already canceled
                result = task_func(*args, **kwargs)
                if not self.cancel_event.is_set():
                    self.task_queue.put(("SUCCESS", result))
                else:
                    self.task_queue.put(("CANCELED", {"Error": "Task canceled by user."}))
            except Exception as e:
                if self.cancel_event.is_set():
                    self.task_queue.put(("CANCELED", {"Error": "Task canceled by user."}))
                else:
                    import traceback
                    err_msg = f"Error in task: {e}\n{traceback.format_exc()}"
                    print(f"Error in worker thread: {e}")
                    error_payload = {"Error": err_msg}
                    self.task_queue.put(("ERROR", error_payload))

        self.current_thread = threading.Thread(target=worker, daemon=True)
        self.current_thread.start()
        self.root.after(100, self._check_queue, result_handler)

    def _check_queue(self, result_handler):
        try:
            status, result = self.task_queue.get_nowait()
            self._hide_loading()
            self.cancel_event = None  # NEW: Clear the cancellation event
            self.current_thread = None  # NEW: Clear the thread reference
            if status == "SUCCESS":
                if result_handler:
                    try:
                        result_handler(result)
                    except Exception as e:
                        print(f"Error in result handler: {e}")
                        import traceback
                        traceback.print_exc()
                        messagebox.showerror("GUI Error", f"Error processing result:\n{e}")
            elif status == "ERROR":
                print(f"Task failed:\n{result}")
                if isinstance(result, dict) and "Error" in result:
                    error_detail = result["Error"]
                else:
                    error_detail = str(result)
                short_error = error_detail.split('\n')[0]
                messagebox.showerror("Task Error", f"Operation failed:\n{short_error}\n\n(See console/log for details)")
                if result_handler:
                    try:
                        result_handler(result)
                    except Exception as e:
                        print(f"Error in result handler during error processing: {e}")
            elif status == "CANCELED":  # NEW: Handle cancellation
                print("Task was canceled.")
                if isinstance(result, dict) and "Error" in result:
                    error_detail = result["Error"]
                else:
                    error_detail = "Task canceled by user."
                messagebox.showinfo("Canceled", error_detail)
                if result_handler:
                    try:
                        result_handler(result)
                    except Exception as e:
                        print(f"Error in result handler during cancellation: {e}")

        except queue.Empty:
            if self.loading_window and self.loading_window.winfo_exists():
                self.root.after(100, self._check_queue, result_handler)
        except Exception as e:
            print(f"Error checking queue or handling result: {e}")
            import traceback
            traceback.print_exc()
            self._hide_loading()
            self.cancel_event = None
            self.current_thread = None
            messagebox.showerror("GUI Error", f"Error checking task status:\n{e}")

    # --- Image/GUI Update Methods ---
    def _clear_image_references(self):
        for widget in list(self.image_references.keys()):
            if widget is None:
                continue
            try:
                if widget.winfo_exists():
                    widget.config(image='')
            except tk.TclError:
                pass
            except Exception as e:
                print(f"Warning: Error clearing widget {widget}: {e}")
            self.image_references.pop(widget, None)

    def on_closing(self):
        print("Closing application...")
        if self.cancel_event:
            self.cancel_event.set()  # NEW: Cancel any running task
        self._hide_loading()
        self._clear_image_references()
        for attr_name in dir(self):
            if attr_name.startswith('redirector_'):
                redirector = getattr(self, attr_name, None)
                if isinstance(redirector, RedirectText):
                    try:
                        redirector.stop_redirect()
                    except Exception as e:
                        print(f"Error stopping redirector {attr_name}: {e}")
        try:
            self._unbind_mousewheel(None)
        except Exception as e:
            print(f"Error unbinding mousewheel: {e}")
        try:
            self.root.destroy()
        except Exception as e:
            print(f"Error destroying root window: {e}")

    def _display_image(self, label_widget, cv_image_pre_resized):
        if not label_widget or not label_widget.winfo_exists():
            self.image_references.pop(label_widget, None)
            return
        if cv_image_pre_resized is None or cv_image_pre_resized.size == 0:
            try:
                label_widget.config(image='')
                self.image_references.pop(label_widget, None)
            except tk.TclError:
                pass
            return
        img_display = cv_image_pre_resized
        try:
            img_pil = None
            if len(img_display.shape) == 2:
                img_pil = Image.fromarray(img_display)
            elif len(img_display.shape) == 3 and img_display.shape[2] == 3:
                img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
            else:
                print(f"Error: Unsupported image shape for display: {img_display.shape}")
            if img_pil:
                img_tk = ImageTk.PhotoImage(image=img_pil)
                if label_widget.winfo_exists():
                    label_widget.config(image=img_tk)
                    self.image_references[label_widget] = img_tk
                else:
                    self.image_references.pop(label_widget, None)
            elif label_widget.winfo_exists():
                label_widget.config(image='')
                self.image_references.pop(label_widget, None)
        except Exception as e:
            print(f"Error converting/displaying image: {e}")
            import traceback
            traceback.print_exc()
            try:
                if label_widget.winfo_exists():
                    label_widget.config(image='')
                    self.image_references.pop(label_widget, None)
            except tk.TclError:
                pass
            except Exception as e2:
                print(f"Error clearing image after error: {e2}")

    # --- Tab Creation Methods ---
    def create_cam_calib_tab(self):
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="1. Camera Calibration")
        controls_frame = ttk.Frame(tab1)
        controls_frame.pack(pady=5, padx=5, fill='x')
        ttk.Label(controls_frame, text="Chessboard Images Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.cam_calib_path_entry = ttk.Entry(controls_frame, width=60)
        self.cam_calib_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        browse_btn = ttk.Button(controls_frame, text="Browse", command=self.cam_calib_browse_folder)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        run_btn = ttk.Button(controls_frame, text="Run Calibration", command=self.run_cam_calib_threaded)
        run_btn.grid(row=1, column=1, pady=10)
        controls_frame.columnconfigure(1, weight=1)
        ttk.Label(tab1, text="Output Log & Results:").pack(pady=(10,0), padx=5, anchor='w')
        self.cam_calib_output_text = scrolledtext.ScrolledText(tab1, width=80, height=20, wrap=tk.WORD)
        self.cam_calib_output_text.pack(pady=5, padx=5, expand=True, fill='both')
        self.redirector_calib = RedirectText(self.cam_calib_output_text)

    def create_stereo_rect_tab(self):
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="2. Stereo Rectification")
        controls_frame = ttk.Frame(tab2)
        controls_frame.pack(pady=5, padx=5, fill='x')
        controls_frame.columnconfigure(1, weight=1)
        ttk.Label(controls_frame, text="Stereo Pair Folder (img1.jpg, img2.jpg):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.stereo_rect_path_entry = ttk.Entry(controls_frame, width=60)
        self.stereo_rect_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        browse_btn = ttk.Button(controls_frame, text="Browse", command=self.stereo_rect_browse_folder)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        run_btn = ttk.Button(controls_frame, text="Run Rectification", command=self.run_stereo_rect_threaded)
        run_btn.grid(row=1, column=1, pady=10)
        img_frame = ttk.Frame(tab2)
        img_frame.pack(pady=10, padx=5, expand=True, fill='both')
        self.stereo_img_labels = {}
        titles = ["Original Left", "Original Right", "Drawn Rectified Left", "Drawn Rectified Right"]
        cols = 2
        for i, title in enumerate(titles):
            frame = ttk.LabelFrame(img_frame, text=title)
            row, col = divmod(i, cols)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            img_frame.columnconfigure(col, weight=1)
            img_frame.rowconfigure(row, weight=1)
            label = tk.Label(frame)
            label.pack(padx=5, pady=5)
            self.stereo_img_labels[title] = label

    def create_feat_detect_match_tab(self):
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="3. Feature Detection & Matching")
        controls_frame = ttk.Frame(tab3)
        controls_frame.pack(pady=5, padx=5, fill='x')
        controls_frame.columnconfigure(1, weight=1)
        ttk.Label(controls_frame, text="Stereo Pair Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.feat_detect_match_path_entry = ttk.Entry(controls_frame, width=60)
        self.feat_detect_match_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        browse_btn = ttk.Button(controls_frame, text="Browse", command=self.feat_detect_match_browse_folder)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        run_btn = ttk.Button(controls_frame, text="Run Detection & Matching", command=self.run_feat_detect_match_threaded)
        run_btn.grid(row=1, column=1, pady=10)
        img_frame = ttk.Frame(tab3)
        img_frame.pack(pady=10, padx=5, expand=True, fill='both')
        img_frame.columnconfigure(0, weight=1)
        img_frame.columnconfigure(1, weight=1)
        img_frame.rowconfigure(0, weight=1)
        img_frame.rowconfigure(1, weight=1)
        img_frame.rowconfigure(2, weight=1)
        self.feat_img_labels = {}
        titles = ["Left Image with Keypoints", "Right Image with Keypoints", "Matched Images Before Lowe's Ratio", "Matched Images After Lowe's Ratio"]
        frame_kp_l = ttk.LabelFrame(img_frame, text=titles[0])
        frame_kp_l.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.feat_img_labels[titles[0]] = tk.Label(frame_kp_l)
        self.feat_img_labels[titles[0]].pack(padx=5, pady=5)
        frame_kp_r = ttk.LabelFrame(img_frame, text=titles[1])
        frame_kp_r.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.feat_img_labels[titles[1]] = tk.Label(frame_kp_r)
        self.feat_img_labels[titles[1]].pack(padx=5, pady=5)
        frame_match_raw = ttk.LabelFrame(img_frame, text=titles[2])
        frame_match_raw.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.feat_img_labels[titles[2]] = tk.Label(frame_match_raw)
        self.feat_img_labels[titles[2]].pack(padx=5, pady=5)
        frame_match_good = ttk.LabelFrame(img_frame, text=titles[3])
        frame_match_good.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.feat_img_labels[titles[3]] = tk.Label(frame_match_good)
        self.feat_img_labels[titles[3]].pack(padx=5, pady=5)
        log_frame = ttk.Frame(tab3)
        log_frame.pack(pady=(0,5), padx=5, fill='x')
        ttk.Label(log_frame, text="Output Log (Counts & Status):").pack(pady=(5,0), anchor='w')
        self.feat_detect_match_output_text = scrolledtext.ScrolledText(log_frame, width=80, height=5, wrap=tk.WORD)
        self.feat_detect_match_output_text.pack(pady=5, padx=0, fill='x', expand=False)
        self.redirector_feat = RedirectText(self.feat_detect_match_output_text)

    def create_stereo_geometry_tab(self):
        tab4 = ttk.Frame(self.notebook)
        self.notebook.add(tab4, text="4. Stereo Geometry")
        controls_frame = ttk.Frame(tab4)
        controls_frame.pack(pady=5, padx=5, fill='x')
        ttk.Label(controls_frame, text="Stereo Pair Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.stereo_geometry_estimation_path_entry = ttk.Entry(controls_frame, width=60)
        self.stereo_geometry_estimation_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        browse_btn = ttk.Button(controls_frame, text="Browse", command=self.stereo_geometry_estimation_browse_folder)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        run_btn = ttk.Button(controls_frame, text="Run", command=self.run_stereo_geometry_estimation_threaded)
        run_btn.grid(row=1, column=1, pady=10)
        controls_frame.columnconfigure(1, weight=1)
        ttk.Label(tab4, text="Output Log & Results:").pack(pady=(10,0), padx=5, anchor='w')
        self.stereo_geometry_output_text = scrolledtext.ScrolledText(tab4, width=80, height=20, wrap=tk.WORD)
        self.stereo_geometry_output_text.pack(pady=5, padx=5, expand=True, fill='both')
        self.redirector_geom = RedirectText(self.stereo_geometry_output_text)

    def create_triangulation_and_reconstruction_tab(self):
        tab5 = ttk.Frame(self.notebook)
        self.notebook.add(tab5, text="5. Triangulation (Feature-Based)")
        controls_frame = ttk.Frame(tab5)
        controls_frame.pack(pady=5, padx=5, fill='x', side='top')
        ttk.Label(controls_frame, text="Requires results from Tab 1 (Calibration) and Tab 3 (Matching).").pack(pady=5)
        run_btn = ttk.Button(controls_frame, text="Run Triangulation & Visualize 3D", command=self.run_triangulation_and_reconstruction_threaded)
        run_btn.pack(pady=10)
        output_frame = ttk.Frame(tab5)
        output_frame.pack(pady=5, padx=5, fill='x', side='top')
        ttk.Label(output_frame, text="Output Log & Status:").pack(pady=(5,0), anchor='w')
        self.triangulation_output_text = scrolledtext.ScrolledText(output_frame, width=80, height=10, wrap=tk.WORD)
        self.triangulation_output_text.pack(pady=5, padx=0, expand=False, fill='x')
        self.redirector_tri = RedirectText(self.triangulation_output_text)

    def create_disparity_calculation_tab(self):
        tab6 = ttk.Frame(self.notebook)
        self.notebook.add(tab6, text="6. Disparity & 3D (Dense)")
        controls_frame = ttk.Frame(tab6)
        controls_frame.pack(pady=5, padx=5, fill='x')
        controls_frame.columnconfigure(1, weight=1)
        ttk.Label(controls_frame, text="Requires results from Tab 1 (Calibration) and Tab 2 (Rectification).").grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        run_disparity_btn = ttk.Button(controls_frame, text="Run Disparity", command=self.run_disparity_calculation_threaded)
        run_disparity_btn.grid(row=1, column=0, pady=10, padx=5)
        run_3d_btn = ttk.Button(controls_frame, text="Visualize 3D Point Cloud", command=self.run_visualize_3d_threaded)
        run_3d_btn.grid(row=1, column=1, pady=10, padx=5)
        output_frame = ttk.Frame(tab6)
        output_frame.pack(pady=10, padx=5, expand=True, fill='both')
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        output_frame.rowconfigure(1, weight=0)
        img_frame = ttk.LabelFrame(output_frame, text="Filtered Disparity Heatmap (Normalized)")
        img_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        img_inner_frame = ttk.Frame(img_frame)
        img_inner_frame.pack(expand=True)
        self.disparity_img_label = tk.Label(img_inner_frame)
        self.disparity_img_label.pack(padx=5, pady=5)
        log_frame = ttk.Frame(output_frame)
        log_frame.grid(row=1, column=0, padx=5, pady=(10, 5), sticky="ew")
        ttk.Label(log_frame, text="Output Log (Status):").pack(pady=(0,2), anchor='w')
        self.disparity_output_text = scrolledtext.ScrolledText(log_frame, width=80, height=5, wrap=tk.WORD)
        self.disparity_output_text.pack(pady=(0,5), padx=0, fill='x', expand=False)
        self.redirector_disp = RedirectText(self.disparity_output_text)

    def create_sparse_xfeat_tab(self):
        tab7 = ttk.Frame(self.notebook)
        self.notebook.add(tab7, text="7. Sparse Reconstruction (XFeat Exp.)")
        controls_frame = ttk.Frame(tab7)
        controls_frame.pack(pady=5, padx=5, fill='x')
        controls_frame.columnconfigure(1, weight=1)
        ttk.Label(controls_frame, text="Stereo Pair Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.sparse_xfeat_path_entry = ttk.Entry(controls_frame, width=60)
        self.sparse_xfeat_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        browse_btn = ttk.Button(controls_frame, text="Browse", command=self.sparse_xfeat_browse_folder)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        dep_label_text = ("Matching requires images. "
                          "Reconstruction requires successful Matching results \n"
                          "AND successful Calibration (Tab 1 for K) "
                          "AND successful Feature Matching (Tab 3 for R, T).")
        ttk.Label(controls_frame, text=dep_label_text, foreground="blue").grid(row=1, column=0, columnspan=3, padx=5, pady=(5,5), sticky="w")
        button_frame = ttk.Frame(controls_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=5)
        match_btn = ttk.Button(button_frame, text="1. Run XFeat Matching", command=self.run_sparse_xfeat_matching_threaded)
        match_btn.pack(side=tk.LEFT, padx=10, pady=5)
        recon_btn = ttk.Button(button_frame, text="2. Run 3D Reconstruction", command=self.run_sparse_xfeat_reconstruction_threaded)
        recon_btn.pack(side=tk.LEFT, padx=10, pady=5)
        output_frame = ttk.Frame(tab7)
        output_frame.pack(pady=10, padx=5, expand=True, fill='both')
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        output_frame.rowconfigure(1, weight=0)
        img_frame = ttk.LabelFrame(output_frame, text="XFeat Matches Visualization")
        img_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        img_inner_frame = ttk.Frame(img_frame)
        img_inner_frame.pack(expand=True)
        self.sparse_xfeat_match_label = tk.Label(img_inner_frame)
        self.sparse_xfeat_match_label.pack(padx=5, pady=5)
        log_frame = ttk.Frame(output_frame)
        log_frame.grid(row=1, column=0, padx=5, pady=(10, 5), sticky="ew")
        ttk.Label(log_frame, text="Output Log (Status & Counts):").pack(pady=(0,2), anchor='w')
        self.sparse_xfeat_output_text = scrolledtext.ScrolledText(log_frame, width=80, height=8, wrap=tk.WORD)
        self.sparse_xfeat_output_text.pack(pady=(0,5), padx=0, fill='x', expand=False)
        self.redirector_xfeat = RedirectText(self.sparse_xfeat_output_text)

    # --- Action Methods ---
    def run_cam_calib_threaded(self):
        folder_path = self.cam_calib_path_entry.get()
        if self.cam_calib_output_text:
            self.cam_calib_output_text.delete(1.0, tk.END)
        else:
            print("Warning: cam_calib_output_text not initialized.")
            return
        if not folder_path or not os.path.isdir(folder_path):
            messagebox.showerror("Error", "Please select a valid folder path.")
            return
        if not hasattr(self, 'redirector_calib') or not self.redirector_calib:
            print("Error: redirector_calib not initialized.")
            return
        self.redirector_calib.start_redirect()
        print(f"Starting Camera Calibration: {folder_path}")
        self._run_task_with_loading(task_func=backend.cam_calib, args=(folder_path,), loading_msg="Running Calibration...", result_handler=self._handle_cam_calib_result)

    def _handle_cam_calib_result(self, results):
        redirector = getattr(self, 'redirector_calib', None)
        if isinstance(results, dict) and "Error" in results:
            error_msg = results["Error"]
            print(f"Error: {error_msg}")
            if redirector:
                redirector.stop_redirect()
            messagebox.showerror("Calibration Failed", error_msg)
            self.cam_calib_results = results
        elif isinstance(results, dict):
            self.cam_calib_results = results
            print("\n--- Calibration Results ---")
            with np.printoptions(precision=4, suppress=True):
                for key, value in self.cam_calib_results.items():
                    print(f"{key}:")
                    if isinstance(value, np.ndarray):
                        print(value)
                    elif isinstance(value, float):
                        print(f"{value:.4f}")
                    else:
                        print(value)
                    print("-" * 20)
            print("\nCalibration process completed successfully.")
            if redirector:
                redirector.stop_redirect()
            reproj_error = self.cam_calib_results.get('Reprojection Error', 'N/A')
            messagebox.showinfo("Success", f"Calibration complete.\nReprojection Error: {reproj_error:.4f}" if isinstance(reproj_error, float) else "Calibration complete.")
        else:
            err_msg = f"Unexpected calibration result type: {type(results)}. Content: {results}"
            print(f"Error: {err_msg}")
            if redirector:
                redirector.stop_redirect()
            messagebox.showerror("Error", err_msg)
            self.cam_calib_results = {"Error": err_msg}

    def run_stereo_rect_threaded(self):
        folder_path = self.stereo_rect_path_entry.get()
        for label in self.stereo_img_labels.values():
            self._display_image(label, None)
        if not folder_path or not os.path.isdir(folder_path):
            messagebox.showerror("Error", "Please select a valid folder path.")
            return
        if not self.cam_calib_results or "Error" in self.cam_calib_results:
            messagebox.showerror("Missing Data", "Run Camera Calibration (Tab 1) successfully first.")
            return
        camera_matrix = self.cam_calib_results.get("Camera Matrix")
        dist_coeffs = self.cam_calib_results.get("Distortion Parameters")
        if camera_matrix is None:
            messagebox.showerror("Missing Data", "Camera Matrix not found in Calibration results.")
            return
        print(f"Starting Stereo Rectification: {folder_path}")
        if dist_coeffs is None:
            print("Assuming zero distortion (None found in calibration results).")
        self._run_task_with_loading(
            task_func=backend.stereo_rect,
            kwargs={'stereo_path': folder_path, 'cameraMatrix': camera_matrix, 'distCoeffs': dist_coeffs},
            loading_msg="Running Rectification...",
            result_handler=self._handle_stereo_rect_result
        )

    def _handle_stereo_rect_result(self, results):
        if isinstance(results, dict) and "Error" in results:
            error_msg = results['Error']
            print(f"Error: {error_msg}")
            messagebox.showerror("Rectification Failed", error_msg)
            self.stereo_rect_results = results
        elif isinstance(results, dict):
            self.stereo_rect_results = results
            print("Stereo Rectification Success. Displaying images...")
            display_keys = ["Original Left", "Original Right", "Drawn Rectified Left", "Drawn Rectified Right"]
            fixed_size = (640, 360)
            for title in display_keys:
                img_label = self.stereo_img_labels.get(title)
                img_to_display = self.stereo_rect_results.get(title)
                if img_label and img_to_display is not None:
                    try:
                        self._display_image(img_label, cv2.resize(img_to_display, fixed_size, interpolation=cv2.INTER_AREA))
                    except Exception as e:
                        print(f"Error resizing/displaying image '{title}': {e}")
                        self._display_image(img_label, None)
                elif img_label:
                    print(f"Warning: Image '{title}' is None in results.")
                    self._display_image(img_label, None)
            messagebox.showinfo("Success", "Rectification complete.")
        else:
            err_msg = f"Unexpected rectification result type: {type(results)}. Content: {results}"
            print(f"Error: {err_msg}")
            messagebox.showerror("Error", err_msg)
            self.stereo_rect_results = {"Error": err_msg}

    def run_feat_detect_match_threaded(self):
        folder_path = self.feat_detect_match_path_entry.get()
        if self.feat_detect_match_output_text:
            self.feat_detect_match_output_text.delete(1.0, tk.END)
        else:
            print("Warning: feat_detect_match_output_text not initialized.")
            return
        for label in self.feat_img_labels.values():
            self._display_image(label, None)
        if not folder_path or not os.path.isdir(folder_path):
            messagebox.showerror("Error", "Please select a valid folder path.")
            return
        camera_matrix = self.cam_calib_results.get("Camera Matrix") if self.cam_calib_results and "Error" not in self.cam_calib_results else None
        if not hasattr(self, 'redirector_feat') or not self.redirector_feat:
            print("Error: redirector_feat not initialized.")
            return
        self.redirector_feat.start_redirect()
        print(f"Starting Feature Detection & Matching: {folder_path}")
        if camera_matrix is None:
            print("Using default intrinsics (Calibration not run or failed).")
        else:
            print("Using intrinsics from Calibration tab.")
        self._run_task_with_loading(
            task_func=backend.feat_detect_match,
            kwargs={'stereo_path': folder_path, 'cameraMatrix': camera_matrix},
            loading_msg="Running Detection & Matching...",
            result_handler=self._handle_feat_detect_match_result
        )

    def _handle_feat_detect_match_result(self, results):
        redirector = getattr(self, 'redirector_feat', None)
        if isinstance(results, dict) and "Error" in results:
            error_msg = results['Error']
            print(f"Error: {error_msg}")
            if redirector:
                redirector.stop_redirect()
            messagebox.showerror("Detection/Matching Failed", error_msg)
            self.feat_detect_match_results = results
        elif isinstance(results, dict):
            self.feat_detect_match_results = results
            if self.feat_detect_match_output_text:
                self.feat_detect_match_output_text.delete(1.0, tk.END)
            kp_l = results.get("Num Keypoints Left", "N/A")
            kp_r = results.get("Num Keypoints Right", "N/A")
            match_raw = results.get("Num Raw Matches", "N/A")
            match_good = results.get("Num Good Matches", "N/A")
            summary = (f"--- Detection & Matching Summary ---\n"
                       f"Left Image Keypoints: {kp_l}\nRight Image Keypoints: {kp_r}\n"
                       f"Raw Matches (knn=2): {match_raw}\nGood Matches (Ratio Test): {match_good}\n"
                       f"------------------------------------\nSUCCESS: Detection & matching complete.\n")
            if self.feat_detect_match_output_text:
                self.feat_detect_match_output_text.insert(tk.END, summary)
                self.feat_detect_match_output_text.see(tk.END)
            if redirector:
                redirector.stop_redirect()
            for title, img_label in self.feat_img_labels.items():
                img = self.feat_detect_match_results.get(title)
                if img is not None:
                    fixed_size = (1280, 360) if "Matched Images" in title else (640, 360)
                    try:
                        self._display_image(img_label, cv2.resize(img, fixed_size, interpolation=cv2.INTER_AREA))
                    except Exception as e:
                        print(f"Error resizing/displaying image '{title}': {e}")
                        self._display_image(img_label, None)
                else:
                    print(f"Warning: Image '{title}' is None.")
                    self._display_image(img_label, None)
            messagebox.showinfo("Success", "Detection & matching complete.")
        else:
            err_msg = f"Unexpected matching result type: {type(results)}. Content: {results}"
            print(f"Error: {err_msg}")
            if redirector:
                redirector.stop_redirect()
            messagebox.showerror("Error", err_msg)
            self.feat_detect_match_results = {"Error": err_msg}

    def run_stereo_geometry_estimation_threaded(self):
        folder_path = self.stereo_geometry_estimation_path_entry.get()
        if self.stereo_geometry_output_text:
            self.stereo_geometry_output_text.delete(1.0, tk.END)
        else:
            print("Warning: stereo_geometry_output_text not initialized.")
            return
        if not folder_path or not os.path.isdir(folder_path):
            messagebox.showerror("Error", "Please select a valid folder path.")
            return
        camera_matrix = self.cam_calib_results.get("Camera Matrix") if self.cam_calib_results and "Error" not in self.cam_calib_results else None
        if not hasattr(self, 'redirector_geom') or not self.redirector_geom:
            print("Error: redirector_geom not initialized.")
            return
        self.redirector_geom.start_redirect()
        print(f"Starting Standalone Stereo Geometry Estimation: {folder_path}")
        if camera_matrix is None:
            print("Using default intrinsics (Calibration not run or failed).")
        else:
            print("Using intrinsics from Calibration tab.")
        self._run_task_with_loading(
            task_func=backend.stereo_geometry_estimation,
            kwargs={'stereo_path': folder_path, 'cameraMatrix': camera_matrix},
            loading_msg="Estimating Geometry...",
            result_handler=self._handle_stereo_geometry_result
        )

    def _handle_stereo_geometry_result(self, results):
        redirector = getattr(self, 'redirector_geom', None)
        if isinstance(results, dict) and "Error" in results:
            error_msg = results['Error']
            print(f"Error: {error_msg}")
            if redirector:
                redirector.stop_redirect()
            messagebox.showerror("Geometry Estimation Failed", error_msg)
            self.stereo_geometry_results = results
        elif isinstance(results, dict):
            self.stereo_geometry_results = results
            if self.stereo_geometry_output_text:
                self.stereo_geometry_output_text.delete(1.0, tk.END)
            output_str = "--- Stereo Geometry Results ---\n\n"
            with np.printoptions(precision=4, suppress=True):
                f_mat = results.get("Fundamental Matrix")
                output_str += "Fundamental Matrix (F):\n" + (np.array2string(f_mat) if f_mat is not None else "Not computed or failed.") + "\n\n"
                e_mat = results.get("Essential Matrix")
                det_e = results.get("Determinant E")
                svd_count_e = results.get("Non Zero SVD E")
                svd_vals_e = results.get("Singular Values E")
                output_str += "Essential Matrix (E):\n"
                if e_mat is not None:
                    output_str += np.array2string(e_mat) + "\n"
                    if det_e is not None:
                        output_str += f"  Determinant: {det_e:.4e}\n"
                    if svd_vals_e is not None:
                        output_str += f"  Singular Values: {np.array2string(svd_vals_e)}\n"
                    if svd_count_e is not None:
                        output_str += f"  Non-Zero Singular Values (>1e-6): {svd_count_e} / 3\n"
                else:
                    output_str += "Not computed or failed."
                output_str += "\n"
                r_mat = results.get("Rotation Matrix")
                output_str += "Rotation Matrix (R):\n" + (np.array2string(r_mat) if r_mat is not None else "Not computed or failed.") + "\n\n"
                t_vec = results.get("Translation Vector")
                output_str += "Translation Vector (T):\n" + (np.array2string(t_vec) if t_vec is not None else "Not computed or failed.") + "\n"
            output_str += "\n-------------------------------\nSUCCESS: Stereo geometry estimation complete.\n"
            if self.stereo_geometry_output_text:
                self.stereo_geometry_output_text.insert(tk.END, output_str)
                self.stereo_geometry_output_text.see(tk.END)
            if redirector:
                redirector.stop_redirect()
            messagebox.showinfo("Success", "Stereo geometry estimation complete.")
        else:
            err_msg = f"Unexpected geometry result type: {type(results)}. Content: {results}"
            print(f"Error: {err_msg}")
            if redirector:
                redirector.stop_redirect()
            messagebox.showerror("Error", err_msg)
            self.stereo_geometry_results = {"Error": err_msg}

    def run_triangulation_and_reconstruction_threaded(self):
        if self.triangulation_output_text:
            self.triangulation_output_text.delete(1.0, tk.END)
        else:
            print("Warning: triangulation_output_text not initialized.")
            return
        if not self.cam_calib_results or "Error" in self.cam_calib_results:
            messagebox.showerror("Missing Data", "Run Camera Calibration (Tab 1) successfully first.")
            return
        if not self.feat_detect_match_results or "Error" in self.feat_detect_match_results:
            messagebox.showerror("Missing Data", "Run Feature Detection & Matching (Tab 3) successfully first.")
            return
        pts1 = self.feat_detect_match_results.get("Left Aligned Keypoints")
        pts2 = self.feat_detect_match_results.get("Right Aligned Keypoints")
        camera_matrix = self.cam_calib_results.get("Camera Matrix")
        rotation_matrix = self.feat_detect_match_results.get("Rotation Matrix")
        translation_vector = self.feat_detect_match_results.get("Translation Vector")
        imgL_color = self.feat_detect_match_results.get("Left Color Image")
        missing = [name for name, var in [
            ("Left Aligned Keypoints", pts1), ("Right Aligned Keypoints", pts2),
            ("Camera Matrix", camera_matrix), ("Rotation Matrix", rotation_matrix),
            ("Translation Vector", translation_vector), ("Left Color Image", imgL_color)
        ] if var is None]
        if missing:
            messagebox.showerror("Missing Data", f"Could not retrieve the following required data: {', '.join(missing)}")
            return
        if not hasattr(self, 'redirector_tri') or not self.redirector_tri:
            print("Error: redirector_tri not initialized.")
            return
        self.redirector_tri.start_redirect()
        print("Starting Triangulation (Feature-Based)...")
        print(f"Using {len(pts1)} points for triangulation.")
        self._run_task_with_loading(
            task_func=backend.triangulation_and_3D_reconstruction,
            args=(pts1, pts2, camera_matrix, rotation_matrix, translation_vector, imgL_color),
            loading_msg="Triangulating & Visualizing...",
            result_handler=self._handle_triangulation_result
        )

    def _handle_triangulation_result(self, results):
        redirector = getattr(self, 'redirector_tri', None)
        if isinstance(results, dict) and "Error" in results:
            error_msg = results['Error']
            print(f"Error: {error_msg}")
            if redirector:
                redirector.stop_redirect()
            messagebox.showerror("Triangulation Failed", error_msg)
            self.triangulation_results = results
        elif isinstance(results, dict) and "3D Points" in results:
            self.triangulation_results = results
            num_points = len(results["3D Points"])
            if self.triangulation_output_text:
                self.triangulation_output_text.delete(1.0, tk.END)
            log_msg = (f"SUCCESS: Generated {num_points} 3D points.\n"
                       f"Attempted to display colored point cloud in Open3D window.\n")
            if self.triangulation_output_text:
                self.triangulation_output_text.insert(tk.END, log_msg)
                self.triangulation_output_text.see(tk.END)
            if redirector:
                redirector.stop_redirect()
            messagebox.showinfo("Success", f"Triangulation successful ({num_points} points).\nCheck Open3D window (if backend opened one).")
        else:
            err_msg = f"Unexpected result from triangulation: {type(results)}. Content: {results}"
            print(err_msg)
            if redirector:
                redirector.stop_redirect()
            messagebox.showerror("Error", err_msg)
            self.triangulation_results = {"Error": err_msg}

    def run_disparity_calculation_threaded(self):
        if self.disparity_output_text:
            self.disparity_output_text.delete(1.0, tk.END)
        else:
            print("Warning: disparity_output_text not initialized.")
            return
        self._display_image(self.disparity_img_label, None)
        if not self.stereo_rect_results or "Error" in self.stereo_rect_results:
            messagebox.showerror("Missing Data", "Run 'Stereo Rectification' (Tab 2) successfully first.")
            return
        imgL_rect_gray = self.stereo_rect_results.get("Rectified Left")
        imgR_rect_gray = self.stereo_rect_results.get("Rectified Right")
        imgL_rect_color = self.stereo_rect_results.get("Rectified Color Left")
        imgL_for_disp = imgL_rect_gray if imgL_rect_gray is not None else imgL_rect_color
        imgR_for_disp = imgR_rect_gray if imgR_rect_gray is not None else self.stereo_rect_results.get("Rectified Color Right")
        if imgL_for_disp is None or imgR_for_disp is None:
            messagebox.showerror("Missing Data", "Rectified images not found in Tab 2 results.")
            return
        guide_image = imgL_rect_color if imgL_rect_color is not None else imgL_for_disp
        print("Starting Disparity Calculation (SGBM + WLS)... (Output to console)")
        self._run_task_with_loading(
            task_func=backend.disparity_calculation,
            args=(imgL_for_disp, imgR_for_disp),
            kwargs={'guide_image': guide_image},
            loading_msg="Calculating Disparity...",
            result_handler=self._handle_disparity_result
        )

    def _handle_disparity_result(self, results):
        if isinstance(results, dict) and "Error" in results:
            error_msg = results['Error']
            print(f"Error: {error_msg}")
            if self.disparity_output_text:
                self.disparity_output_text.insert(tk.END, f"ERROR: {error_msg}\n")
                self.disparity_output_text.see(tk.END)
            messagebox.showerror("Disparity Failed", error_msg)
            self.disparity_calculation_results = results
        elif isinstance(results, dict) and "Disparity Map Color" in results:
            self.disparity_calculation_results = results
            disparity_map_vis = results.get("Disparity Map Color")
            print("Success. Displaying filtered disparity heatmap.")
            if self.disparity_output_text:
                self.disparity_output_text.insert(tk.END, "SUCCESS: Disparity map calculated and filtered.\n")
                self.disparity_output_text.see(tk.END)
            if disparity_map_vis is not None:
                fixed_size = (640, 480)
                try:
                    self._display_image(self.disparity_img_label, cv2.resize(disparity_map_vis, fixed_size, interpolation=cv2.INTER_AREA))
                except Exception as e:
                    print(f"Error resizing/displaying disparity map: {e}")
                    self._display_image(self.disparity_img_label, None)
            else:
                print("Warning: Disparity Map Color image not found in results.")
                self._display_image(self.disparity_img_label, None)
            messagebox.showinfo("Success", "Disparity map calculated and filtered.")
        else:
            err_msg = f"Unknown result from disparity calculation: {type(results)}. Content: {results}"
            print(err_msg)
            if self.disparity_output_text:
                self.disparity_output_text.insert(tk.END, f"ERROR: {err_msg}\n")
                self.disparity_output_text.see(tk.END)
            messagebox.showerror("Error", err_msg)
            self.disparity_calculation_results = {"Error": err_msg}

    def run_visualize_3d_threaded(self):
        if self.disparity_output_text:
            self.disparity_output_text.delete(1.0, tk.END)
        else:
            print("Warning: disparity_output_text not initialized.")
            return
        if not self.disparity_calculation_results or "Error" in self.disparity_calculation_results:
            messagebox.showerror("Missing Data", "Run 'Run Disparity' successfully first on this tab.")
            return
        if not self.stereo_rect_results or "Error" in self.stereo_rect_results:
            messagebox.showerror("Missing Data", "Run 'Stereo Rectification' (Tab 2) successfully first.")
            return
        raw_disparity_map = self.disparity_calculation_results.get("Raw Disparity")
        Q = self.stereo_rect_results.get("disp2depth map")
        colors = self.stereo_rect_results.get("Rectified Color Left")
        missing = [name for name, var in [
            ("Raw Disparity Map", raw_disparity_map),
            ("Q Matrix (disp2depth)", Q),
            ("Rectified Left Color Image", colors)
        ] if var is None]
        if missing:
            messagebox.showerror("Missing Data", f"Could not retrieve the following required data: {', '.join(missing)}")
            return
        print("Starting 3D Visualization from Dense Disparity... (Output to console)")
        self._run_task_with_loading(
            task_func=backend.visualize_point_cloud_disparity,
            args=(raw_disparity_map, Q, colors),
            loading_msg="Generating & Visualizing 3D Cloud...",
            result_handler=self._handle_visualize_3d_result
        )

    def _handle_visualize_3d_result(self, error_message):
        if error_message:
            print(f"Error: {error_message}")
            if self.disparity_output_text:
                self.disparity_output_text.insert(tk.END, f"ERROR: {error_message}\n")
                self.disparity_output_text.see(tk.END)
            messagebox.showerror("Visualization Failed", error_message)
        else:
            print("Point cloud display attempted.")
            if self.disparity_output_text:
                self.disparity_output_text.insert(tk.END, "SUCCESS: Point cloud display attempted.\n")
                self.disparity_output_text.see(tk.END)

    def run_sparse_xfeat_matching_threaded(self):
        if self.sparse_xfeat_output_text:
            self.sparse_xfeat_output_text.delete(1.0, tk.END)
        else:
            print("Warning: sparse_xfeat_output_text not initialized.")
            return
        self._display_image(self.sparse_xfeat_match_label, None)
        self.sparse_xfeat_results = {}
        folder_path = self.sparse_xfeat_path_entry.get()
        if not folder_path or not os.path.isdir(folder_path):
            messagebox.showerror("Error", "Please select a valid stereo pair folder path.")
            return
        redirector = getattr(self, 'redirector_xfeat', None)
        if not redirector:
            print("Error: redirector_xfeat not initialized.")
            return
        redirector.start_redirect()
        print(f"Starting XFeat Matching: {folder_path}")
        self._run_task_with_loading(
            task_func=backend.xfeat_matching,
            args=(folder_path,),
            loading_msg="Running XFeat Matching...",
            result_handler=self._handle_sparse_xfeat_matching_result
        )

    def _handle_sparse_xfeat_matching_result(self, results):
        redirector = getattr(self, 'redirector_xfeat', None)
        if isinstance(results, dict) and "Error" in results:
            error_msg = results['Error']
            print(f"Error: {error_msg}")
            if redirector:
                redirector.stop_redirect()
            messagebox.showerror("XFeat Matching Failed", error_msg)
            self.sparse_xfeat_results = results
        elif isinstance(results, dict) and "Matched Image" in results:
            self.sparse_xfeat_results = {
                "mkpts_0": results.get("mkpts_0"),
                "mkpts_1": results.get("mkpts_1"),
                "Left Color Image": results.get("Left Color Image"),
                "Num Matches": results.get("Num Matches"),
            }
            if self.sparse_xfeat_output_text:
                self.sparse_xfeat_output_text.delete(1.0, tk.END)
            num_matches = self.sparse_xfeat_results.get("Num Matches", "N/A")
            summary = (f"--- XFeat Matching Summary ---\n"
                       f"Raw XFeat Matches Found: {num_matches}\n"
                       f"------------------------------------\n"
                       f"SUCCESS: XFeat matching complete.\n"
                       f"Ready for 3D Reconstruction (Button 2).\n")
            if self.sparse_xfeat_output_text:
                self.sparse_xfeat_output_text.insert(tk.END, summary)
                self.sparse_xfeat_output_text.see(tk.END)
            if redirector:
                redirector.stop_redirect()
            match_img = results.get("Matched Image")
            if match_img is not None:
                fixed_size = (1280, 480)
                try:
                    self._display_image(self.sparse_xfeat_match_label, cv2.resize(match_img, fixed_size, interpolation=cv2.INTER_AREA))
                except Exception as e:
                    print(f"Error resizing/displaying XFeat matched image: {e}")
                    self._display_image(self.sparse_xfeat_match_label, None)
            else:
                print("Warning: XFeat Matched Image not found in results.")
                self._display_image(self.sparse_xfeat_match_label, None)
            messagebox.showinfo("Success", f"XFeat matching complete ({num_matches} matches).\nReady for 3D Reconstruction.")
        else:
            err_msg = f"Unexpected XFeat matching result type: {type(results)}. Content: {results}"
            print(err_msg)
            if redirector:
                redirector.stop_redirect()
            messagebox.showerror("Error", err_msg)
            self.sparse_xfeat_results = {"Error": err_msg}

    def run_sparse_xfeat_reconstruction_threaded(self):
        redirector = getattr(self, 'redirector_xfeat', None)
        if not redirector:
            print("Error: redirector_xfeat not initialized.")
            return
        if not self.sparse_xfeat_results or "Error" in self.sparse_xfeat_results or \
           self.sparse_xfeat_results.get("mkpts_0") is None or \
           self.sparse_xfeat_results.get("mkpts_1") is None or \
           self.sparse_xfeat_results.get("Left Color Image") is None:
            messagebox.showerror("Missing Data", "Run '1. Run XFeat Matching' successfully first on this tab.")
            return
        if not self.cam_calib_results or "Error" in self.cam_calib_results or \
           self.cam_calib_results.get("Camera Matrix") is None:
            messagebox.showerror("Missing Data", "Run Camera Calibration (Tab 1) successfully first (provides K).")
            return
        if not self.feat_detect_match_results or "Error" in self.feat_detect_match_results or \
           self.feat_detect_match_results.get("Rotation Matrix") is None or \
           self.feat_detect_match_results.get("Translation Vector") is None:
            messagebox.showerror("Missing Data", "Run Feature Detection & Matching (Tab 3) successfully first (provides R, T).")
            return
        pts1 = self.sparse_xfeat_results.get("mkpts_0")
        pts2 = self.sparse_xfeat_results.get("mkpts_1")
        imgL_color = self.sparse_xfeat_results.get("Left Color Image")
        camera_matrix = self.cam_calib_results.get("Camera Matrix")
        rotation_matrix = self.feat_detect_match_results.get("Rotation Matrix")
        translation_vector = self.feat_detect_match_results.get("Translation Vector")
        if pts1 is None or pts2 is None or imgL_color is None or \
           camera_matrix is None or rotation_matrix is None or translation_vector is None:
            messagebox.showerror("Internal Error", "Failed to retrieve required data for reconstruction.")
            return
        redirector.start_redirect()
        print("\n--- Starting XFeat 3D Reconstruction ---")
        print(f"Using {len(pts1)} points from XFeat matching.")
        print("Using K from Tab 1 and R, T from Tab 3.")
        self._run_task_with_loading(
            task_func=backend.xfeat_reconstruction,
            args=(pts1, pts2, camera_matrix, rotation_matrix, translation_vector, imgL_color),
            loading_msg="Running XFeat 3D Reconstruction...",
            result_handler=self._handle_sparse_xfeat_reconstruction_result
        )

    def _handle_sparse_xfeat_reconstruction_result(self, results):
        redirector = getattr(self, 'redirector_xfeat', None)
        if isinstance(results, dict) and "Error" in results:
            error_msg = results['Error']
            print(f"Error: {error_msg}")
            if redirector:
                redirector.stop_redirect()
            messagebox.showerror("XFeat Reconstruction Failed", error_msg)
            if "Reconstruction Error" not in self.sparse_xfeat_results:
                self.sparse_xfeat_results["Reconstruction Error"] = error_msg
        elif isinstance(results, dict) and "Num 3D Points" in results:
            num_3d = results.get("Num 3D Points", "N/A")
            vis_error = results.get("Visualization Error")
            summary = (f"\n--- XFeat Reconstruction Summary ---\n"
                       f"Filtered 3D Points Generated: {num_3d}\n")
            if vis_error:
                summary += f"WARNING: Point cloud visualization failed:\n{vis_error}\n"
            else:
                summary += f"Point cloud display attempted (check Open3D window).\n"
            summary += "------------------------------------\n"
            summary += "SUCCESS: XFeat 3D reconstruction complete.\n"
            if self.sparse_xfeat_output_text:
                self.sparse_xfeat_output_text.insert(tk.END, summary)
                self.sparse_xfeat_output_text.see(tk.END)
            if redirector:
                redirector.stop_redirect()
            self.sparse_xfeat_results["3D Points"] = results.get("3D Points")
            self.sparse_xfeat_results["Num 3D Points"] = num_3d
            if vis_error:
                self.sparse_xfeat_results["Visualization Error"] = vis_error
            msg = f"XFeat reconstruction complete.\nGenerated {num_3d} 3D points."
            if vis_error:
                msg += f"\n\nNote: {vis_error}"
            else:
                msg += "\nCheck Open3D window (if backend opened one)."
            messagebox.showinfo("Success", msg)
        else:
            err_msg = f"Unexpected XFeat reconstruction result type: {type(results)}. Content: {results}"
            print(err_msg)
            if redirector:
                redirector.stop_redirect()
            messagebox.showerror("Error", err_msg)
            if "Reconstruction Error" not in self.sparse_xfeat_results:
                self.sparse_xfeat_results["Reconstruction Error"] = err_msg

    # --- Browse Methods ---
    def _set_path_entry(self, entry_widget, folder_path):
        if entry_widget:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, folder_path)
        else:
            print("Warning: Tried to set path for a non-existent widget.")

    def _auto_populate_paths(self, selected_folder):
        path_entries = [
            self.stereo_rect_path_entry,
            self.feat_detect_match_path_entry,
            self.stereo_geometry_estimation_path_entry,
            self.sparse_xfeat_path_entry
        ]
        for entry in path_entries:
            if entry and not entry.get():
                self._set_path_entry(entry, selected_folder)

    def cam_calib_browse_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Chessboard Images (*.jpg)")
        if folder:
            self._set_path_entry(self.cam_calib_path_entry, folder)

    def stereo_rect_browse_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with img1.jpg and img2.jpg")
        if folder:
            self._set_path_entry(self.stereo_rect_path_entry, folder)
            self._auto_populate_paths(folder)

    def feat_detect_match_browse_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with img1.jpg and img2.jpg")
        if folder:
            self._set_path_entry(self.feat_detect_match_path_entry, folder)
            self._auto_populate_paths(folder)

    def stereo_geometry_estimation_browse_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with img1.jpg and img2.jpg")
        if folder:
            self._set_path_entry(self.stereo_geometry_estimation_path_entry, folder)
            self._auto_populate_paths(folder)

    def sparse_xfeat_browse_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with img1.jpg and img2.jpg")
        if folder:
            self._set_path_entry(self.sparse_xfeat_path_entry, folder)
            self._auto_populate_paths(folder)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = NotebookGUI(root)
    root.mainloop()