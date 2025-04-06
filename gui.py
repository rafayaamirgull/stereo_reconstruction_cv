# gui.py
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import os
import numpy as np
import cv2
from PIL import Image, ImageTk
from io import StringIO
import sys
import backend
import threading
import queue
import numpy.core.arrayprint as arrayprint

class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.output = StringIO()
        self.stdout_orig = sys.stdout
        self.stderr_orig = sys.stderr

    def write(self, string):
        self.output.write(string)
        try:
            if self.text_widget.winfo_exists():
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
        except tk.TclError:
            pass

    def flush(self): pass

    def start_redirect(self):
        sys.stdout = self

    def stop_redirect(self):
        # Ensure sys.stdout is restored only if it's the current redirector instance
        if isinstance(sys.stdout, RedirectText) and sys.stdout is self:
            if self.stdout_orig:
                sys.stdout = self.stdout_orig

    def get_output(self):
        return self.output.getvalue()

# Main GUI class


class NotebookGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stereo Vision Toolkit GUI")
        self.root.geometry("1450x950")

        # --- Scrolling Setup ---
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(
            self.main_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        # Bind mouse wheel events more specifically to the canvas and its children
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        # --- Notebook ---
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.pack(pady=10, padx=10, expand=True, fill="both")

        # --- State & Image Refs ---
        self.stereo_calib_results = None  # Stores results from stereo calibration
        self.stereo_rect_results = None
        self.feat_detect_match_results = None
        self.stereo_geometry_results = None
        self.triangulation_results = None
        self.disparity_calculation_results = None
        self.image_references = {}  # Stores {widget: ImageTk.PhotoImage}

        # --- Loading Indicator State ---
        self.loading_window = None
        self.task_queue = queue.Queue()

        # --- Create Tabs ---
        self.create_stereo_calib_tab()
        self.create_stereo_rect_tab()
        self.create_feat_detect_match_tab()
        self.create_stereo_geometry_tab()
        self.create_triangulation_and_reconstruction_tab()
        self.create_disparity_calculation_tab()

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
        self.canvas.bind_all(
            "<Button-4>", self._on_mousewheel)  # Linux scroll up
        # Linux scroll down
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        # Determine the scroll direction and amount
        if event.num == 4 or event.delta > 0:  # Linux scroll up or Windows/macOS scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:  # Linux scroll down or Windows/macOS scroll down
            self.canvas.yview_scroll(1, "units")

    # --- Loading Indicator Methods ---
    def _show_loading(self, message="Processing..."):
        if self.loading_window is not None and self.loading_window.winfo_exists():
            self.loading_window.destroy()
        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.transient(self.root)
        self.loading_window.title("Working")
        self.loading_window.resizable(False, False)
        self.loading_window.grab_set()
        self.loading_window.protocol(
            "WM_DELETE_WINDOW", lambda: None)  # Prevent closing
        ttk.Label(self.loading_window, text=message, font=(
            "Helvetica", 12)).pack(pady=10, padx=20)
        pb = ttk.Progressbar(
            self.loading_window, orient='horizontal', mode='indeterminate', length=200)
        pb.pack(pady=(0, 15), padx=20, fill='x', expand=True)
        pb.start(15)
        # Center the loading window
        self.loading_window.update_idletasks()  # Ensure window size is calculated
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

    # --- Threading and Queue Handling ---
    def _run_task_with_loading(self, task_func, args=(), kwargs={}, loading_msg="Processing...", result_handler=None):
        # Clear queue before starting a new task to prevent handling old results
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
        self._show_loading(loading_msg)

        def worker():
            try:
                result = task_func(*args, **kwargs)
                self.task_queue.put(("SUCCESS", result))
            except Exception as e:
                import traceback
                err_msg = f"Error in task: {e}\n{traceback.format_exc()}"
                print(f"Error in worker thread: {e}")  # Print error to console
                self.task_queue.put(("ERROR", err_msg))
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        # Start checking the queue
        self.root.after(100, self._check_queue, result_handler)

    def _check_queue(self, result_handler):
        try:
            status, result = self.task_queue.get_nowait()
            # If we got a result, hide loading and process it
            self._hide_loading()
            if status == "SUCCESS":
                if result_handler:
                    try:
                        result_handler(result)
                    except Exception as e:
                        # Print handler error to console
                        print(f"Error in result handler: {e}")
                        import traceback
                        traceback.print_exc()
                        messagebox.showerror(
                            "GUI Error", f"Error processing result:\n{e}")
                # No automatic success message box if a handler was provided
            elif status == "ERROR":
                print(f"Task failed:\n{result}")  # Print task error to console
                short_error = str(result).split('\n')[0]
                messagebox.showerror(
                    "Task Error", f"Operation failed:\n{short_error}\n\n(See console log for details)")
        
        except queue.Empty:
            # If the queue is empty, check if the loading window still exists
            # If it does, schedule another check
            if self.loading_window and self.loading_window.winfo_exists():
                self.root.after(100, self._check_queue, result_handler)
            # If loading window is gone (or never existed), stop checking
        
        except Exception as e:
            # Print GUI error to console
            print(f"Error checking queue or handling result: {e}")
            import traceback
            traceback.print_exc()
            self._hide_loading()  # Ensure loading is hidden on unexpected error
            messagebox.showerror(
                "GUI Error", f"Error checking task status:\n{e}")

    # --- Image/GUI Update Methods ---
    def _clear_image_references(self):
        for widget in list(self.image_references.keys()):
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
        self._hide_loading()
        self._clear_image_references()
        
        # Stop all redirectors safely
        for attr_name in dir(self):
            if attr_name.startswith('redirector_'):
                redirector = getattr(self, attr_name, None)
                if isinstance(redirector, RedirectText):
                    try:
                        redirector.stop_redirect()
                    except Exception as e:
                        print(f"Error stopping redirector {attr_name}: {e}")
        
        # Explicitly unbind mouse wheel to prevent errors after destroy
        self._unbind_mousewheel(None)
        self.root.destroy()

    def _display_image(self, label_widget, cv_image_pre_resized):
        """Displays a pre-resized OpenCV image in a Tkinter Label."""
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
            if len(img_display.shape) == 2:  # Grayscale
                img_pil = Image.fromarray(img_display)
            # Color BGR
            elif len(img_display.shape) == 3 and img_display.shape[2] == 3:
                img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
            else:
                print(
                    f"Error: Unsupported image shape for display: {img_display.shape}")
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
    def create_stereo_calib_tab(self):
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="1. Stereo Calibration")
    
        controls_frame = ttk.Frame(tab1)
        controls_frame.pack(pady=5, padx=5, fill='x')
        ttk.Label(controls_frame, text="Folder with Stereo Pairs (cam1_TS.ext, cam2_TS.ext):").grid(
            row=0, column=0, padx=5, pady=5, sticky="w")
    
        self.stereo_calib_path_entry = ttk.Entry(controls_frame, width=60)
        self.stereo_calib_path_entry.grid(
            row=0, column=1, padx=5, pady=5, sticky="ew")
        browse_btn = ttk.Button(
            controls_frame, text="Browse", command=self.stereo_calib_browse_folder)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
    
        run_btn = ttk.Button(controls_frame, text="Run Stereo Calibration",
                             command=self.run_stereo_calib_threaded)
        run_btn.grid(row=1, column=1, pady=10)
        controls_frame.columnconfigure(1, weight=1)
        ttk.Label(tab1, text="Output Log & Results:").pack(
            pady=(10, 0), padx=5, anchor='w')
    
        self.stereo_calib_output_text = scrolledtext.ScrolledText(
            tab1, width=80, height=20, wrap=tk.WORD)
        self.stereo_calib_output_text.pack(
            pady=5, padx=5, expand=True, fill='both')
        self.redirector_calib = RedirectText(self.stereo_calib_output_text)

    def create_stereo_rect_tab(self):
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="2. Stereo Rectification")
    
        controls_frame = ttk.Frame(tab2)
        controls_frame.pack(pady=5, padx=5, fill='x')
        controls_frame.columnconfigure(1, weight=1)
        ttk.Label(controls_frame, text="Folder with ONE Stereo Pair (cam1_TS.ext, cam2_TS.ext):").grid(
            row=0, column=0, padx=5, pady=5, sticky="w")
    
        self.stereo_rect_path_entry = ttk.Entry(controls_frame, width=60)
        self.stereo_rect_path_entry.grid(
            row=0, column=1, padx=5, pady=5, sticky="ew")
    
        browse_btn = ttk.Button(
            controls_frame, text="Browse", command=self.stereo_rect_browse_folder)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
    
        ttk.Label(controls_frame, text="Requires results from Tab 1 (Stereo Calibration).").grid(
            row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        run_btn = ttk.Button(
            controls_frame, text="Run Rectification", command=self.run_stereo_rect_threaded)
        run_btn.grid(row=2, column=1, pady=10)
    
        img_frame = ttk.Frame(tab2)
        img_frame.pack(pady=10, padx=5, expand=True, fill='both')
        self.stereo_img_labels = {}
    
        titles = ["Original Left", "Original Right",
                  "Drawn Rectified Left", "Drawn Rectified Right"]
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
    
        ttk.Label(controls_frame, text="Folder with ONE Stereo Pair (cam1_TS.ext, cam2_TS.ext):").grid(
            row=0, column=0, padx=5, pady=5, sticky="w")
    
        self.feat_detect_match_path_entry = ttk.Entry(controls_frame, width=60)
        self.feat_detect_match_path_entry.grid(
            row=0, column=1, padx=5, pady=5, sticky="ew")
    
        browse_btn = ttk.Button(
            controls_frame, text="Browse", command=self.feat_detect_match_browse_folder)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
    
        ttk.Label(controls_frame, text="Uses Left Camera Matrix (M1) from Tab 1 if available.").grid(
            row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")
    
        run_btn = ttk.Button(controls_frame, text="Run Detection & Matching",
                             command=self.run_feat_detect_match_threaded)
        run_btn.grid(row=2, column=1, pady=10)
    
        img_frame = ttk.Frame(tab3)
        img_frame.pack(pady=10, padx=5, expand=True, fill='both')
        img_frame.columnconfigure(0, weight=1)
        img_frame.columnconfigure(1, weight=1)
        img_frame.rowconfigure(0, weight=1)
        img_frame.rowconfigure(1, weight=1)
        img_frame.rowconfigure(2, weight=1)
    
        self.feat_img_labels = {}
    
        titles = ["Left Image with Keypoints", "Right Image with Keypoints",
                  "Matched Images Before Lowe's Ratio", "Matched Images After Lowe's Ratio"]
    
        frame_kp_l = ttk.LabelFrame(img_frame, text=titles[0])
        frame_kp_l.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    
        self.feat_img_labels[titles[0]] = tk.Label(frame_kp_l)
        self.feat_img_labels[titles[0]].pack(padx=5, pady=5)
    
        frame_kp_r = ttk.LabelFrame(img_frame, text=titles[1])
        frame_kp_r.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
    
        self.feat_img_labels[titles[1]] = tk.Label(frame_kp_r)
        self.feat_img_labels[titles[1]].pack(padx=5, pady=5)
    
        frame_match_raw = ttk.LabelFrame(img_frame, text=titles[2])
        frame_match_raw.grid(row=1, column=0, columnspan=2,
                             padx=5, pady=5, sticky="nsew")
    
        self.feat_img_labels[titles[2]] = tk.Label(frame_match_raw)
        self.feat_img_labels[titles[2]].pack(padx=5, pady=5)
    
        frame_match_good = ttk.LabelFrame(img_frame, text=titles[3])
        frame_match_good.grid(row=2, column=0, columnspan=2,
                              padx=5, pady=5, sticky="nsew")
    
        self.feat_img_labels[titles[3]] = tk.Label(frame_match_good)
        self.feat_img_labels[titles[3]].pack(padx=5, pady=5)
    
        log_frame = ttk.Frame(tab3)
        log_frame.pack(pady=(0, 5), padx=5, fill='x')
    
        ttk.Label(log_frame, text="Output Log (Counts & Status):").pack(
            pady=(5, 0), anchor='w')
    
        self.feat_detect_match_output_text = scrolledtext.ScrolledText(
            log_frame, width=80, height=5, wrap=tk.WORD)
        self.feat_detect_match_output_text.pack(
            pady=5, padx=0, fill='x', expand=False)
        self.redirector_feat = RedirectText(self.feat_detect_match_output_text)

    def create_stereo_geometry_tab(self):
        tab4 = ttk.Frame(self.notebook)
        self.notebook.add(tab4, text="4. Stereo Geometry (Feature-Based)")
        controls_frame = ttk.Frame(tab4)
        controls_frame.pack(pady=5, padx=5, fill='x')
    
        ttk.Label(controls_frame, text="Folder with ONE Stereo Pair (cam1_TS.ext, cam2_TS.ext):").grid(
            row=0, column=0, padx=5, pady=5, sticky="w")
    
        self.stereo_geometry_estimation_path_entry = ttk.Entry(
            controls_frame, width=60)
        self.stereo_geometry_estimation_path_entry.grid(
            row=0, column=1, padx=5, pady=5, sticky="ew")
    
        browse_btn = ttk.Button(controls_frame, text="Browse",
                                command=self.stereo_geometry_estimation_browse_folder)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
    
        ttk.Label(controls_frame, text="Uses Left Camera Matrix (M1) from Tab 1 if available.").grid(
            row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        run_btn = ttk.Button(controls_frame, text="Run Estimation",
                             command=self.run_stereo_geometry_estimation_threaded)
        run_btn.grid(row=2, column=1, pady=10)
        controls_frame.columnconfigure(1, weight=1)
    
        ttk.Label(tab4, text="Output Log & Results:").pack(
            pady=(10, 0), padx=5, anchor='w')
    
        self.stereo_geometry_output_text = scrolledtext.ScrolledText(
            tab4, width=80, height=20, wrap=tk.WORD)
        self.stereo_geometry_output_text.pack(
            pady=5, padx=5, expand=True, fill='both')
        self.redirector_geom = RedirectText(self.stereo_geometry_output_text)
    
    def create_triangulation_and_reconstruction_tab(self):
        tab5 = ttk.Frame(self.notebook)
        self.notebook.add(tab5, text="5. Triangulation (Feature-Based)")
        controls_frame = ttk.Frame(tab5)
        controls_frame.pack(pady=5, padx=5, fill='x', side='top')
    
        ttk.Label(
            controls_frame, text="Requires results from Tab 1 (Stereo Calib) and Tab 3 (Matching).").pack(pady=5)
     
        run_btn = ttk.Button(controls_frame, text="Run Triangulation & Visualize 3D",
                             command=self.run_triangulation_and_reconstruction_threaded)
        run_btn.pack(pady=10)
        output_frame = ttk.Frame(tab5)
        output_frame.pack(pady=5, padx=5, fill='x', side='top')
    
        ttk.Label(output_frame, text="Output Log & Status:").pack(
            pady=(5, 0), anchor='w')
    
        self.triangulation_output_text = scrolledtext.ScrolledText(
            output_frame, width=80, height=10, wrap=tk.WORD)
        self.triangulation_output_text.pack(
            pady=5, padx=0, expand=False, fill='x')
        self.redirector_tri = RedirectText(self.triangulation_output_text)
    
    def create_disparity_calculation_tab(self):
        tab6 = ttk.Frame(self.notebook)
        self.notebook.add(tab6, text="6. Disparity & 3D (Dense)")
    
        controls_frame = ttk.Frame(tab6)
        controls_frame.pack(pady=5, padx=5, fill='x')
        controls_frame.columnconfigure(1, weight=1)
    
        ttk.Label(controls_frame, text="Requires results from Tab 1 (Stereo Calib -> Tab 2) and Tab 2 (Rectification).").grid(
            row=0, column=0, columnspan=3, padx=5, pady=5, sticky="w")
    
        run_disparity_btn = ttk.Button(
            controls_frame, text="Run Disparity", command=self.run_disparity_calculation_threaded)
        run_disparity_btn.grid(row=1, column=0, pady=10, padx=5)
    
        run_3d_btn = ttk.Button(
            controls_frame, text="Visualize 3D Point Cloud", command=self.run_visualize_3d_threaded)
        run_3d_btn.grid(row=1, column=1, pady=10, padx=5)
    
        img_frame = ttk.LabelFrame(
            tab6, text="Filtered Disparity Heatmap (Normalized)")
        img_frame.pack(pady=10, padx=5, fill='both')
    
        self.disparity_img_label = tk.Label(img_frame)
        self.disparity_img_label.pack(padx=5, pady=5)
    
        log_frame = ttk.Frame(tab6)
        log_frame.pack(pady=(0, 5), padx=5, fill='x')
    
        ttk.Label(log_frame, text="Output Log (Status):").pack(
            pady=(5, 0), anchor='w')
    
        self.disparity_output_text = scrolledtext.ScrolledText(
            log_frame, width=80, height=5, wrap=tk.WORD)
    
        self.disparity_output_text.pack(pady=5, padx=0, fill='x', expand=False)
        self.redirector_disp = RedirectText(self.disparity_output_text)

    # --- Action Methods ---
    # 1. Stereo Calibration
    def run_stereo_calib_threaded(self):
        folder_path = self.stereo_calib_path_entry.get()
        self.stereo_calib_output_text.delete(1.0, tk.END)
        
        if not folder_path or not os.path.isdir(folder_path):
            messagebox.showerror(
                "Error", "Please select a valid folder containing stereo image pairs (cam1_TS.ext / cam2_TS.ext).")
            return
        
        self.redirector_calib.start_redirect()
        
        print(f"Starting Stereo Calibration: {folder_path}")
        self._run_task_with_loading(
            task_func=backend.stereo_calib,
            args=(folder_path,),
            loading_msg="Running Stereo Calibration...",
            result_handler=self._handle_stereo_calib_result
        )

    # 1. Handler
    def _handle_stereo_calib_result(self, results):
        if isinstance(results, str):
            print(f"Error: {results}")
            self.redirector_calib.stop_redirect()
            messagebox.showerror("Stereo Calibration Failed", results)
            self.stereo_calib_results = None
        
        elif isinstance(results, dict) and "Error" in results:
            print(f"Error: {results['Error']}")
            self.redirector_calib.stop_redirect()
            messagebox.showerror("Stereo Calibration Failed", results['Error'])
            self.stereo_calib_results = None
        
        elif isinstance(results, dict):
            self.stereo_calib_results = results
            print("\n--- Stereo Calibration Results ---")
            result_keys = ["M1", "d1", "M2", "d2", "R", "T",
                           "E", "F", "image_size", "Reprojection Error"]
        
            for key in result_keys:
                if key in self.stereo_calib_results:
                    value = self.stereo_calib_results[key]
                    print(f"{key}:")
                    if isinstance(value, np.ndarray):
                        print(np.array2string(
                            value, precision=5, suppress_small=True))
                    elif isinstance(value, (tuple, list)):
                        print(str(value))
                    elif isinstance(value, float):
                        print(f"{value:.5f}")
                    else:
                        print(value)
                    print("-" * 20)
        
            reproj_err = self.stereo_calib_results.get(
                "Reprojection Error", "N/A")
            print(f"\nStereo calibration process completed successfully.")
            if isinstance(reproj_err, float):
                print(f"Final Stereo Reprojection Error: {reproj_err:.5f}")
            self.redirector_calib.stop_redirect()
            msg = "Stereo Calibration complete."
            if isinstance(reproj_err, float):
                msg += f"\nReprojection Error: {reproj_err:.5f}"
            messagebox.showinfo("Success", msg)
        
        else:
            print(
                f"Error: Unexpected stereo calibration result type: {type(results)}")
            self.redirector_calib.stop_redirect()
            messagebox.showerror(
                "Error", "Stereo Calibration returned unexpected data.")
            self.stereo_calib_results = None

    # 2. Stereo Rectification
    def run_stereo_rect_threaded(self):
        folder_path = self.stereo_rect_path_entry.get()
        
        for label in self.stereo_img_labels.values():
            self._display_image(label, None)
        
        if not folder_path or not os.path.isdir(folder_path):
            messagebox.showerror(
                "Error", "Please select a valid folder containing ONE stereo pair (cam1_TS.ext / cam2_TS.ext).")
            return
        if not self.stereo_calib_results:
            messagebox.showerror(
                "Missing Data", "Run Stereo Calibration (Tab 1) first.")
            return
        
        M1 = self.stereo_calib_results.get("M1")
        d1 = self.stereo_calib_results.get("d1")
        M2 = self.stereo_calib_results.get("M2")
        d2 = self.stereo_calib_results.get("d2")
        R = self.stereo_calib_results.get("R")
        T = self.stereo_calib_results.get("T")
        image_size = self.stereo_calib_results.get("image_size")
        F = self.stereo_calib_results.get("F")  # Get F for epiline drawing
        missing_calib = [k for k, v in {"M1": M1, "d1": d1, "M2": M2, "d2": d2,
                                        "R": R, "T": T, "F": F, "image_size": image_size}.items() if v is None]
        if missing_calib:
            messagebox.showerror(
                "Missing Calibration Data", f"Could not find required parameters from Tab 1: {', '.join(missing_calib)}")
            return
        
        print(
            f"Starting Stereo Rectification for pair in folder: {folder_path}")
        print("Using parameters from Stereo Calibration (Tab 1).")
        
        self._run_task_with_loading(
            task_func=backend.stereo_rect,
            kwargs={'stereo_path': folder_path, 'M1': M1, 'd1': d1, 'M2': M2, 'd2': d2,
                    'R': R, 'T': T, 'F': F, 'image_size': image_size},  # Pass F too
            loading_msg="Running Rectification...", result_handler=self._handle_stereo_rect_result
        )

    # 2. Handler
    def _handle_stereo_rect_result(self, results):
        if isinstance(results, str):
            print(f"Error: {results}")
            messagebox.showerror("Rectification Failed", results)
            self.stereo_rect_results = None
        
        elif isinstance(results, dict) and "Error" in results:
            print(f"Error: {results['Error']}")
            messagebox.showerror("Rectification Failed", results['Error'])
            self.stereo_rect_results = None
        
        elif isinstance(results, dict):
            self.stereo_rect_results = results
            print("Stereo Rectification Success. Displaying images...")
            display_keys = ["Original Left", "Original Right",
                            "Drawn Rectified Left", "Drawn Rectified Right"]
            fixed_size = (640, 360)
            for title in display_keys:
                img_label = self.stereo_img_labels.get(title)
                img_to_display = self.stereo_rect_results.get(title)
                if img_label and img_to_display is not None:
                    try:
                        self._display_image(img_label, cv2.resize(
                            img_to_display, fixed_size, interpolation=cv2.INTER_AREA))
                    except Exception as e:
                        print(f"Error resizing/displaying '{title}': {e}")
                        self._display_image(img_label, None)
                elif img_label:
                    print(f"Warning: Image '{title}' is None.")
                    self._display_image(img_label, None)
            messagebox.showinfo("Success", "Rectification complete.")
        
        else:
            print(
                f"Error: Unexpected rectification result type: {type(results)}")
            messagebox.showerror(
                "Error", "Rectification returned unexpected data.")
            self.stereo_rect_results = None

    # 3. Feature Detection and Matching
    def run_feat_detect_match_threaded(self):
        folder_path = self.feat_detect_match_path_entry.get()
        self.feat_detect_match_output_text.delete(1.0, tk.END)
        
        for label in self.feat_img_labels.values():
            self._display_image(label, None)
        
        if not folder_path or not os.path.isdir(folder_path):
            messagebox.showerror(
                "Error", "Please select a valid folder containing ONE stereo pair (cam1_TS.ext / cam2_TS.ext).")
            return
        
        M1 = self.stereo_calib_results.get(
            "M1") if self.stereo_calib_results else None
        self.redirector_feat.start_redirect()
        
        print(
            f"Starting Feature Detection & Matching for pair in folder: {folder_path}")
        
        if M1 is None:
            print(
                "Stereo Calib (Tab 1) not run/failed. Using default intrinsics for E matrix.")
        else:
            print("Using M1 from Stereo Calibration for E matrix calculation.")
        self._run_task_with_loading(task_func=backend.feat_detect_match, kwargs={
                                    'stereo_path': folder_path, 'cameraMatrix1': M1}, loading_msg="Running Detection & Matching...", result_handler=self._handle_feat_detect_match_result)

    # 3. Handler
    def _handle_feat_detect_match_result(self, results):
        if isinstance(results, str):
            print(f"Error: {results}")
            self.redirector_feat.stop_redirect()
            messagebox.showerror("Detection/Matching Failed", results)
            self.feat_detect_match_results = None
        
        elif isinstance(results, dict) and "Error" in results:
            print(f"Error: {results['Error']}")
            self.redirector_feat.stop_redirect()
            messagebox.showerror("Detection/Matching Failed", results['Error'])
            self.feat_detect_match_results = None
        
        elif isinstance(results, dict):
            self.feat_detect_match_results = results
            self.feat_detect_match_output_text.delete(1.0, tk.END)
        
            kp_l = results.get("Num Keypoints Left", "N/A")
            kp_r = results.get("Num Keypoints Right", "N/A")
        
            match_raw = results.get("Num Raw Matches", "N/A")
            match_good = results.get("Num Good Matches", "N/A")
        
            summary = (
                f"--- Detection & Matching Summary ---\nLeft KPs: {kp_l}\nRight KPs: {kp_r}\nRaw Matches: {match_raw}\nGood Matches: {match_good}\n" + "-"*36 + "\nSUCCESS: Detection & matching complete.\n")
        
            self.feat_detect_match_output_text.insert(tk.END, summary)
            self.feat_detect_match_output_text.see(tk.END)
            self.redirector_feat.stop_redirect()
        
            for title, img_label in self.feat_img_labels.items():
                img = self.feat_detect_match_results.get(title)
                if img is not None:
                    fixed_size = (
                        1280, 360) if "Matched Images" in title else (640, 360)
                    try:
                        self._display_image(img_label, cv2.resize(
                            img, fixed_size, interpolation=cv2.INTER_AREA))
                    except Exception as e:
                        print(f"Error resizing/displaying '{title}': {e}")
                        self._display_image(img_label, None)
                else:
                    print(f"Warning: Image '{title}' is None.")
                    self._display_image(img_label, None)
            messagebox.showinfo("Success", "Detection & matching complete.")
        
        else:
            print(f"Error: Unexpected matching result type: {type(results)}")
            self.redirector_feat.stop_redirect()
            messagebox.showerror("Error", "Matching returned unexpected data.")
            self.feat_detect_match_results = None

    # 4. Stereo Geometry Estimation
    def run_stereo_geometry_estimation_threaded(self):
        folder_path = self.stereo_geometry_estimation_path_entry.get()
        self.stereo_geometry_output_text.delete(1.0, tk.END)
        
        if not folder_path or not os.path.isdir(folder_path):
            messagebox.showerror(
                "Error", "Please select a valid folder containing ONE stereo pair (cam1_TS.ext / cam2_TS.ext).")
            return
        
        M1 = self.stereo_calib_results.get(
            "M1") if self.stereo_calib_results else None
        self.redirector_geom.start_redirect()
        
        print(
            f"Starting Standalone Stereo Geometry Estimation for pair in folder: {folder_path}")
        
        if M1 is None:
            print(
                "Stereo Calib (Tab 1) not run/failed. Using default intrinsics for E matrix.")
        else:
            print("Using M1 from Stereo Calibration for E matrix calculation.")
        
        self._run_task_with_loading(task_func=backend.stereo_geometry_estimation, kwargs={
                                    'stereo_path': folder_path, 'cameraMatrix1': M1}, loading_msg="Estimating Geometry...", result_handler=self._handle_stereo_geometry_result)

    # 4. Handler
    def _handle_stereo_geometry_result(self, results):
        if isinstance(results, str):
            print(f"Error: {results}")
            self.redirector_geom.stop_redirect()
            messagebox.showerror("Geometry Estimation Failed", results)
            self.stereo_geometry_results = None
        
        elif isinstance(results, dict) and "Error" in results:
            print(f"Error: {results['Error']}")
            self.redirector_geom.stop_redirect()
            messagebox.showerror(
                "Geometry Estimation Failed", results['Error'])
            self.stereo_geometry_results = None
        
        elif isinstance(results, dict):
            self.stereo_geometry_results = results
            self.stereo_geometry_output_text.delete(1.0, tk.END)
            output_str = "--- Feature-Based Stereo Geometry Results ---\n\n"
            f_mat = results.get("Fundamental Matrix")
        
            output_str += "Fundamental Matrix (F):\n" + (np.array2string(
                f_mat, precision=4, suppress_small=True) if f_mat is not None else "N/A") + "\n\n"
        
            e_mat = results.get("Essential Matrix")
            det_e = results.get("Determinant E")
            svd_count_e = results.get("Non Zero SVD E")
            svd_vals_e = results.get("Singular Values E")
        
            output_str += "Essential Matrix (E):\n" + (np.array2string(
                e_mat, precision=4, suppress_small=True) if e_mat is not None else "N/A") + "\n"
        
            if det_e is not None:
                output_str += f"  Determinant: {det_e:.4e}\n"
            if svd_vals_e is not None:
                output_str += f"  Singular Values: {np.array2string(svd_vals_e, precision=3, suppress_small=True)}\n"
            if svd_count_e is not None:
                output_str += f"  Non-Zero Singular Values: {svd_count_e} / 3\n"
        
            output_str += "\n"
            r_mat = results.get("Rotation Matrix")
        
            output_str += "Rotation Matrix (R):\n" + (np.array2string(
                r_mat, precision=4, suppress_small=True) if r_mat is not None else "N/A") + "\n\n"
            t_vec = results.get("Translation Vector")
        
            output_str += "Translation Vector (T):\n" + (np.array2string(
                t_vec, precision=4, suppress_small=True) if t_vec is not None else "N/A") + "\n"
            output_str += "\n" + "-"*31 + \
                "\nSUCCESS: Feature-based geometry estimation complete.\n"
        
            self.stereo_geometry_output_text.insert(tk.END, output_str)
            self.stereo_geometry_output_text.see(tk.END)
            self.redirector_geom.stop_redirect()
            messagebox.showinfo(
                "Success", "Feature-based stereo geometry estimation complete.")
        
        else:
            print(f"Error: Unexpected geometry result type: {type(results)}")
            self.redirector_geom.stop_redirect()
            messagebox.showerror(
                "Error", "Geometry estimation returned unexpected data.")
            self.stereo_geometry_results = None

    # 5. Triangulation
    def run_triangulation_and_reconstruction_threaded(self):
        self.triangulation_output_text.delete(1.0, tk.END)
        
        if not self.stereo_calib_results:
            messagebox.showerror(
                "Missing Data", "Run Stereo Calibration (Tab 1) first.")
            return
        
        if not self.feat_detect_match_results:
            messagebox.showerror(
                "Missing Data", "Run Feature Detection & Matching (Tab 3) first.")
            return
        pts1 = self.feat_detect_match_results.get("Left Aligned Keypoints")
        pts2 = self.feat_detect_match_results.get("Right Aligned Keypoints")
        
        imgL_color = self.feat_detect_match_results.get("Left Color Image")
        
        camera_matrix = self.stereo_calib_results.get("M1")
        rotation_matrix = self.stereo_calib_results.get("R")
        translation_vector = self.stereo_calib_results.get("T")
        
        missing_match = [n for n, v in [
            ("L KP", pts1), ("R KP", pts2), ("L Color", imgL_color)] if v is None]
        missing_calib = [n for n, v in [
            ("M1", camera_matrix), ("R", rotation_matrix), ("T", translation_vector)] if v is None]
        
        if missing_match or missing_calib:
            messagebox.showerror(
                "Missing Data", f"Could not retrieve: {', '.join(missing_match + missing_calib)}")
            return
        
        self.redirector_tri.start_redirect()
        
        print("Starting Triangulation (Feature-Based Points, Calibrated Pose)...")
        print(f"Using {len(pts1)} points from Tab 3.")
        print("Using M1, R, T from Tab 1.")
        self._run_task_with_loading(task_func=backend.triangulation_and_3D_reconstruction, args=(pts1, pts2, camera_matrix, rotation_matrix,
                                    translation_vector, imgL_color), loading_msg="Triangulating & Visualizing...", result_handler=self._handle_triangulation_result)

    # 5. Handler
    def _handle_triangulation_result(self, results):
        if isinstance(results, dict) and "Error" in results:
            print(f"Error: {results['Error']}")
            self.redirector_tri.stop_redirect()
            messagebox.showerror("Triangulation Failed", results['Error'])
            self.triangulation_results = None
        
        elif isinstance(results, dict) and "3D Points" in results:
            self.triangulation_results = results
            num_points = len(results["3D Points"])
            warning_msg = results.get("Warning")
            print(f"Success. Generated {num_points} 3D points.")
        
            if warning_msg:
                print(f"Warning during visualization: {warning_msg}")
            else:
                print("Colored point cloud display attempted.")
        
            self.redirector_tri.stop_redirect()
            msg = f"Triangulation successful ({num_points} points)."
        
            if warning_msg:
                msg += f"\nNote: {warning_msg}"
            else:
                msg += "\nCheck Open3D window (if backend opened one)."
            messagebox.showinfo("Success", msg)
        
        else:
            print("Unexpected result from triangulation:", results)
            self.redirector_tri.stop_redirect()
            messagebox.showerror("Error", "Unknown triangulation result.")
            self.triangulation_results = None

    # 6a. Disparity Calculation (Dense)
    def run_disparity_calculation_threaded(self):
        self.disparity_output_text.delete(1.0, tk.END)
        self._display_image(self.disparity_img_label, None)
        
        if not self.stereo_rect_results:
            messagebox.showerror(
                "Missing Data", "Run 'Stereo Rectification' (Tab 2) first.")
            return
        
        if "Error" in self.stereo_rect_results:
            messagebox.showerror(
                "Cannot Proceed", f"Previous Rectification failed: {self.stereo_rect_results['Error']}")
            return
        
        imgL_rect_gray = self.stereo_rect_results.get("Rectified Left")
        imgR_rect_gray = self.stereo_rect_results.get("Rectified Right")
        imgL_rect_color = self.stereo_rect_results.get("Rectified Color Left")
        
        imgL_for_disp = imgL_rect_gray if imgL_rect_gray is not None else imgL_rect_color
        imgR_for_disp = imgR_rect_gray if imgR_rect_gray is not None else self.stereo_rect_results.get(
            "Rectified Color Right")
        if imgL_for_disp is None or imgR_for_disp is None:
            messagebox.showerror(
                "Missing Data", "Rectified images not found in Tab 2 results.")
            return
        guide_image = imgL_rect_color if imgL_rect_color is not None else imgL_for_disp
        print("Starting Disparity Calculation (SGBM + WLS)...")
        self._run_task_with_loading(task_func=backend.disparity_calculation, args=(imgL_for_disp, imgR_for_disp), kwargs={
                                    'guide_image': guide_image}, loading_msg="Calculating Disparity...", result_handler=self._handle_disparity_result)

    # 6a. Handler
    def _handle_disparity_result(self, results):
        if isinstance(results, dict) and "Error" in results:
            error_msg = results['Error']
            print(f"Error: {error_msg}")
            self.disparity_output_text.insert(tk.END, f"ERROR: {error_msg}\n")
            self.disparity_output_text.see(tk.END)
            messagebox.showerror("Disparity Failed", error_msg)
            self.disparity_calculation_results = results
        elif isinstance(results, dict) and "Disparity Map Color" in results:
            self.disparity_calculation_results = results
            disparity_map_vis = results["Disparity Map Color"]
            print("Success. Displaying filtered disparity heatmap.")
            self.disparity_output_text.insert(
                tk.END, "SUCCESS: Disparity map calculated and filtered.\n")
            self.disparity_output_text.see(tk.END)
            fixed_size = (640, 480)
            try:
                self._display_image(self.disparity_img_label, cv2.resize(
                    disparity_map_vis, fixed_size, interpolation=cv2.INTER_AREA))
            except Exception as e:
                print(f"Error resizing/displaying disparity map: {e}")
                self._display_image(self.disparity_img_label, None)
            messagebox.showinfo(
                "Success", "Disparity map calculated and filtered.")
        else:
            print("Unexpected result from disparity calculation:", results)
            self.disparity_output_text.insert(
                tk.END, f"ERROR: Unknown result from disparity calculation.\n")
            self.disparity_output_text.see(tk.END)
            messagebox.showerror(
                "Error", "Unknown result from disparity calculation.")
            self.disparity_calculation_results = None

    # 6b. Visualize 3D Point Cloud (Dense)
    def run_visualize_3d_threaded(self):
        self.disparity_output_text.delete(1.0, tk.END)
        if not self.disparity_calculation_results:
            messagebox.showerror(
                "Missing Data", "Run 'Run Disparity' (step 6a) first.")
            return
        if "Error" in self.disparity_calculation_results:
            messagebox.showerror(
                "Cannot Visualize", f"Previous disparity failed: {self.disparity_calculation_results['Error']}")
            return
        if not self.stereo_rect_results:
            messagebox.showerror(
                "Missing Data", "Run 'Stereo Rectification' (Tab 2) first.")
            return
        if "Error" in self.stereo_rect_results:
            messagebox.showerror(
                "Cannot Visualize", f"Rectification failed: {self.stereo_rect_results['Error']}")
            return
        raw_disparity_map = self.disparity_calculation_results.get(
            "Raw Disparity")
        Q = self.stereo_rect_results.get("disp2depth map")
        colors = self.stereo_rect_results.get("Rectified Color Left")
        missing = [n for n, v in [
            ("Raw Disp", raw_disparity_map), ("Q", Q), ("L Color", colors)] if v is None]
        if missing:
            messagebox.showerror(
                "Missing Data", f"Could not retrieve from previous steps: {', '.join(missing)}")
            return
        print("Starting 3D Visualization from Dense Disparity...")
        self._run_task_with_loading(task_func=backend.visualize_point_cloud_disparity, args=(
            raw_disparity_map, Q, colors), loading_msg="Generating & Visualizing 3D Cloud...", result_handler=self._handle_visualize_3d_result)

    # 6b. Handler
    def _handle_visualize_3d_result(self, error_message):
        if error_message:
            print(f"Error: {error_message}")
            self.disparity_output_text.insert(
                tk.END, f"ERROR: {error_message}\n")
            self.disparity_output_text.see(tk.END)
            messagebox.showerror("Visualization Failed", error_message)
        else:
            print("Point cloud display attempted.")
            self.disparity_output_text.insert(
                tk.END, "SUCCESS: Point cloud display attempted.\n")
            self.disparity_output_text.see(tk.END)

    # --- Browse Methods ---
    def _set_path_entry(self, entry_widget, folder_path): entry_widget.delete(
        0, tk.END); entry_widget.insert(0, folder_path)

    def _auto_populate_paths(self, selected_folder):
        if not self.stereo_rect_path_entry.get():
            self._set_path_entry(self.stereo_rect_path_entry, selected_folder)
        if not self.feat_detect_match_path_entry.get():
            self._set_path_entry(
                self.feat_detect_match_path_entry, selected_folder)
        if not self.stereo_geometry_estimation_path_entry.get():
            self._set_path_entry(
                self.stereo_geometry_estimation_path_entry, selected_folder)

    def stereo_calib_browse_folder(self):
        folder = filedialog.askdirectory(
            title="Select Folder with Stereo Calibration Pairs (cam1_TS.ext / cam2_TS.ext)")
        if folder:
            self.stereo_calib_path_entry.delete(0, tk.END)
            self.stereo_calib_path_entry.insert(0, folder)
            self._auto_populate_paths(folder)

    def stereo_rect_browse_folder(self):
        folder = filedialog.askdirectory(
            title="Select Folder with ONE Stereo Pair (cam1_TS.ext / cam2_TS.ext)")
        if folder:
            self._set_path_entry(self.stereo_rect_path_entry, folder)

    def feat_detect_match_browse_folder(self):
        folder = filedialog.askdirectory(
            title="Select Folder with ONE Stereo Pair (cam1_TS.ext / cam2_TS.ext)")
        if folder:
            self._set_path_entry(self.feat_detect_match_path_entry, folder)

    def stereo_geometry_estimation_browse_folder(self):
        folder = filedialog.askdirectory(
            title="Select Folder with ONE Stereo Pair (cam1_TS.ext / cam2_TS.ext)")
        if folder:
            self._set_path_entry(
                self.stereo_geometry_estimation_path_entry, folder)


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = NotebookGUI(root)
    root.mainloop()
