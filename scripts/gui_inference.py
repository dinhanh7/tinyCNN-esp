#!/usr/bin/env python3
"""
gui_inference.py

Tkinter GUI for sending images to ESP32 via Serial.
Allows selecting COM port, browsing images, and viewing results graphically.
"""
import sys
import os
import time
import threading
import subprocess
import platform
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import serial
import serial.tools.list_ports

# Import inference logic from serial_inference.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from serial_inference import preprocess_image, send_and_receive, INPUT_SIZE, CIFAR10_CLASSES
except ImportError:
    messagebox.showerror("Error", "Could not import serial_inference.py. Ensure it is in the same folder.")
    sys.exit(1)


class InferenceGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TinyCNN ESP32 - Serial Inference")
        self.geometry("700x560")
        self.resizable(False, False)

        self.ser = None
        self.image_path = None
        self.photo_image = None
        self.is_scanning = False
        
        self.setup_ui()
        self.refresh_ports()

    def setup_ui(self):
        # Base padding
        self.config(padx=10, pady=10)

        # --- Top frame (Connection) ---
        conn_frame = ttk.LabelFrame(self, text="Connection", padding=10)
        conn_frame.pack(fill=tk.X, pady=(0, 10))

        conn_row = ttk.Frame(conn_frame)
        conn_row.pack(fill=tk.X)

        ttk.Label(conn_row, text="COM Port:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(conn_row, textvariable=self.port_var, state="readonly", width=15)
        self.port_combo.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(conn_row, text="Refresh", command=self.refresh_ports).pack(side=tk.LEFT, padx=(0, 10))
        self.btn_scan = ttk.Button(conn_row, text="Scan ESP32", command=self.scan_active_esp32)
        self.btn_scan.pack(side=tk.LEFT, padx=(0, 10))
        
        self.btn_connect = ttk.Button(conn_row, text="Connect", command=self.toggle_connection)
        self.btn_connect.pack(side=tk.LEFT)

        self.btn_flash = ttk.Button(conn_row, text="Flash ESP32", command=self.flash_esp32)
        self.btn_flash.pack(side=tk.LEFT, padx=(10, 0))

        self.lbl_status = ttk.Label(conn_frame, text="Disconnected", foreground="red", font=("Arial", 10, "bold"))
        self.lbl_status.pack(anchor=tk.W, pady=(6, 0))

        # --- Main frame (Split left/right) ---
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left side: Image selection
        left_frame = ttk.LabelFrame(main_frame, text="Image Input", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.lbl_image = tk.Label(left_frame, bg="gray", width=32, height=16)
        self.lbl_image.pack(pady=10)

        self.lbl_img_path = ttk.Label(left_frame, text="No image selected")
        self.lbl_img_path.pack(pady=(0, 10))

        btn_browse = ttk.Button(left_frame, text="Browse Image", command=self.browse_image)
        btn_browse.pack(pady=5)

        self.btn_infer = ttk.Button(left_frame, text="Run Inference", command=self.run_inference, state=tk.DISABLED)
        self.btn_infer.pack(pady=10)

        # Right side: Results
        right_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.lbl_time = ttk.Label(right_frame, text="Inference time: -- ms", font=("Arial", 10, "bold"))
        self.lbl_time.pack(anchor=tk.W, pady=(0, 2))

        self.lbl_cpu = ttk.Label(right_frame, text="CPU Freq: -- MHz", font=("Arial", 10))
        self.lbl_cpu.pack(anchor=tk.W, pady=(0, 2))

        self.lbl_static_ram = ttk.Label(right_frame, text="Static RAM: -- KB", font=("Arial", 10))
        self.lbl_static_ram.pack(anchor=tk.W, pady=(0, 2))

        self.lbl_free_ram = ttk.Label(right_frame, text="Free Heap: -- KB", font=("Arial", 10))
        self.lbl_free_ram.pack(anchor=tk.W, pady=(0, 2))

        self.lbl_min_heap = ttk.Label(right_frame, text="Min Free Heap: -- KB", font=("Arial", 10))
        self.lbl_min_heap.pack(anchor=tk.W, pady=(0, 2))

        self.lbl_max_alloc = ttk.Label(right_frame, text="Max Alloc Block: -- KB", font=("Arial", 10))
        self.lbl_max_alloc.pack(anchor=tk.W, pady=(0, 10))
        
        # Result bars
        self.bars = []
        for class_name in CIFAR10_CLASSES:
            row = ttk.Frame(right_frame)
            row.pack(fill=tk.X, pady=2)
            
            lbl_name = ttk.Label(row, text=class_name, width=12)
            lbl_name.pack(side=tk.LEFT)
            
            progress = ttk.Progressbar(row, orient=tk.HORIZONTAL, length=150, mode='determinate')
            progress.pack(side=tk.LEFT, padx=5)
            
            lbl_pct = ttk.Label(row, text="0.0%", width=6)
            lbl_pct.pack(side=tk.LEFT)
            
            self.bars.append((progress, lbl_pct, lbl_name))

    def flash_esp32(self):
        if self.ser and self.ser.is_open:
            self.toggle_connection()  # Disconnect to free the port for PlatformIO
            
        self.lbl_status.config(text="Flashing...", foreground="blue")
        self.btn_flash.config(state=tk.DISABLED)
        self.btn_connect.config(state=tk.DISABLED)
        self.btn_infer.config(state=tk.DISABLED)
        
        # Run in thread so GUI doesn't freeze
        threading.Thread(target=self._flash_worker, daemon=True).start()

    def _flash_worker(self):
        proj_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "esp32_inference")
        port = self.port_var.get()
        
        # Use sys.executable to run platformio module universally (works on both Windows & Ubuntu)
        cmd = [sys.executable, "-m", "platformio", "run", "--target", "upload"]
        if port:
            cmd.extend(["--upload-port", port])

        try:
            result = subprocess.run(cmd, cwd=proj_dir, capture_output=True, text=True)
            if result.returncode == 0:
                self.after(0, lambda: messagebox.showinfo("Success", "Flashed ESP32 successfully!"))
                self.after(0, lambda: self.lbl_status.config(text="Flash Success", foreground="green"))
            else:
                print(result.stdout)  # Print to terminal for deep debugging
                print(result.stderr)
                
                # Show trailing end of error in GUI
                err_msg = result.stderr or result.stdout
                self.after(0, lambda: messagebox.showerror("Upload Error", f"Failed to flash code.\n\n{err_msg[-500:]}"))
                self.after(0, lambda: self.lbl_status.config(text="Flash Failed", foreground="red"))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to execute platformio:\n{str(e)}"))
            self.after(0, lambda: self.lbl_status.config(text="Error", foreground="red"))
        finally:
            self.after(0, lambda: self.btn_flash.config(state=tk.NORMAL))
            self.after(0, lambda: self.btn_connect.config(state=tk.NORMAL))
            self.after(0, self.check_ready)

    def refresh_ports(self):
        ports = sorted([port.device for port in serial.tools.list_ports.comports()])
        self.port_combo['values'] = ports
        if ports and not self.port_var.get():
            self.port_combo.current(0)

    def scan_active_esp32(self):
        if self.is_scanning:
            return
        if self.ser and self.ser.is_open:
            messagebox.showinfo("Scan ESP32", "Please disconnect current serial connection before scanning.")
            return

        ports = sorted([port.device for port in serial.tools.list_ports.comports()])
        if not ports:
            messagebox.showwarning("Scan ESP32", "No serial ports found.")
            return

        self.is_scanning = True
        self.lbl_status.config(text="Scanning ports...", foreground="blue")
        self.btn_scan.config(state=tk.DISABLED)
        self.btn_connect.config(state=tk.DISABLED)
        self.btn_flash.config(state=tk.DISABLED)

        threading.Thread(target=self._scan_worker, args=(ports,), daemon=True).start()

    def _probe_port_for_esp32(self, port):
        try:
            with serial.Serial(port, 115200, timeout=0.25, write_timeout=0.5) as test_ser:
                # Opening a port often resets ESP32; wait long enough for setup() to finish.
                deadline = time.time() + 7.0
                last_ping = 0.0
                while time.time() < deadline:
                    now = time.time()
                    if now - last_ping >= 0.35:
                        test_ser.write(b"IMG\n")
                        test_ser.flush()
                        last_ping = now

                    if test_ser.in_waiting:
                        line = test_ser.readline().decode("utf-8", errors="ignore").strip()
                        if line == "READY":
                            return True

                    time.sleep(0.02)
                return False
        except (serial.SerialException, OSError):
            return False

    def _scan_worker(self, ports):
        active_ports = []
        for port in ports:
            if self._probe_port_for_esp32(port):
                active_ports.append(port)

        def finish_scan():
            self.is_scanning = False
            self.btn_scan.config(state=tk.NORMAL)
            self.btn_connect.config(state=tk.NORMAL)
            self.btn_flash.config(state=tk.NORMAL)

            if active_ports:
                self.port_combo['values'] = active_ports
                self.port_var.set(active_ports[0])
                self.lbl_status.config(text=f"Found {len(active_ports)} ESP32", foreground="green")
                if len(active_ports) == 1:
                    messagebox.showinfo("Scan ESP32", f"Detected ESP32 on {active_ports[0]}.")
                else:
                    messagebox.showinfo("Scan ESP32", "Detected ESP32 ports: " + ", ".join(active_ports))
            else:
                self.refresh_ports()
                self.lbl_status.config(text="No ESP32 detected", foreground="red")
                messagebox.showwarning(
                    "Scan ESP32",
                    "No active ESP32 detected. Check cable/driver, then try Refresh or Flash.",
                )

            self.check_ready()

        self.after(0, finish_scan)

    def toggle_connection(self):
        if self.ser and self.ser.is_open:
            # Disconnect
            self.ser.close()
            self.ser = None
            self.btn_connect.config(text="Connect")
            self.lbl_status.config(text="Disconnected", foreground="red")
            self.port_combo.config(state="readonly")
            self.check_ready()
        else:
            # Connect
            port = self.port_var.get()
            if not port:
                messagebox.showwarning("Warning", "Select a COM port first.")
                return
            
            try:
                self.ser = serial.Serial(port, 115200, timeout=2)
                self.btn_connect.config(text="Disconnect")
                self.lbl_status.config(text="Connecting...", foreground="orange")
                self.port_combo.config(state="disabled")
                self.check_ready()
                
                # Run background thread to drain startup messages and wait for boot
                threading.Thread(target=self._connection_worker, daemon=True).start()
            except serial.SerialException as e:
                messagebox.showerror("Connection Error", str(e))

    def _connection_worker(self):
        time.sleep(1.5)  # Let ESP32 reboot
        if self.ser and self.ser.is_open:
            self.ser.reset_input_buffer()
            self.after(0, lambda: self.lbl_status.config(text="Connected", foreground="green"))

    def browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if path:
            self.image_path = path
            name = os.path.basename(path)
            self.lbl_img_path.config(text=name if len(name) < 30 else "..." + name[-27:])
            
            try:
                # Load and display preview (scaled to 256x256 max)
                img = Image.open(path)
                img.thumbnail((256, 256))
                self.photo_image = ImageTk.PhotoImage(img)
                self.lbl_image.config(image=self.photo_image, width=0, height=0)
                self.check_ready()
            except Exception as e:
                messagebox.showerror("Error", f"Could not open image: {e}")

    def check_ready(self):
        if self.ser and self.ser.is_open and self.image_path:
            self.btn_infer.config(state=tk.NORMAL)
        else:
            self.btn_infer.config(state=tk.DISABLED)

    def run_inference(self):
        if not self.ser or not self.ser.is_open or not self.image_path:
            return
            
        self.btn_infer.config(state=tk.DISABLED, text="Running...")
        self.lbl_time.config(text="Inference time: -- ms")
        self.lbl_cpu.config(text="CPU Freq: -- MHz")
        self.lbl_static_ram.config(text="Static RAM: -- KB")
        self.lbl_free_ram.config(text="Free Heap: -- KB")
        self.lbl_min_heap.config(text="Min Free Heap: -- KB")
        self.lbl_max_alloc.config(text="Max Alloc Block: -- KB")
        
        # Reset bars
        for prog, lbl, name in self.bars:
            prog['value'] = 0
            lbl.config(text="0.0%")
            name.config(font=("Arial", 9, "normal"))

        # Run in thread so GUI doesn't freeze
        threading.Thread(target=self._inference_worker, daemon=True).start()

    def _inference_worker(self):
        try:
            # 1. Preprocess
            img_data = preprocess_image(self.image_path)
            
            # 2. Transmit & Receive
            res = send_and_receive(self.ser, img_data)
            
            # 3. Update GUI
            if res[0] is not None:
                results, metrics = res
                self.after(0, self._update_results, results, metrics)
            else:
                self.after(0, lambda: messagebox.showerror("Error", "Inference timed out or failed."))
                
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.after(0, lambda: self.btn_infer.config(state=tk.NORMAL, text="Run Inference"))

    def _update_results(self, results, metrics):
        self.lbl_time.config(text=f"Inference time: {metrics.get('TIME', 0)} ms")
        self.lbl_cpu.config(text=f"CPU Freq: {metrics.get('CPU_FREQ', 0)} MHz")
        self.lbl_static_ram.config(text=f"Static RAM: {metrics.get('STATIC_RAM', 0) / 1024:.1f} KB")
        self.lbl_free_ram.config(text=f"Free Heap: {metrics.get('FREE_HEAP', 0) / 1024:.1f} KB")
        self.lbl_min_heap.config(text=f"Min Free Heap: {metrics.get('MIN_HEAP', 0) / 1024:.1f} KB")
        self.lbl_max_alloc.config(text=f"Max Alloc Block: {metrics.get('MAX_ALLOC', 0) / 1024:.1f} KB")
        
        # Result dict is {"class_name": float_prob}
        # Update bars matching CIFAR10_CLASSES
        best_prob = -1
        best_idx = 0
        
        for i, class_name in enumerate(CIFAR10_CLASSES):
            prob = results.get(class_name, 0.0)
            pct = prob * 100
            
            prog, lbl, name_lbl = self.bars[i]
            prog['value'] = pct
            lbl.config(text=f"{pct:.1f}%")
            
            if pct > best_prob:
                best_prob = pct
                best_idx = i
                
        # Bold the best prediction
        for i, (_, _, name_lbl) in enumerate(self.bars):
            if i == best_idx:
                name_lbl.config(font=("Arial", 9, "bold"))
            else:
                name_lbl.config(font=("Arial", 9, "normal"))


if __name__ == "__main__":
    app = InferenceGUI()
    app.mainloop()
