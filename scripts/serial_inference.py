#!/usr/bin/env python3
"""
serial_inference.py

Send images to ESP32 via Serial for inference.
Handles image loading, preprocessing, quantization, and result display.

Usage:
  python scripts/serial_inference.py --port COM3 --image image/image.png
  python scripts/serial_inference.py --port COM3 --interactive
  python scripts/serial_inference.py --port COM3 --folder image/

Requirements:
  pip install pyserial pillow numpy
"""
import argparse
import os
import sys
import time
import glob
import numpy as np
from PIL import Image

try:
    import serial
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

# ---- Model parameters (must match model_int8.onnx) ----
INPUT_SCALE = 0.016141029074788094
INPUT_ZP    = 123
INPUT_SIZE  = 3072  # 3 * 32 * 32

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def preprocess_image(image_path):
    """Load image, resize to 32x32, normalize, quantize to uint8."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((32, 32), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0  # HWC
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - CIFAR10_MEAN[c]) / CIFAR10_STD[c]

    arr_chw = arr.transpose(2, 0, 1)  # CHW
    arr_q = np.clip(np.round(arr_chw / INPUT_SCALE + INPUT_ZP), 0, 255).astype(np.uint8)
    return arr_q.flatten()


def send_and_receive(ser, image_data):
    """Send image to ESP32 and receive inference results."""
    # Flush any pending data
    ser.reset_input_buffer()

    # Send header
    ser.write(b"IMG\n")
    ser.flush()

    # Wait for READY
    ready_timeout = time.time() + 2
    ready = False
    while time.time() < ready_timeout:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line == "READY":
                ready = True
                break
    
    if not ready:
        print("  Error: Did not receive READY signal from ESP32")
        return None, 0

    # Send image data
    ser.write(image_data.tobytes())
    ser.flush()

    # Wait for READY or RES
    results = {}
    inference_time = 0
    timeout = time.time() + 10  # 10 second timeout

    while time.time() < timeout:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()

            if line == "RES":
                # Parse key-value pairs until END
                metrics = {}
                results = {}
                while time.time() < timeout:
                    if ser.in_waiting:
                        sub_line = ser.readline().decode("utf-8", errors="ignore").strip()
                        if sub_line == "END":
                            return results, metrics
                        elif sub_line.startswith("CLASS:"):
                            parts = sub_line.split(":")
                            if len(parts) >= 3:
                                results[parts[1]] = float(parts[2])
                        elif ":" in sub_line:
                            key, val = sub_line.split(":", 1)
                            try:
                                metrics[key] = int(val)
                            except ValueError:
                                metrics[key] = val
                
                return results, metrics

            elif line.startswith("ERR"):
                print(f"  ESP32 error: {line}")
                return None, {}

    print("  Timeout waiting for ESP32 response")
    return None, {}


def print_results(results, metrics, image_path):
    """Pretty-print inference results."""
    if not results:
        print("  No results received.")
        return

    # Sort by probability
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  Image: {os.path.basename(image_path)}")
    print(f"  Inference time: {metrics.get('TIME', 0)} ms")
    print(f"  CPU Frequency : {metrics.get('CPU_FREQ', 240)} MHz")
    print(f"  Static RAM    : {metrics.get('STATIC_RAM', 0) / 1024:.1f} KB")
    print(f"  Free Heap     : {metrics.get('FREE_HEAP', 0) / 1024:.1f} KB")
    print(f"  Min Free Heap : {metrics.get('MIN_HEAP', 0) / 1024:.1f} KB")
    print(f"  Max Alloc Blk : {metrics.get('MAX_ALLOC', 0) / 1024:.1f} KB")
    print(f"  {'─' * 35}")

    for i, (cls, prob) in enumerate(sorted_results):
        bar_len = int(prob * 100 / 2)
        bar = "█" * bar_len
        marker = " ◄" if i == 0 else ""
        print(f"  {cls:12s} {prob*100:6.2f}% {bar}{marker}")

    best_class = sorted_results[0][0]
    best_prob = sorted_results[0][1]
    print(f"\n  >> {best_class} ({best_prob*100:.2f}%)")


def run_single(ser, image_path):
    """Run inference on a single image."""
    print(f"\n  Preprocessing {os.path.basename(image_path)}...")
    img_data = preprocess_image(image_path)

    print(f"  Sending {INPUT_SIZE} bytes to ESP32...")
    res = send_and_receive(ser, img_data)
    if res[0] is not None:
        results, metrics = res
        print_results(results, metrics, image_path)


def run_interactive(ser):
    """Interactive mode: keep asking for images."""
    print("\n  Interactive mode. Type image path or 'q' to quit.\n")
    while True:
        try:
            path = input("  Image path (or 'q'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if path.lower() in ('q', 'quit', 'exit'):
            break

        if not os.path.exists(path):
            print(f"  File not found: {path}")
            continue

        run_single(ser, path)
        print()


def run_folder(ser, folder_path):
    """Run inference on all images in a folder."""
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not files:
        print(f"  No images found in {folder_path}")
        return

    print(f"\n  Found {len(files)} images in {folder_path}")
    for i, f in enumerate(sorted(files)):
        print(f"\n  [{i+1}/{len(files)}]", end="")
        run_single(ser, f)


def main():
    parser = argparse.ArgumentParser(description="ESP32 Serial Inference Client")
    parser.add_argument("--port", required=True, help="Serial port (e.g. COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--folder", help="Folder of images to process")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if not args.image and not args.folder and not args.interactive:
        print("Specify --image, --folder, or --interactive")
        parser.print_help()
        sys.exit(1)

    print(f"Connecting to ESP32 on {args.port} at {args.baud} baud...")
    print(f"  Port: {args.port} @ {args.baud} baud")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=2)
        time.sleep(2)  # Wait for ESP32 boot

        # Drain boot messages
        while ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(f"  ESP32: {line}")

        print(f"  Connected!\n")

        if args.image:
            run_single(ser, args.image)
        elif args.folder:
            run_folder(ser, args.folder)
        elif args.interactive:
            run_interactive(ser)

        ser.close()
        print("\n  Serial closed.")

    except serial.SerialException as e:
        print(f"  Serial error: {e}")
        print(f"  Make sure:")
        print(f"    1. ESP32 is connected")
        print(f"    2. Port {args.port} is correct")
        print(f"    3. Arduino IDE Serial Monitor is closed")
        sys.exit(1)

    print("")


if __name__ == "__main__":
    main()
