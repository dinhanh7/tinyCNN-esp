# TinyCNN-ESP: TinyMobileNet CIFAR-10 on ESP32

INT8 quantized MobileNet inference engine for ESP32, written in pure C.

## Project Structure

```
tinyCNN-esp/
├── src/                    # Training & conversion (PyTorch)
│   ├── train.py            #   Train TinyMobileNet on CIFAR-10
│   └── requirements.txt    #   Python dependencies
│
├── models/                 # Trained model files
│   ├── tiny_mobilenet_cifar10.pth   # PyTorch weights
│   ├── model.onnx          #   ONNX FP32 model
│   ├── model_int8.onnx     #   ONNX INT8 quantized
│   └── model_quant.tflite  #   TFLite quantized
│
├── scripts/                # Utility scripts
│   ├── quantize_int8.py    #   Quantize ONNX FP32 -> INT8
│   ├── generate_c.py       #   Generate C inference engine from INT8 ONNX
│   ├── convert_image.py    #   Convert any image -> input_image.h
│   ├── serial_inference.py #   CLI for sending images over Serial
│   └── gui_inference.py    #   Tkinter GUI for sending images over Serial
│
├── esp32_inference/        # PlatformIO ESP32 project
│   ├── platformio.ini      #   PlatformIO config
│   ├── src/                #   Source code
│   │   ├── main.cpp        #   Dual-mode serial/standalone inference
│   │   └── model.c         #   Pure-C inference engine
│   └── include/            #   Header files
│       ├── model.h         #   Model defines & prototypes
│       ├── model_weights.h #   All INT8 weights (PROGMEM)
│       └── input_image.h   #   Test image (quantized int8)
│
├── image/                  # Test images
│   └── image.png           #   Sample image for testing
│
└── data/                   # CIFAR-10 dataset (auto-downloaded)
```

## Quick Start (ESP32)

We use **PlatformIO**, replacing the Arduino IDE entirely. **You can now compile, flash, and test the model entirely from the GUI!**

### 1. Install Dependencies
```bash
pip install -r src/requirements.txt
```

*(Ubuntu/Linux users only)* - You may need to install Tkinter and grant Serial permissions:
```bash
# Install Tkinter for the GUI
sudo apt update && sudo apt install -y python3-tk

# Grant your user permission to read/write Serial ports
sudo usermod -a -G dialout $USER
sudo usermod -a -G plugdev $USER

# Install PlatformIO udev rules for board recognition
curl -fsSL https://raw.githubusercontent.com/platformio/platformio-core/develop/platformio/assets/system/99-platformio-udev.rules | sudo tee /etc/udev/rules.d/99-platformio-udev.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```
*(Note: Restart your PC or log out/in for `dialout` permissions to take effect).*

### 2. Launch the GUI (All-in-one Toolkit)
Run the following script. It works on both **Windows** and **Ubuntu**.
```bash
python scripts/gui_inference.py
```

### 3. Flash & Run
1. Select your COM Port (e.g., `COM3` on Windows or `/dev/ttyUSB0` on Ubuntu).
2. Click **Flash ESP32** directly in the GUI to compile the C inference engine and upload it to your board.
3. Click **Connect**, browse for an image (e.g., `image/image.png`), and click **Run Inference**!

---

## Alternative: Command Line Interface (CLI)

If you prefer using the terminal instead of the GUI:

1. Build and upload to ESP32 manually:
   ```bash
   cd esp32_inference
   pio run --target upload
   ```
2. Test using the interactive Serial Inference script:
   ```bash
   cd ..
   python scripts/serial_inference.py --port COM3 --interactive  # Use /dev/ttyUSB0 on Linux
   ```

## Scripts Usage

All scripts auto-resolve paths — run from anywhere:

```bash
# INTERACTIVE GUI (Recommended for testing multiple images vs ESP32)
python scripts/gui_inference.py

# CLI Serial Inference (Terminal equivalent)
python scripts/serial_inference.py --port COM3 --interactive

# Re-quantize model (if retraining)
python scripts/quantize_int8.py

# Re-generate C code (if re-quantizing)
python scripts/generate_c.py
```

## Model Info

| Property | Value |
|----------|-------|
| Architecture | TinyMobileNet (MBConv) |
| Dataset | CIFAR-10 (10 classes) |
| Input | 3×32×32 (RGB) |
| Params | 22,306 |
| FP32 model | 93 KB |
| INT8 model | 37 KB |
| CIFAR-10 Test Set | 10,000 images |
| FP32 Accuracy (CIFAR-10 Test) | 78.53% |
| INT8 Accuracy (CIFAR-10 Test) | 78.37% (Drop: 0.16%) |
| ESP32 RAM | ~80 KB static buffers |
