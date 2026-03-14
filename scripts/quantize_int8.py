"""
Quantize model.onnx to INT8 using onnxruntime static quantization
with CIFAR-10 calibration data.

Output: models/model_int8.onnx
"""
import os
import numpy as np
import onnxruntime
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)
import onnx
from onnxruntime.quantization.shape_inference import quant_pre_process


# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_MODEL = os.path.join(PROJECT_ROOT, "models", "model.onnx")
PREPROCESSED_MODEL = os.path.join(PROJECT_ROOT, "models", "model_preprocessed.onnx")
OUTPUT_MODEL = os.path.join(PROJECT_ROOT, "models", "model_int8.onnx")
NUM_CALIBRATION_SAMPLES = 200  # number of images for calibration

# CIFAR-10 normalization (same as train.py)
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)


def generate_calibration_data(num_samples):
    """Generate random calibration data matching CIFAR-10 distribution.
    
    Uses random noise normalized with CIFAR-10 stats.
    For best accuracy, replace with real CIFAR-10 images.
    """
    data = []
    for _ in range(num_samples):
        # Random image in [0, 1], then normalize like CIFAR-10
        img = np.random.rand(1, 3, 32, 32).astype(np.float32)
        img = (img - MEAN) / STD
        data.append(img)
    return data


def load_real_cifar10(num_samples):
    """Download and load real CIFAR-10 data using only stdlib + numpy."""
    import urllib.request
    import tarfile
    import pickle

    cifar_dir = os.path.join(PROJECT_ROOT, "data", "cifar-10-batches-py")
    tar_path = os.path.join(PROJECT_ROOT, "data", "cifar-10-python.tar.gz")
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    # Download if needed
    if not os.path.isdir(cifar_dir):
        os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
        print(f"  Downloading CIFAR-10 from {url} ...")
        urllib.request.urlretrieve(url, tar_path)
        print(f"  Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(os.path.join(PROJECT_ROOT, "data"))
        print(f"  Done.")
    else:
        print(f"  CIFAR-10 already exists at {cifar_dir}")

    # Load training batches (data_batch_1 .. data_batch_5)
    all_images = []
    for i in range(1, 6):
        batch_file = os.path.join(cifar_dir, f"data_batch_{i}")
        with open(batch_file, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        # batch[b'data'] is (10000, 3072) uint8
        all_images.append(batch[b"data"])
    
    all_images = np.concatenate(all_images, axis=0)  # (50000, 3072)

    # Shuffle and pick num_samples
    indices = np.random.permutation(len(all_images))[:num_samples]
    selected = all_images[indices]  # (num_samples, 3072)

    # Reshape to NCHW: (N, 3, 32, 32), normalize to float32
    images = selected.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    # Apply CIFAR-10 normalization (same as train.py)
    images = (images - MEAN) / STD

    # Split into individual samples (batch_size=1 each)
    data = [images[i:i+1] for i in range(len(images))]
    print(f"  Loaded {len(data)} real CIFAR-10 images for calibration")
    return data


class CifarCalibrationReader(CalibrationDataReader):
    """Feeds calibration data to the quantizer."""

    def __init__(self, calibration_data, input_name):
        self.data = calibration_data
        self.input_name = input_name
        self.index = 0

    def get_next(self):
        if self.index >= len(self.data):
            return None
        sample = {self.input_name: self.data[self.index]}
        self.index += 1
        return sample

    def rewind(self):
        self.index = 0


def main():
    print("Pre-processing model...")
    quant_pre_process(INPUT_MODEL, PREPROCESSED_MODEL)

    print("Loading model to get input info...")
    session = onnxruntime.InferenceSession(PREPROCESSED_MODEL)
    input_name = session.get_inputs()[0].name

    print(f"Preparing calibration data ({NUM_CALIBRATION_SAMPLES} samples)...")
    cal_data = load_real_cifar10(NUM_CALIBRATION_SAMPLES)
    reader = CifarCalibrationReader(cal_data, input_name)

    print("Quantizing to INT8...")
    quantize_static(
        model_input=PREPROCESSED_MODEL,
        model_output=OUTPUT_MODEL,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=True,
    )

    # Report sizes
    orig_size = os.path.getsize(INPUT_MODEL) / 1024
    quant_size = os.path.getsize(OUTPUT_MODEL) / 1024
    ratio = quant_size / orig_size * 100

    print("\nQuantization complete.")
    print(f"Original model:  {orig_size:.1f} KB")
    print(f"Quantized model: {quant_size:.1f} KB")
    print(f"Output saved to: {OUTPUT_MODEL}")

    # Cleanup preprocessed
    if os.path.exists(PREPROCESSED_MODEL):
        os.remove(PREPROCESSED_MODEL)
        print(f"  Cleaned up {PREPROCESSED_MODEL}")


if __name__ == "__main__":
    main()
