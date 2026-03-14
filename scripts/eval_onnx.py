import os
import pickle
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FP32_MODEL = os.path.join(PROJECT_ROOT, "models", "model.onnx")
INT8_MODEL = os.path.join(PROJECT_ROOT, "models", "model_int8.onnx")

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)

def load_cifar10_test():
    cifar_dir = os.path.join(PROJECT_ROOT, "data", "cifar-10-batches-py")
    test_batch = os.path.join(cifar_dir, "test_batch")

    if not os.path.exists(test_batch):
        print(f"Test batch not found at {test_batch}.")
        print("Please run quantize_int8.py first to download the dataset.")
        return None, None

    with open(test_batch, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    
    images = batch[b"data"]
    labels = np.array(batch[b"labels"])

    # Reshape and normalize
    images = images.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    images = (images - MEAN) / STD

    return images, labels

def evaluate_model(model_path, images, labels):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return 0.0

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    correct = 0
    total = len(images)
    batch_size = 1

    print(f"Evaluating {os.path.basename(model_path)}...")
    for i in range(0, total, batch_size):
        batch_images = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        outputs = session.run(None, {input_name: batch_images})[0]
        preds = np.argmax(outputs, axis=1)
        correct += np.sum(preds == batch_labels)
        
        if (i + batch_size) % 1000 == 0:
            print(f"  Progress: {i + batch_size}/{total}")

    accuracy = correct / total * 100.0
    print(f"Accuracy: {accuracy:.2f}%\n")
    return accuracy

def main():
    print("Loading CIFAR-10 Test Set (10,000 images)...")
    images, labels = load_cifar10_test()
    if images is None:
        return
    
    fp32_acc = evaluate_model(FP32_MODEL, images, labels)
    int8_acc = evaluate_model(INT8_MODEL, images, labels)

    print("=== Evaluation Summary ===")
    print(f"FP32 Model Accuracy: {fp32_acc:.2f}%")
    print(f"INT8 Model Accuracy: {int8_acc:.2f}%")
    if fp32_acc > 0:
        drop = fp32_acc - int8_acc
        print(f"Accuracy Drop:       {drop:.2f}%")

if __name__ == "__main__":
    main()
