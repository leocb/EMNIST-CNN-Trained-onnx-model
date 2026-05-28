# Handwritten Number Digit Recognition

This is my try at training a CNN Model based on the [EMNIST (Extented MNIST) Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

## 99.63% Accuracy
- Trained on 240000 data points
- Validated on 40000 data points

## Use / Download:
Just save the raw `emnist_cnn.onnx` file and load that and use for inference
[Or click here to download it](https://raw.githubusercontent.com/leocb/EMNIST-CNN-Trained-onnx-model/refs/heads/main/emnist_cnn.onnx)

## Model details:
- Input: 28x28x1 grayscale image (pixel values normalized to [0,1])
- Output: 10 probabilities (digits 0-9, softmax)
- Architecture: CNN with ~20K parameters
  - Block 1: Conv(16) -> BN -> ReLU -> Conv(16) -> BN -> ReLU -> MaxPool -> Dropout
  - Block 2: Conv(32) -> BN -> ReLU -> Conv(32) -> BN -> ReLU -> MaxPool -> Dropout
  - Block 3: Conv(16, 1x1 bottleneck) -> BN -> ReLU -> Conv(16) -> BN -> ReLU
  - Global Average Pooling -> Dense(10, Softmax)
- Framework: Keras (TensorFlow 2.16) with Metal GPU acceleration on Apple Silicon
- Training samples: 240,000
- Validation samples: 40,000

## Training

Open and run [emnist_cnn.ipynb](emnist_cnn.ipynb) to train from scratch:

```bash
jupyter notebook emnist_cnn.ipynb
```

The notebook:
1. Loads the EMNIST Digits dataset from the `training/` folder (IDX format, gzipped)
2. Corrects the EMNIST orientation (images stored rotated 90° CW + mirrored; fixed via transpose)
3. Normalizes and preprocesses the data
4. Builds a CNN inspired by [aviban15/cnn-mnist](https://github.com/aviban15/cnn-mnist)
5. Trains with SGD + momentum, early stopping, and learning rate scheduling
6. Evaluates on the test set
7. Exports to ONNX via `tf2onnx`

## Inference with ONNX

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

session = ort.InferenceSession("emnist_cnn.onnx")
input_name = session.get_inputs()[0].name

def predict_digit(image_path):
    img = Image.open(image_path).convert('L').resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    output = session.run(None, {input_name: img_array})[0][0]
    return int(np.argmax(output)), output.tolist()

# Returns (predicted_digit, [prob_0, prob_1, ..., prob_9])
```

## Files
- `emnist_cnn.ipynb` - Training notebook
- `emnist_cnn.onnx`  - Trained model in ONNX format
- `emnist_cnn.keras` - Trained model in Keras format
- `training/` - EMNIST Digits dataset (gzipped IDX files)
