# MNIST Handwritten Digit Recognition

A Digital Image Processing project for recognizing handwritten digits **0–9** using a Convolutional Neural Network (CNN) trained on the MNIST dataset, with an OpenCV pipeline for recognizing multiple digits from custom images.

## Project Highlights

- MNIST dataset loading and preprocessing
- CNN-based handwritten digit classification
- Training, validation, and test evaluation
- Training/validation loss and accuracy curves
- Model saving and reloading
- Custom-image digit recognition
- OpenCV contour-based digit detection
- Multi-digit recognition in left-to-right order
- Accuracy, precision, recall, F1-score and confusion-matrix analysis

The project report records approximately **99.1% test accuracy**, with results varying slightly between runs. 

## How It Works

### MNIST Classification

```text
MNIST Images (28×28 grayscale)
              │
              ▼
       Normalize /255
              │
              ▼
       Reshape (28×28×1)
              │
              ▼
        Conv2D 32
              │
        MaxPooling
              │
        Conv2D 64
              │
        Conv2D 64
              │
        MaxPooling
              │
           Flatten
              │
        Dense 100
              │
    Batch Normalization
              │
        Dense 10
              │
           Softmax
              │
              ▼
        Digit 0–9
```

### Custom Image Recognition

```text
Input Image
    ↓
Median Blur
    ↓
Grayscale
    ↓
Adaptive Threshold
    ↓
Contour Detection
    ↓
Sort Digits Left → Right
    ↓
Crop Each Digit
    ↓
Resize 18×18
    ↓
Pad to 28×28
    ↓
CNN Prediction
    ↓
Recognized Number
```

## Model Architecture

The documented CNN uses:

| Layer | Configuration |
|---|---|
| Input | 28 × 28 × 1 grayscale |
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | 1024 features |
| Dense | 100 units, ReLU |
| Batch Normalization | Default configuration |
| Output Dense | 10 units, Softmax |

## Dataset

MNIST contains:

- **70,000** handwritten digit images
- **60,000** training images
- **10,000** original test images
- **28×28** grayscale images
- **10 classes**: digits 0 through 9

The project splits the original 10,000-image test portion into:

- 5,000 validation images
- 5,000 final test images

The MNIST dataset is downloaded/cached by Keras/TensorFlow when required, so it should **not** be committed to this repository.

## Preprocessing

### Normalization

```python
x_train_norm = x_train / 255.
```

Pixel values are converted from 0–255 to 0.0–1.0.

### One-Hot Encoding

```python
y_train_enc = to_categorical(y_train)
```

Labels are converted to categorical vectors for multi-class classification.

### Image Shape

```text
(N, 28, 28) → (N, 28, 28, 1)
```

The final dimension represents the single grayscale channel.

## Training Configuration

| Parameter | Value |
|---|---|
| Loss | Categorical Cross-Entropy |
| Optimizer | SGD with Momentum |
| Learning Rate | 0.01 |
| Momentum | 0.9 |
| Batch Size | 64 |
| Epochs | 15 |
| Metric | Accuracy |
| Training Samples | 60,000 |
| Validation Samples | 5,000 |
| Test Samples | 5,000 |

## Reported Results

The project report gives the following representative results:

| Metric | Training | Validation | Test |
|---|---:|---:|---:|
| Accuracy | ~99.5% | ~99.2% | ~99.1% |
| Loss | ~0.016 | ~0.026 | ~0.028 |

The report states that these values can vary slightly between runs because of random initialization and mini-batch sampling, while test accuracy consistently remains above 99%.

## OpenCV Digit Recognition

The custom-image pipeline uses:

- `cv2.imread()` to load an image
- `cv2.medianBlur()` for noise reduction
- `cv2.cvtColor()` for grayscale conversion
- `cv2.adaptiveThreshold()` for binarization
- `cv2.findContours()` for digit localization
- x-coordinate sorting for left-to-right ordering
- resizing and padding to reproduce the 28×28 MNIST input format
- the trained CNN for inference

The adaptive threshold uses a local neighborhood to handle uneven lighting. Detected digits are resized to 18×18 and padded by 5 pixels on each side to produce 28×28 inputs.

## Project Structure

```text
MNIST-Digit-Recognition/
│
├── .github/
│   └── copilot-instructions.md
│
├── train.py
├── predict.py
├── utils.py
├── visualize.py              # if present in the project
│
├── requirements.txt
├── README.md
│
└── models/                   # generated/saved model files
```

> Do not commit the local `.venv/` directory, Python cache files, or generated model/checkpoint files unless you intentionally want to distribute them.

## Installation

### Prerequisites

- Python 3.8+
- pip

### Windows

```powershell
python -m venv .venv
.venv\Scriptsctivate
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Train the model:

```bash
python train.py
```

Run prediction:

```bash
python predict.py
```

If the current source files expose additional command-line arguments, use the options documented in those scripts.

## Dependencies

The supplied dependency file specifies:

- TensorFlow >= 2.12
- NumPy >= 1.21
- pandas >= 1.3
- Matplotlib >= 3.4
- scikit-learn >= 0.24
- Pillow >= 8.3

The report additionally documents OpenCV and ipywidgets as project libraries. If the current Python implementation imports either package, add them to `requirements.txt` before pushing.

## Model Persistence

The report documents saving the trained model using Keras native model serialization:

```python
model.save("my_model.keras")
```

and reloading it with:

```python
tf.keras.models.load_model("my_model.keras")
```

Generated model files should normally remain outside Git unless you want the repository to distribute a trained model.

## Limitations

- The OpenCV pipeline can struggle with overlapping or touching digits.
- A model trained only on MNIST may not generalize to unusual handwriting.
- No data augmentation was used in the documented training.
- Fixed contour detection can fail with complex backgrounds or low contrast.
- Real-world handwriting is substantially harder than the clean MNIST benchmark.

## Future Improvements

Possible improvements include:

- Data augmentation
- Dropout
- Adam optimizer
- Learning-rate scheduling
- Transfer learning
- YOLO-based digit detection
- Larger and more diverse real-world datasets

## Applications

Potential applications include:

- Postal-code recognition
- Cheque processing
- Form digitization
- Number recognition from photographs
- Accessibility tools
- Automated document processing

## References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. — *Gradient-Based Learning Applied to Document Recognition*
- Goodfellow, Bengio & Courville — *Deep Learning*
- TensorFlow / Keras documentation
- OpenCV documentation
- scikit-learn documentation
- MNIST dataset

## Author

**Chethan J D**  
Roll No: `20231ECE0174`  
Department of Electronics & Communication Engineering

This repository contains an academic Digital Image Processing project demonstrating CNN-based handwritten digit recognition and practical image-processing inference.
