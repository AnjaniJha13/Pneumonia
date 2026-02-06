# Pneumonia Detection from Chest X-Rays

A deep learning application that detects pneumonia from chest X-ray images using MobileNetV2 and provides a Flask REST API for predictions.

## Features

- **Deep Learning Model**: Transfer learning with MobileNetV2 pretrained on ImageNet
- **REST API**: Flask-based API for easy integration
- **Data Augmentation**: Improves model generalization
- **Binary Classification**: Detects NORMAL vs PNEUMONIA cases

## Model Architecture

- Base Model: MobileNetV2 (pretrained on ImageNet)
- Additional Layers:
  - Global Average Pooling
  - Dropout (0.3)
  - Dense layer with sigmoid activation
- Input Size: 224x224x3
- Output: Binary classification (NORMAL/PNEUMONIA)

## Training Performance

- Training Accuracy: ~90%
- Validation Accuracy: 75%
- Epochs: 8
- Batch Size: 32
- Optimizer: Adam (learning rate: 0.0001)

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Epicb0i/pneumonia-xray-detection.git
cd pneumonia-xray-detection
```

2. Install dependencies:
```bash
pip install tensorflow flask numpy pillow
```

3. Download the Chest X-Ray dataset and place it in the `dataset/` directory with the following structure:
```
dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Usage

### Training the Model

```bash
python train.py
```

This will train the model and save it as `pneumonia_model.h5`.

### Running the API Server

```bash
python app.py
```

The API will be available at `http://127.0.0.1:5000`

### API Endpoints

#### Home
```
GET /
```
Returns: "Pneumonia Detection API is running"

#### Predict
```
POST /predict
Content-Type: multipart/form-data
```

**Parameters:**
- `file`: Chest X-ray image file (JPEG/PNG)

**Response:**
```json
{
  "prediction": "PNEUMONIA" | "NORMAL",
  "confidence": 85.42
}
```

**Example using curl:**
```bash
curl -X POST -F "file=@xray_image.jpg" http://127.0.0.1:5000/predict
```

## Project Structure

```
.
├── app.py              # Flask API server
├── train.py            # Model training script
├── dataset/            # Training data (not included)
├── pneumonia_model.h5  # Trained model (generated)
└── README.md           # This file
```

## Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

## Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **MobileNetV2**: Efficient CNN architecture
- **Flask**: Web framework for API
- **NumPy**: Numerical computing
- **Pillow**: Image processing

## License

MIT License

## Author

**Epicb0i**
- Email: gauravchopade913@gmail.com
- GitHub: [@Epicb0i](https://github.com/Epicb0i)

## Acknowledgments

- Dataset: Kermany, D. et al. (2018)
- MobileNetV2: Sandler, M. et al. (2018)
