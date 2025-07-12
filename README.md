# 👗 Fashion MNIST Classification with Custom CNN

A deep learning project that classifies fashion items using a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras. This project achieves **89% accuracy** on the Fashion MNIST dataset.

## 🎯 Project Overview

This project implements a custom CNN architecture to classify fashion items from the Fashion MNIST dataset. The model can distinguish between 10 different types of clothing and accessories with high accuracy.

## 📊 Dataset Information

**Fashion MNIST** is a dataset of clothing images that serves as a more challenging replacement for the classic MNIST digit dataset.

### Dataset Details:
- **Training samples**: 60,000 images
- **Test samples**: 10,000 images
- **Image dimensions**: 28×28 pixels (grayscale)
- **Classes**: 10 fashion categories

### Fashion Categories:
| Label | Description |
|-------|-------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## 🏗️ Model Architecture

The CNN model uses a custom architecture optimized for fashion item classification:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(6, kernel_size=(5,5), activation='relu', padding='same', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Conv2D(16, kernel_size=(5,5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
```

### Architecture Highlights:
- **Convolutional Layers**: 2 Conv2D layers with ReLU activation
- **Batch Normalization**: Improves training stability and convergence
- **Max Pooling**: Reduces spatial dimensions and computational load
- **Dropout**: Prevents overfitting with 20% dropout rate
- **Dense Layers**: Fully connected layers for final classification

## 📈 Model Performance

- **Final Accuracy**: 90%
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam

## Data Visualization

The project includes comprehensive data visualization:

### Training Data Exploration
- Sample images from each fashion category
- Class distribution analysis
- Data preprocessing visualization

### Training Progress
- **Loss curves**: Training and validation loss over epochs
- **Accuracy curves**: Training and validation accuracy progression
- **Performance metrics**: Detailed classification report

```

## Key Features

- **Custom CNN Architecture**: Tailored for fashion item classification
- **Batch Normalization**: Enhanced training stability
- **Dropout Regularization**: Prevents overfitting
- **Data Visualization**: Comprehensive plots and analysis
- **High Accuracy**: 89% classification accuracy
- **Modular Code**: Clean and well-documented implementation

## Results Summary

| Metric | Value |
|--------|-------|
| Test Accuracy | 89% |
| Model Parameters | ~85K |
| Training Time | ~10 minutes |
| Input Shape | (28, 28, 1) |
| Output Classes | 10 |

## Technical Details

### Model Compilation
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Training Configuration
- **Batch Size**: 128
- **Epochs**: 20-30 (with early stopping)
- **Validation Split**: 20% of training data
- **Data Augmentation**: Optional for improved generalization
