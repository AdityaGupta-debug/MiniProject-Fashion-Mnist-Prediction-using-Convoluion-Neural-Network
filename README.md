# 👗 Fashion MNIST Classification with Custom CNN

A deep learning project that classifies fashion items using a custom 🧠 Convolutional Neural Network (CNN) built with TensorFlow/Keras.  
✅ This model achieves **~90% accuracy** on the Fashion MNIST dataset.

---

## 🎯 Project Overview

This project implements a custom CNN to classify fashion items from the **Fashion MNIST** dataset.  
The model distinguishes between **10 clothing/accessory categories** with high accuracy and efficiency.

---

## 📊 Dataset Information

**Fashion MNIST** is a dataset of grayscale clothing images — a modern, more complex alternative to handwritten digits.

### 🔢 Dataset Details:
- 🧵 **Training samples**: 60,000  
- 🧪 **Test samples**: 10,000  
- 🖼️ **Image dimensions**: 28×28 (grayscale)  
- 🏷️ **Classes**: 10 fashion categories

### 🧾 Fashion Categories:
| Label | Description   |
|-------|---------------|
| 0     | T-shirt/top   |
| 1     | Trouser       |
| 2     | Pullover      |
| 3     | Dress         |
| 4     | Coat          |
| 5     | Sandal        |
| 6     | Shirt         |
| 7     | Sneaker       |
| 8     | Bag           |
| 9     | Ankle boot    |

---

## 🏗️ Model Architecture

The CNN model is designed with performance and simplicity in mind. Here's the architecture:

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

## 🌟 Key Features

- 🧠 **Custom CNN Architecture**: Tailored specifically for fashion item classification  
- 🧪 **Batch Normalization**: Improves training speed and stability  
- 🛡️ **Dropout Regularization**: Reduces overfitting by randomly dropping neurons  
- 📊 **Data Visualization**: Training curves, sample outputs, and insights  
- ✅ **High Accuracy**: Achieves ~89% test accuracy  
- 🧱 **Modular Code**: Clean, reusable, and well-commented implementation  

---

## 📌 Results Summary

| 📈 Metric            | 🔢 Value       |
|----------------------|----------------|
| ✅ Test Accuracy      | **90%**        |
| 📦 Model Parameters   | ~85K           |
| ⏱️ Training Time      | ~6 minutes    |
| 🖼️ Input Shape        | (28, 28, 1)    |
| 🏷️ Output Classes     | 10             |

