# Home Assignment 2 - Neural Networks and Deep Learning
**Course:** CS5720 - Neural Networks and Deep Learning  
**University:** University of Central Missouri  
**Student:** Anvesh Chakali (700758947)  
**Spring 2025**


## Overview  
This repository contains the implementation and demonstration of **Home Assignment 2**, covering the following topics:  
- **Cloud Computing for Deep Learning** (Elasticity & Scalability, AWS SageMaker vs. Google Vertex AI vs. Azure ML)  
- **Convolution Operations with Different Parameters**  
- **CNN Feature Extraction (Sobel Filters & Pooling)**  
- **CNN Architectures (AlexNet & ResNet)**  

Each section includes Python scripts demonstrating the concepts using **TensorFlow, OpenCV, and NumPy**.


## **1️ Cloud Computing for Deep Learning**
###  **Elasticity vs. Scalability**
- **Elasticity**: The ability of a cloud system to dynamically allocate resources based on demand. In deep learning, this helps in scaling up GPUs/TPUs when needed and reducing resources when idle.  
- **Scalability**: The system's ability to handle an increasing workload by adding more resources. This ensures efficient training of deep learning models on large datasets.

###  **Comparison of Cloud Platforms for Deep Learning**
| Feature | AWS SageMaker | Google Vertex AI | Azure ML Studio |
|---------|-------------|----------------|-----------------|
| **Ease of Use** | Jupyter notebooks & automated ML | AutoML & pipelines | No-code UI & ML pipelines |
| **Compute Support** | GPUs, AWS Inferentia | GPUs, TPUs | NVIDIA GPUs, FPGAs |
| **AutoML** | Built-in AutoML | Strong AutoML support | Drag-and-drop AutoML |
| **Model Deployment** | One-click inference endpoints | AI pipelines | MLOps & batch inference |
| **Best For** | AWS users | Large-scale AI with TPUs | Microsoft enterprise users |

 **Conclusion**: AWS is best for AWS users, Vertex AI for large-scale AI workloads, and Azure ML for enterprises using Microsoft services.

---

## **2️ Convolution Operations with Different Parameters**
### ** Code Overview**
This section demonstrates convolution operations on a **5×5 matrix** using a **3×3 kernel**, varying stride and padding.

#### **Python Implementation**
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

# Define input matrix
input_matrix = np.array([[1, 2, 3, 4, 5],
                         [6, 7, 8, 9, 10],
                         [11, 12, 13, 14, 15],
                         [16, 17, 18, 19, 20],
                         [21, 22, 23, 24, 25]], dtype=np.float32).reshape(1, 5, 5, 1)

# Define convolution kernel
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]], dtype=np.float32).reshape(3, 3, 1, 1)

# Apply convolution function
def apply_convolution(stride, padding):
    input_layer = Input(shape=(5, 5, 1))
    conv_layer = Conv2D(filters=1, kernel_size=(3, 3), strides=stride, padding=padding,
                         kernel_initializer=tf.keras.initializers.Constant(kernel), use_bias=False)(input_layer)
    model = Model(inputs=input_layer, outputs=conv_layer)
    output = model.predict(input_matrix)
    return output.squeeze()

# Run different convolutions
output_valid = apply_convolution(1, 'valid')
output_same = apply_convolution(1, 'same')

print("Output (Stride=1, Padding='VALID'):\n", output_valid)
print("\nOutput (Stride=1, Padding='SAME'):\n", output_same)
```
####  **Results & Observations**
- **Padding = 'VALID'** removes the border, producing a smaller output.  
- **Padding = 'SAME'** maintains the input size by adding zero-padding.  
- Increasing **stride** reduces the output size.

---

## **3️ CNN Feature Extraction (Sobel Filters & Pooling)**
###  **Edge Detection Using Sobel Filter**
- The **Sobel filter** detects edges by calculating gradients in the **X** and **Y** directions.
- Implemented using OpenCV and Matplotlib.

####  **Python Code**
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_sobel_filter(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); plt.imshow(image, cmap="gray"); plt.title("Original")
    plt.subplot(1, 3, 2); plt.imshow(sobel_x, cmap="gray"); plt.title("Sobel-X")
    plt.subplot(1, 3, 3); plt.imshow(sobel_y, cmap="gray"); plt.title("Sobel-Y")
    plt.show()

# Usage: apply_sobel_filter('image.jpg')

📌 **Conclusion**: Sobel-X detects **vertical edges**, while Sobel-Y detects **horizontal edges**.

### 🔹 **Max Pooling vs. Average Pooling**
- **Max Pooling** retains the highest pixel value.
- **Average Pooling** computes the mean pixel value.
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D

input_matrix = np.random.randint(0, 256, (1, 4, 4, 1), dtype=np.int32).astype(np.float32)
max_pool = MaxPooling2D(pool_size=(2, 2), strides=2)(tf.convert_to_tensor(input_matrix))
avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2)(tf.convert_to_tensor(input_matrix))

print("\nMax Pooled Matrix:\n", max_pool.numpy().squeeze())
print("\nAverage Pooled Matrix:\n", avg_pool.numpy().squeeze())
```
📌 **Conclusion**: Max pooling is better for detecting edges, while average pooling smooths images.



## **4️ CNN Architectures (AlexNet & ResNet)**
### **AlexNet**
- Introduced deep CNNs with **ReLU activations**, **dropout**, and **overlapping pooling**.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def alexnet():
    model = Sequential([
        Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)),
        MaxPooling2D((3, 3), strides=2),
        Conv2D(256, (5, 5), activation='relu', padding='same'),
        MaxPooling2D((3, 3), strides=2),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.summary()

alexnet()
```
📌 **Conclusion**: AlexNet improved accuracy in ImageNet classification.

### 🔹 **ResNet (Residual Networks)**
- Introduced **skip connections** to prevent the vanishing gradient problem.

from tensorflow.keras.layers import Input, Add

def resnet():
    input_layer = Input(shape=(224, 224, 3))
    x = Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(input_layer)
    x = Add()([x, input_layer])  # Skip Connection
    model = Model(inputs=input_layer, outputs=x)
    model.summary()

resnet()

📌 **Conclusion**: ResNet allows deeper networks by avoiding gradient vanishing.



📌 **How to Run the Code**
1. Install dependencies: `pip install tensorflow numpy opencv-python matplotlib`
2. Run scripts in **Google Colab or Jupyter Notebook**.

**Thank you for checking out my assignment!**


Let me know if you need any modifications! 🚀
