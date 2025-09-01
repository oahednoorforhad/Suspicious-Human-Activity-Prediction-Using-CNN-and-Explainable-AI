# Suspicious Human Activity Detection using CNN and Explainable AI (XAI) 🕵️‍♂️

This repository contains the code and resources for the paper: "Suspicious Human Activity Detection Using CNN and Explainable AI (XAI)". The project presents a robust deep learning framework to automatically classify human activities in surveillance images as either 'Suspicious' or 'Neutral'.

The core of this project is a high-accuracy Convolutional Neural Network (CNN) that not only makes predictions but also provides visual explanations for its decisions using Grad-CAM, making the model transparent and trustworthy.


## 📜 Project Overview

In an era of widespread CCTV surveillance, manual monitoring is impractical and prone to human error due to fatigue and attention lapses. This project addresses this challenge by creating an automated system that acts as a first-line filter, flagging potentially suspicious events for human operators. Unlike traditional computer vision methods that struggle with real-world complexities like dynamic lighting, our CNN learns hierarchical features directly from data, enabling nuanced and accurate classification.

### ✨ Key Features

* **High-Accuracy Detection:** A custom-built CNN model that achieves **96.9% accuracy** on the test set.
* **Binary Classification:** Effectively distinguishes between benign human presence ('Human Only Images') and potentially malicious activities ('Suspicious').
* **Explainable AI (XAI):** Implements **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate heatmaps that highlight the specific image regions influencing the model's decision, enhancing trust and interpretability.
* **Robust Training:** Utilizes extensive data augmentation (rotation, flipping, zooming) to improve the model's generalization and prevent overfitting.
* **High Precision:** Achieved a **precision score of 0.9804** for the 'Suspicious' class, indicating a very low rate of false alarms.

## 🧠 Model Architecture

The model is a Convolutional Neural Network designed for binary classification of $128 \times 128$ RGB images.

The architecture consists of four main convolutional blocks followed by fully connected dense layers:
1.  **Conv Block 1:** 32 filters ($3 \times 3$), Batch Normalization, Max Pooling.
2.  **Conv Block 2:** 64 filters ($3 \times 3$), Batch Normalization, Max Pooling.
3.  **Conv Block 3:** Two Conv layers with 128 filters ($3 \times 3$), Batch Normalization, Max Pooling.
4.  **Conv Block 4:** Two Conv layers with 256 filters ($3 \times 3$), Batch Normalization, Max Pooling.
5.  **Fully Connected Head:** The feature map is flattened and passed through two dense layers (1024 and 512 neurons) with ReLU activation and 50% Dropout to prevent overfitting.
6.  **Output Layer:** A final dense layer with a **Sigmoid** activation function outputs a probability score between 0 and 1 for the classification.

The model is trained using the **Adam optimizer** with a learning rate of $10^{-4}$ and **binary cross-entropy** as the loss function.

## 📊 Performance

The model was evaluated on a test set of 735 images and demonstrated excellent performance across multiple metrics.

| Metric              | Value    |
| ------------------- | -------- |
| **Accuracy** | 0.969    |
| **ROC-AUC** | 0.9822   |
| **Precision-AUC** | 0.9835   |

### Per-Class Performance

| Class               | Precision | Recall | F1-Score |
| ------------------- | --------- | ------ | -------- |
| **Human Only Images** | 0.9138    | 0.9849 | 0.948    |
| **Suspicious** | 0.9804    | 0.8902 | 0.9331   |
| **Weighted Avg** | 0.9443    | 0.9415 | 0.9412   |

### Confusion Matrix

The confusion matrix below visualizes the model's predictions. Out of 735 test images, it correctly classified 392 'Human Only' images and 300 'Suspicious' images.

* **True Positives (Suspicious):** 300
* **True Negatives (Human Only):** 392
* **False Positives:** 6
* **False Negatives:** 37


## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* TensorFlow 2.x
* NumPy
* Matplotlib
* OpenCV

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Suspicious-Activity-Detection-CNN-XAI.git](https://github.com/your-username/Suspicious-Activity-Detection-CNN-XAI.git)
    cd Suspicious-Activity-Detection-CNN-XAI
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download the Dataset:**
    The model was trained on the "Improved Thief Detection Dataset" sourced from Kaggle. Please download and structure it as described in the paper.

### Usage

1.  **Train the model:**
    ```bash
    python train.py --dataset path/to/your/dataset
    ```
2.  **Run inference on an image:**
    ```bash
    python predict.py --image path/to/image.jpg --model path/to/model.h5
    ```

## 🔮 Future Work

This project lays the groundwork for more advanced, context-aware AI systems. Future directions include:

* **Agentic System with LLMs:** Integrate a Large Language Model to interpret Grad-CAM outputs and orchestrate multi-step tasks like detection, analysis, and reporting.
* **Enhanced Vision Models:** Use Vision-Language Models (VLMs) like CLIP to generate descriptive captions for scenes (e.g., "Person carrying suspicious object near entrance").
* **Auto-Notification System:** Implement a real-time alert system via email or SMS when suspicious activity is detected with high confidence.
* **Real-Time Video Processing:** Extend the model using LSTMs or 3D CNNs to analyze video streams and detect suspicious patterns over time, such as loitering.
* **Interactive Dashboard:** Develop a web-based dashboard for real-time monitoring, visualization of heatmaps, and alert management.

## 🤝 Citation

This work is based on the research conducted by:
* Oahed Noor Forhad
* Tohedul Islam Nirzon
* Saiful Islam Rumi

From the Department of Computer Science and Engineering, International Islamic University Chittagong.
