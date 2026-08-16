# Face Expression Recognition System

A deep learning-based facial expression recognition system that detects and classifies human facial expressions using a Convolutional Neural Network (CNN), TensorFlow, Keras, and OpenCV.

The system supports both **image-based expression prediction** and **real-time webcam-based facial expression detection**.

---
## Demo

The system detects a face using OpenCV and predicts the facial expression using the trained CNN model.

![Real-Time Face Expression Detection](...)

## Features

- CNN-based facial expression recognition
- Classification of 7 different facial expressions
- Image preprocessing and normalization
- Data augmentation during training
- Class-weighted training to handle class imbalance
- Model evaluation using accuracy, precision, recall, and F1-score
- Confusion matrix generation
- Image-based facial expression prediction
- Real-time webcam facial expression detection
- Confidence score displayed for predictions
- Modular training, evaluation, prediction, and real-time detection scripts

---

## Facial Expressions

The model recognizes the following 7 expressions:

| Class | Expression |
|---|---|
| 1 | Angry |
| 2 | Disgust |
| 3 | Fear |
| 4 | Happy |
| 5 | Neutral |
| 6 | Sad |
| 7 | Surprise |

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| OpenCV | Face detection and real-time webcam processing |
| NumPy | Numerical computations |
| Pandas | Data processing |
| Scikit-learn | Model evaluation |
| Matplotlib | Visualization |
| Pillow | Image processing |

---

## System Architecture

The overall system works as follows:

```text
                Input Image / Webcam
                        |
                        v
                Face Detection
                        |
                        v
                  Face Cropping
                        |
                        v
                 Grayscale Image
                        |
                        v
                  Resize 48 x 48
                        |
                        v
                   Normalize
                        |
                        v
                 Data Augmentation
                        |
                        v
                    CNN Model
                        |
                        v
              Expression Prediction
                        |
                        v
       Expression + Confidence Score