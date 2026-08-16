# Face Expression Recognition System

A deep learning-based facial expression recognition system that uses a Convolutional Neural Network (CNN) to classify human facial expressions from images and real-time webcam input.

The system combines **TensorFlow, Keras, OpenCV, and Python** to perform face detection, image preprocessing, facial expression classification, and real-time prediction.

---

## Project Overview

Facial expression recognition is a computer vision and deep learning task that identifies human emotions from facial features.

This project was developed to build an end-to-end facial expression recognition pipeline covering:

- Dataset preparation
- Image preprocessing
- Data augmentation
- CNN model development
- Model training
- Model evaluation
- Image-based prediction
- Real-time webcam prediction

The trained model classifies faces into seven expression categories.

---

## Features

- CNN-based facial expression recognition
- Recognition of 7 facial expressions
- Face detection using OpenCV
- Image preprocessing and normalization
- Data augmentation
- Class-weighted model training
- Model evaluation using multiple metrics
- Confusion matrix generation
- Training accuracy visualization
- Training loss visualization
- Image-based expression prediction
- Real-time webcam expression detection
- Prediction confidence score
- Modular Python implementation

---

## Facial Expression Classes

The model recognizes the following expressions:

| Expression |
|------------|
| Angry |
| Disgust |
| Fear |
| Happy |
| Neutral |
| Sad |
| Surprise |

---

## Technologies Used

| Technology | Usage |
|------------|-------|
| Python | Programming |
| TensorFlow | Deep learning |
| Keras | CNN model development |
| OpenCV | Face detection and webcam processing |
| NumPy | Numerical operations |
| Pandas | Data processing |
| Scikit-learn | Model evaluation |
| Matplotlib | Visualization |
| Pillow | Image processing |

---

## System Workflow

```text
Input Image / Webcam
        |
        v
    Face Detection
        |
        v
    Face Extraction
        |
        v
 Grayscale Conversion
        |
        v
   Resize 48 x 48
        |
        v
    Normalization
        |
        v
       CNN
        |
        v
Feature Extraction
        |
        v
Expression Classification
        |
        v
Expression + Confidence