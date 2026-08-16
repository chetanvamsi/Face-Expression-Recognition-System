import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================
# SETTINGS
# =========================

IMG_SIZE = (48, 48)
BATCH_SIZE = 64

TEST_DIR = "data/test"
MODEL_PATH = "models/emotion_model.keras"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# LOAD MODEL
# =========================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)


# =========================
# LOAD TEST DATA
# =========================

print("Loading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print("\nClasses:")
print(class_names)


# =========================
# MAKE PREDICTIONS
# =========================

print("\nMaking predictions...")

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# =========================
# CLASSIFICATION REPORT
# =========================

print("\nClassification Report:\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print(report)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9, 7))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Facial Expression Recognition - Confusion Matrix")
plt.tight_layout()

output_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(output_path)

plt.close()

print(f"\nConfusion matrix saved to: {output_path}")

print("\nEvaluation completed!")