import cv2
import numpy as np
import tensorflow as tf
import os

# =========================
# SETTINGS
# =========================

MODEL_PATH = "models/emotion_model.keras"

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

IMG_SIZE = 48


# =========================
# LOAD MODEL
# =========================

print("Loading emotion recognition model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# =========================
# LOAD FACE DETECTOR
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# =========================
# PREDICT FUNCTION
# =========================

def predict_expression(image_path):

    # Check file
    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        print("No face detected.")
        return

    print(f"\nDetected {len(faces)} face(s).")

    # Process every detected face
    for i, (x, y, w, h) in enumerate(faces):

        face = gray[y:y+h, x:x+w]

        # Resize
        face = cv2.resize(
            face,
            (IMG_SIZE, IMG_SIZE)
        )

        # Normalize
        face = face.astype("float32") / 255.0

        # Add dimensions
        face = np.expand_dims(
            face,
            axis=-1
        )

        face = np.expand_dims(
            face,
            axis=0
        )

        # Prediction
        predictions = model.predict(
            face,
            verbose=0
        )[0]

        predicted_index = np.argmax(
            predictions
        )

        expression = CLASS_NAMES[
            predicted_index
        ]

        confidence = predictions[
            predicted_index
        ] * 100

        print(
            f"Face {i + 1}: "
            f"{expression.upper()} "
            f"({confidence:.2f}%)"
        )

        # Draw result
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        label = (
            f"{expression.upper()} "
            f"{confidence:.1f}%"
        )

        cv2.putText(
            image,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Display image
    cv2.imshow(
        "Face Expression Recognition",
        image
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    image_path = input(
        "\nEnter image path: "
    ).strip()

    predict_expression(
        image_path
    )