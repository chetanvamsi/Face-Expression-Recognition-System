import cv2
import numpy as np
import tensorflow as tf

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

class CompatibleGlorotUniform(tf.keras.initializers.Initializer):
    def __init__(
        self,
        seed=None,
        input_axes=None,
        output_axes=None,
        **kwargs
    ):
        self.seed = seed
        self.input_axes = input_axes
        self.output_axes = output_axes

    def __call__(self, shape, dtype=None):
        initializer = tf.keras.initializers.GlorotUniform(
            seed=self.seed
        )
        return initializer(shape, dtype=dtype)

    def get_config(self):
        return {
            "seed": self.seed,
            "input_axes": self.input_axes,
            "output_axes": self.output_axes
        }


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={
        "GlorotUniform": CompatibleGlorotUniform
    }
)

print("Model loaded successfully.")

# =========================
# LOAD FACE DETECTOR
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error: Could not load face detector.")
    exit()


# =========================
# START WEBCAM
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")


# =========================
# REAL-TIME LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # Mirror the webcam
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Process each face
    for (x, y, w, h) in faces:

        # Crop face
        face = gray[
            y:y + h,
            x:x + w
        ]

        # Resize
        face = cv2.resize(
            face,
            (IMG_SIZE, IMG_SIZE)
        )

        # Normalize
        face = face.astype(
            "float32"
        ) / 255.0

        # Add channel dimension
        face = np.expand_dims(
            face,
            axis=-1
        )

        # Add batch dimension
        face = np.expand_dims(
            face,
            axis=0
        )

        # Predict
        predictions = model.predict(
            face,
            verbose=0
        )[0]

        # Get highest probability
        predicted_index = np.argmax(
            predictions
        )

        expression = CLASS_NAMES[
            predicted_index
        ]

        confidence = (
            predictions[predicted_index]
            * 100
        )

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Prediction label
        label = (
            f"{expression.upper()} "
            f"{confidence:.1f}%"
        )

        # Draw label
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Display window
    cv2.imshow(
        "Face Expression Recognition System",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()

print("Webcam closed.")