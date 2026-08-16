import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# ==========================================
# 1. SETTINGS
# ==========================================

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 40

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# 2. LOAD TRAINING DATA
# ==========================================

print("Loading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=123
)


# ==========================================
# 3. LOAD VALIDATION DATA
# ==========================================

print("Loading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=123
)


# ==========================================
# 4. LOAD TEST DATA
# ==========================================

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


class_names = train_dataset.class_names

print("\nExpression classes:")
print(class_names)


# ==========================================
# 5. CLASS WEIGHTS
# ==========================================

# FER-2013 has an imbalanced number of images
# in different emotion classes.

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=np.concatenate([
        np.argmax(labels.numpy(), axis=1)
        for _, labels in train_dataset
    ])
)

class_weights = {
    i: weight
    for i, weight in enumerate(class_weights_array)
}

print("\nClass weights:")

for i, class_name in enumerate(class_names):
    print(
        f"{class_name}: "
        f"{class_weights[i]:.2f}"
    )


# ==========================================
# 6. DATA PIPELINE PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================
# 7. DATA AUGMENTATION
# ==========================================

data_augmentation = models.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.08
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomTranslation(
        height_factor=0.08,
        width_factor=0.08
    )

], name="data_augmentation")


# ==========================================
# 8. BUILD IMPROVED CNN
# ==========================================

model = models.Sequential([

    layers.Input(
        shape=(48, 48, 1)
    ),

    # -----------------------------
    # Augmentation
    # -----------------------------

    data_augmentation,

    # -----------------------------
    # Normalization
    # -----------------------------

    layers.Rescaling(
        1.0 / 255
    ),

    # -----------------------------
    # CNN BLOCK 1
    # -----------------------------

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(
        0.25
    ),

    # -----------------------------
    # CNN BLOCK 2
    # -----------------------------

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(
        0.25
    ),

    # -----------------------------
    # CNN BLOCK 3
    # -----------------------------

    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(
        0.30
    ),

    # -----------------------------
    # CLASSIFICATION
    # -----------------------------

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Dropout(
        0.50
    ),

    layers.Dense(
        7,
        activation="softmax"
    )
])


# ==========================================
# 9. COMPILE MODEL
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ==========================================
# 10. MODEL SUMMARY
# ==========================================

model.summary()


# ==========================================
# 11. CALLBACKS
# ==========================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            MODEL_DIR,
            "emotion_model.keras"
        ),
        monitor="val_accuracy",
        save_best_only=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]


# ==========================================
# 12. TRAIN
# ==========================================

print("\nStarting improved CNN training...\n")

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# ==========================================
# 13. TEST
# ==========================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)


# ==========================================
# 14. ACCURACY GRAPH
# ==========================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training and Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "accuracy_curve.png"
    )
)

plt.close()


# ==========================================
# 15. LOSS GRAPH
# ==========================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "loss_curve.png"
    )
)

plt.close()


# ==========================================
# 16. SAVE FINAL MODEL
# ==========================================

model.save(
    os.path.join(
        MODEL_DIR,
        "emotion_model_final.keras"
    )
)

print("\n================================")
print("Improved training completed!")
print("================================")

print(
    "\nModel saved in:"
    "\nmodels/"
)

print(
    "\nGraphs saved in:"
    "\nresults/"
)