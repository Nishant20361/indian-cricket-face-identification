import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt


# ==========================================
# Configuration
# ==========================================

VAL_DIR = "dataset_split/val"
MODEL_PATH = "models/cricket_face_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


# ==========================================
# Load class names
# ==========================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, "->", name)


# ==========================================
# Load validation dataset
# ==========================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# Load trained model
# ==========================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ==========================================
# Predictions
# ==========================================

y_true = []
y_pred = []

for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ==========================================
# Accuracy
# ==========================================

accuracy = np.mean(
    y_true == y_pred
)

print("\n================================")
print("MODEL EVALUATION")
print("================================")

print(
    f"Validation Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

print("\nRows = Actual")
print("Columns = Predicted")


# ==========================================
# Plot Confusion Matrix
# ==========================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(cm)

plt.title(
    "Indian Cricket Player Face Identification"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(class_names)),
    class_names
)


# Add numbers

for i in range(len(class_names)):

    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png",
    dpi=200
)

plt.show()


print("\nConfusion matrix saved:")
print("models/confusion_matrix.png")

print("\n================================")
print("EVALUATION COMPLETE")
print("================================")