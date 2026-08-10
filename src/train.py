import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==========================================
# Configuration
# ==========================================

TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/val"

MODEL_DIR = "models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# Load Training Dataset
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

# ==========================================
# Load Validation Dataset
# ==========================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================
# Class Names
# ==========================================

class_names = train_dataset.class_names

print("\n================================")
print("CLASSES")
print("================================")

for index, name in enumerate(class_names):
    print(index, "->", name)

# Save class names

with open(
    os.path.join(MODEL_DIR, "class_names.json"),
    "w"
) as f:
    json.dump(class_names, f)

# ==========================================
# Performance
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)

# ==========================================
# Data Augmentation
# ==========================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")

# ==========================================
# MobileNetV2
# ==========================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially

base_model.trainable = False

# ==========================================
# Build Model
# ==========================================

inputs = keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.35)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(
    inputs,
    outputs
)

# ==========================================
# Compile
# ==========================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# Model Summary
# ==========================================

model.summary()

# ==========================================
# Callbacks
# ==========================================

callbacks = [

    keras.callbacks.ModelCheckpoint(
        "models/cricket_face_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        mode="max"
    )
]

# ==========================================
# Training
# ==========================================

print("\n================================")
print("STARTING TRAINING")
print("================================\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# Final Evaluation
# ==========================================

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print("\n================================")
print("TRAINING COMPLETE")
print("================================")

print(
    f"Validation Loss     : {loss:.4f}"
)

print(
    f"Validation Accuracy : {accuracy * 100:.2f}%"
)

print("\nModel:")
print("models/cricket_face_model.keras")

print("\nClasses:")
print("models/class_names.json")