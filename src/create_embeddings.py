import os
import cv2
import json
import numpy as np


# ==========================================
# Configuration
# ==========================================

DATASET_DIR = "processed_dataset"

MODEL_PATH = (
    "models/face_recognition_sface_2021dec.onnx"
)

OUTPUT_PATH = "models/face_embeddings.npz"

# Same YuNet detector used during preprocessing
DETECTOR_PATH = (
    "models/face_detection_yunet_2023mar.onnx"
)

IMAGE_SIZE = (112, 112)


# ==========================================
# Check files
# ==========================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: SFace model not found!")
    print(MODEL_PATH)
    raise SystemExit(1)

if not os.path.exists(DETECTOR_PATH):
    print("ERROR: YuNet model not found!")
    print(DETECTOR_PATH)
    raise SystemExit(1)


# ==========================================
# Create YuNet detector
# ==========================================

detector = cv2.FaceDetectorYN.create(
    DETECTOR_PATH,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)


# ==========================================
# Create SFace recognizer
# ==========================================

recognizer = cv2.FaceRecognizerSF.create(
    MODEL_PATH,
    ""
)


# ==========================================
# Storage
# ==========================================

embeddings = []
labels = []


# ==========================================
# Process players
# ==========================================

players = sorted([
    name
    for name in os.listdir(DATASET_DIR)
    if os.path.isdir(
        os.path.join(DATASET_DIR, name)
    )
])


print("\nPlayers:")

for player in players:
    print("-", player)


# ==========================================
# Process images
# ==========================================

for player in players:

    player_dir = os.path.join(
        DATASET_DIR,
        player
    )

    print(
        f"\nProcessing: {player}"
    )

    count = 0

    for filename in os.listdir(player_dir):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            continue

        image_path = os.path.join(
            player_dir,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            print(
                "Could not read:",
                filename
            )
            continue

        height, width = image.shape[:2]

        detector.setInputSize(
            (width, height)
        )

        _, faces = detector.detect(image)

        if faces is None or len(faces) == 0:
            print(
                "No face:",
                filename
            )
            continue

        # Select largest detected face
        face = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        # Align face using SFace landmarks
        aligned_face = recognizer.alignCrop(
            image,
            face
        )

        # Generate embedding
        feature = recognizer.feature(
            aligned_face
        )

        # Normalize embedding
        feature = feature.flatten()

        norm = np.linalg.norm(feature)

        if norm > 0:
            feature = feature / norm

        embeddings.append(feature)
        labels.append(player)

        count += 1

    print(
        f"Embeddings created: {count}"
    )


# ==========================================
# Save embeddings
# ==========================================

if len(embeddings) == 0:

    print("\nERROR: No embeddings generated.")
    raise SystemExit(1)


embeddings = np.array(
    embeddings,
    dtype=np.float32
)

labels = np.array(
    labels
)

np.savez_compressed(
    OUTPUT_PATH,
    embeddings=embeddings,
    labels=labels
)


print("\n================================")
print("EMBEDDING CREATION COMPLETE")
print("================================")

print(
    "Total embeddings:",
    len(embeddings)
)

print(
    "Embedding dimension:",
    embeddings.shape[1]
)

print(
    "Saved:",
    OUTPUT_PATH
)