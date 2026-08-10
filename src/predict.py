import os
import sys
import cv2
import numpy as np


# ==========================================
# Configuration
# ==========================================

EMBEDDINGS_PATH = "models/face_embeddings.npz"

SFACE_MODEL = (
    "models/face_recognition_sface_2021dec.onnx"
)

YUNET_MODEL = (
    "models/face_detection_yunet_2023mar.onnx"
)

THRESHOLD = 0.45


# ==========================================
# Input image
# ==========================================

if len(sys.argv) < 2:
    print("\nUsage:")
    print("python src/predict.py path/to/image.jpg")
    raise SystemExit(1)


IMAGE_PATH = sys.argv[1]


if not os.path.exists(IMAGE_PATH):
    print("ERROR: Image not found:")
    print(IMAGE_PATH)
    raise SystemExit(1)


# ==========================================
# Load database
# ==========================================

data = np.load(
    EMBEDDINGS_PATH,
    allow_pickle=True
)

database_embeddings = data["embeddings"]
database_labels = data["labels"]


# ==========================================
# Load models
# ==========================================

detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL,
    ""
)


# ==========================================
# Read image
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Could not read image.")
    raise SystemExit(1)


height, width = image.shape[:2]

detector.setInputSize(
    (width, height)
)


# ==========================================
# Detect face
# ==========================================

_, faces = detector.detect(image)


if faces is None or len(faces) == 0:

    print("\n================================")
    print("NO FACE DETECTED")
    print("================================")

    raise SystemExit(0)


# ==========================================
# Select largest face
# ==========================================

face = max(
    faces,
    key=lambda f: f[2] * f[3]
)


# ==========================================
# Generate embedding
# ==========================================

aligned_face = recognizer.alignCrop(
    image,
    face
)

feature = recognizer.feature(
    aligned_face
)

feature = feature.flatten()

norm = np.linalg.norm(feature)

if norm > 0:
    feature = feature / norm


# ==========================================
# Calculate similarities
# ==========================================

similarities = np.dot(
    database_embeddings,
    feature
)


# ==========================================
# Calculate score for EVERY player
# ==========================================

players = sorted(
    set(
        str(label)
        for label in database_labels
    )
)


player_scores = {}


for player in players:

    indices = np.where(
        database_labels == player
    )[0]

    scores = similarities[indices]

    # Take best 3 images of each player
    top_scores = np.sort(
        scores
    )[::-1][:3]

    # Average top 3
    player_scores[player] = float(
        np.mean(top_scores)
    )


# ==========================================
# Rank players
# ==========================================

ranked_players = sorted(
    player_scores.items(),
    key=lambda x: x[1],
    reverse=True
)


best_player = ranked_players[0][0]
best_score = ranked_players[0][1]


# ==========================================
# Display result
# ==========================================

print("\n================================")
print("FACE IDENTIFICATION RESULT")
print("================================")

print(
    f"Best Match : {best_player}"
)

print(
    f"Similarity : {best_score * 100:.2f}%"
)


print("\nAll Player Matches:")

for player, score in ranked_players:

    print(
        f"{player:20s} "
        f"{score * 100:.2f}%"
    )


# ==========================================
# Final decision
# ==========================================

print("\n================================")

if best_score >= THRESHOLD:

    print(
        f"IDENTIFIED: {best_player}"
    )

else:

    print(
        "UNKNOWN PERSON"
    )

    print(
        "Confidence is too low."
    )

print("================================")