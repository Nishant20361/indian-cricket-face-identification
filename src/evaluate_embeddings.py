import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


from pathlib import Path

# ==========================================
# Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "test_set"
EMBEDDINGS_PATH = BASE_DIR / "models" / "face_embeddings.npz"
SFACE_MODEL = BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
YUNET_MODEL = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"

THRESHOLD = 0.45


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
    str(YUNET_MODEL),
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    str(SFACE_MODEL),
    ""
)


# ==========================================
# Player names
# ==========================================

players = sorted(
    set(
        str(label)
        for label in database_labels
    )
)


print("\nPlayers:")
for player in players:
    print("-", player)


# ==========================================
# Prediction function
# ==========================================

def predict(image):

    height, width = image.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None, 0.0

    # Largest face
    face = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    # Align
    aligned_face = recognizer.alignCrop(
        image,
        face
    )

    # Embedding
    feature = recognizer.feature(
        aligned_face
    )

    feature = feature.flatten()

    norm = np.linalg.norm(feature)

    if norm > 0:
        feature = feature / norm

    # Similarity
    similarities = np.dot(
        database_embeddings,
        feature
    )

    # Score for every player
    player_scores = {}

    for player in players:

        indices = np.where(
            database_labels == player
        )[0]

        scores = similarities[indices]

        # Best 3 images
        top_scores = np.sort(
            scores
        )[::-1][:3]

        player_scores[player] = float(
            np.mean(top_scores)
        )

    # Best player
    best_player = max(
        player_scores,
        key=player_scores.get
    )

    best_score = player_scores[
        best_player
    ]

    return best_player, best_score


# ==========================================
# Evaluate test dataset
# ==========================================

actual = []
predicted = []

total = 0
correct = 0
no_face = 0


print("\n================================")
print("EMBEDDING MODEL EVALUATION")
print("================================")


for actual_player in players:

    player_dir = os.path.join(
        TEST_DIR,
        actual_player
    )

    if not os.path.isdir(player_dir):
        continue

    for filename in sorted(
        os.listdir(player_dir)
    ):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            continue

        image_path = os.path.join(
            player_dir,
            filename
        )

        image = cv2.imread(
            image_path
        )

        if image is None:
            continue

        total += 1

        prediction, score = predict(
            image
        )

        if prediction is None:

            no_face += 1

            print(
                f"{actual_player:20s} "
                f"{filename:20s} "
                f"NO FACE"
            )

            continue

        actual.append(
            actual_player
        )

        predicted.append(
            prediction
        )

        if prediction == actual_player:
            correct += 1
            result = "✓"
        else:
            result = "✗"

        print(
            f"{actual_player:20s} "
            f"{filename:20s} "
            f"{prediction:20s} "
            f"{score * 100:6.2f}% "
            f"{result}"
        )


# ==========================================
# Accuracy
# ==========================================

evaluated = len(actual)

if evaluated > 0:

    accuracy = (
        correct / evaluated
    ) * 100

else:

    accuracy = 0


print("\n================================")
print("RESULT")
print("================================")

print(
    f"Total test images : {total}"
)

print(
    f"Faces detected    : {evaluated}"
)

print(
    f"No face detected  : {no_face}"
)

print(
    f"Correct           : {correct}"
)

print(
    f"Wrong             : {evaluated - correct}"
)

print(
    f"Accuracy          : {accuracy:.2f}%"
)


# ==========================================
# Classification report
# ==========================================

if evaluated > 0:

    print("\n================================")
    print("CLASSIFICATION REPORT")
    print("================================")

    print(
        classification_report(
            actual,
            predicted,
            labels=players,
            zero_division=0
        )
    )


    # ======================================
    # Confusion matrix
    # ======================================

    print("================================")
    print("CONFUSION MATRIX")
    print("================================")

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=players
    )

    print()

    print(
        "Actual \\ Predicted"
    )

    print(
        " " * 20 +
        " ".join(
            f"{p[:8]:>10}"
            for p in players
        )
    )

    for i, player in enumerate(players):

        print(
            f"{player:20s}"
            +
            " ".join(
                f"{value:10d}"
                for value in matrix[i]
            )
        )


print("\n================================")
print("EVALUATION COMPLETE")
print("================================")