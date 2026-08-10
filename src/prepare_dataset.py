import cv2
import os

# ==============================
# Paths
# ==============================

INPUT_DIR = "dataset"
OUTPUT_DIR = "processed_dataset"

# ==============================
# OpenCV 5 YuNet Face Detector
# ==============================

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "face_detection_yunet_2023mar.onnx"
)
# ==============================
# Check model
# ==============================

if not os.path.exists(MODEL_PATH):
    print("ERROR: YuNet face detection model not found!")
    print()
    print("Expected file:")
    print(MODEL_PATH)
    print()
    print("We will download it in the next step.")
    exit()

# ==============================
# Create detector
# ==============================

detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

# ==============================
# Players
# ==============================

players = [
    "Virat_Kohli",
    "Rohit_Sharma",
    "MS_Dhoni",
    "Jasprit_Bumrah",
    "Hardik_Pandya",
    "KL_Rahul",
    "Bhuvneshwar_Kumar",
    "Dinesh_Karthik",
    "Kedar_Jadhav",
    "Kuldeep_Yadav",
    "Mohammed_Shami",
    "Ravindra_Jadeja",
    "Shikhar_Dhawan",
    "Vijay_Shankar",
    "Yuzvendra_Chahal"
]

total_images = 0
total_faces = 0
failed_images = 0


# ==============================
# Process dataset
# ==============================

for player in players:

    input_folder = os.path.join(INPUT_DIR, player)
    output_folder = os.path.join(OUTPUT_DIR, player)

    os.makedirs(output_folder, exist_ok=True)

    print(f"\nProcessing: {player}")

    if not os.path.exists(input_folder):
        print(f"Folder not found: {input_folder}")
        continue

    image_files = os.listdir(input_folder)

    player_faces = 0

    for filename in image_files:

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            continue

        file_path = os.path.join(input_folder, filename)

        total_images += 1

        # Read image
        image = cv2.imread(file_path)

        if image is None:
            print(f"Could not read: {filename}")
            failed_images += 1
            continue

        # Get image dimensions
        height, width = image.shape[:2]

        # Update detector input size
        detector.setInputSize((width, height))

        # Detect faces
        _, faces = detector.detect(image)

        if faces is None or len(faces) == 0:
            print(f"No face found: {filename}")
            continue

        # ==============================
        # Select largest face
        # ==============================

        largest_face = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        x, y, w, h = largest_face[:4]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # ==============================
        # Add margin
        # ==============================

        margin = int(0.20 * max(w, h))

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)

        x2 = min(width, x + w + margin)
        y2 = min(height, y + h + margin)

        # ==============================
        # Crop face
        # ==============================

        face = image[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # ==============================
        # Resize
        # ==============================

        face = cv2.resize(face, (224, 224))

        # ==============================
        # Save
        # ==============================

        output_filename = f"{player_faces:04d}.jpg"

        output_path = os.path.join(
            output_folder,
            output_filename
        )

        cv2.imwrite(output_path, face)

        player_faces += 1
        total_faces += 1

    print(f"Faces saved for {player}: {player_faces}")


# ==============================
# Summary
# ==============================

print("\n================================")
print("DATASET PROCESSING COMPLETE")
print("================================")

print(f"Total images checked : {total_images}")
print(f"Total faces saved    : {total_faces}")
print(f"Failed images        : {failed_images}")

print("\nProcessed dataset:")
print(OUTPUT_DIR)