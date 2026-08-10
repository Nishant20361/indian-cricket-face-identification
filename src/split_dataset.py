import os
import shutil
import random

SOURCE_DIR = "processed_dataset"
OUTPUT_DIR = "dataset_split"

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")

TRAIN_RATIO = 0.80
SEED = 42

random.seed(SEED)

# ==========================================
# Create directories
# ==========================================

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# ==========================================
# Process every player
# ==========================================

players = [
    player
    for player in os.listdir(SOURCE_DIR)
    if os.path.isdir(
        os.path.join(SOURCE_DIR, player)
    )
]

print("\nPlayers found:")

for player in players:

    source_player_dir = os.path.join(
        SOURCE_DIR,
        player
    )

    train_player_dir = os.path.join(
        TRAIN_DIR,
        player
    )

    val_player_dir = os.path.join(
        VAL_DIR,
        player
    )

    os.makedirs(
        train_player_dir,
        exist_ok=True
    )

    os.makedirs(
        val_player_dir,
        exist_ok=True
    )

    # --------------------------------------
    # Get images
    # --------------------------------------

    images = [
        file
        for file in os.listdir(
            source_player_dir
        )
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    random.shuffle(images)

    split_index = int(
        len(images) * TRAIN_RATIO
    )

    train_images = images[:split_index]
    val_images = images[split_index:]

    # --------------------------------------
    # Copy training images
    # --------------------------------------

    for image in train_images:

        source = os.path.join(
            source_player_dir,
            image
        )

        destination = os.path.join(
            train_player_dir,
            image
        )

        shutil.copy2(
            source,
            destination
        )

    # --------------------------------------
    # Copy validation images
    # --------------------------------------

    for image in val_images:

        source = os.path.join(
            source_player_dir,
            image
        )

        destination = os.path.join(
            val_player_dir,
            image
        )

        shutil.copy2(
            source,
            destination
        )

    print(
        f"{player}: "
        f"{len(train_images)} train, "
        f"{len(val_images)} validation"
    )

print("\n================================")
print("DATASET SPLIT COMPLETE")
print("================================")