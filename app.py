import streamlit as st
import cv2
import numpy as np
import json
from pathlib import Path


# ==============================
# CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Indian Cricket Face Identification",
    page_icon="🏏",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
EMBEDDINGS_PATH = BASE_DIR / "models" / "face_embeddings.npz"


# ==============================
# LOAD MODELS
# ==============================

@st.cache_resource
def load_models():

    recognizer = cv2.FaceRecognizerSF.create(
        str(MODEL_PATH),
        ""
    )

    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)

    embeddings = data["embeddings"]
    labels = data["labels"]

    return recognizer, embeddings, labels


recognizer, known_embeddings, known_labels = load_models()


# ==============================
# LOAD CLASS NAMES
# ==============================
class_names = {
    "Hardik_Pandya": "Hardik Pandya",
    "Jasprit_Bumrah": "Jasprit Bumrah",
    "MS_Dhoni": "MS Dhoni",
    "Rohit_Sharma": "Rohit Sharma",
    "Virat_Kohli": "Virat Kohli",
    "KL_Rahul": "KL Rahul",
    "Bhuvneshwar_Kumar": "Bhuvneshwar Kumar",
    "Dinesh_Karthik": "Dinesh Karthik",
    "Kedar_Jadhav": "Kedar Jadhav",
    "Kuldeep_Yadav": "Kuldeep Yadav",
    "Mohammed_Shami": "Mohammed Shami",
    "Ravindra_Jadeja": "Ravindra Jadeja",
    "Shikhar_Dhawan": "Shikhar Dhawan",
    "Vijay_Shankar": "Vijay Shankar",
    "Yuzvendra_Chahal": "Yuzvendra Chahal"
}


# ==============================
# FACE DETECTOR
# ==============================

@st.cache_resource
def load_detector():

    model_path = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"

    detector = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (320, 320),
        0.6,
        0.3,
        5000
    )

    return detector


detector = load_detector()


# ==============================
# PAGE HEADER
# ==============================

st.title("🏏 Indian Cricket Team")
st.subheader("Face Identification System")

st.write(
    "Upload an image of one of the supported Indian cricket players "
    "and the system will identify the closest matching player."
)

st.info(
    "Supported players: Virat Kohli, Rohit Sharma, MS Dhoni, "
    "Jasprit Bumrah, Hardik Pandya, KL Rahul, "
    "Bhuvneshwar Kumar, Dinesh Karthik, Kedar Jadhav, "
    "Kuldeep Yadav, Mohammed Shami, Ravindra Jadeja, "
    "Shikhar Dhawan, Vijay Shankar and Yuzvendra Chahal"
)


# ==============================
# IMAGE UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "📸 Upload a player image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error("Unable to read the image.")

    else:

        st.image(
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    caption="Uploaded Image",
    width=700
)
        # ==============================
        # FACE DETECTION
        # ==============================

        height, width = image.shape[:2]

        detector.setInputSize((width, height))

        _, faces = detector.detect(image)

        if faces is None or len(faces) == 0:

            st.error(
                "❌ No face detected. Please upload a clear face image."
            )

        else:

            # Use largest detected face
            face = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            # ==============================
            # FACE CROP
            # ==============================

            x, y, w, h = face[:4].astype(int)

            x = max(0, x)
            y = max(0, y)

            w = min(w, width - x)
            h = min(h, height - y)

            face_crop = image[
                y:y+h,
                x:x+w
            ]

            if face_crop.size == 0:

                st.error("Unable to crop detected face.")

            else:

                st.success("✅ Face detected successfully!")

                st.image(
                    cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB),
                    caption="Detected Face",
                    width=250
                )

                # ==============================
                # ALIGN FACE
                # ==============================

                aligned_face = recognizer.alignCrop(
                    image,
                    face
                )

                # ==============================
                # CREATE EMBEDDING
                # ==============================
                # CREATE FACE EMBEDDING
                # ==============================

                feature = recognizer.feature(aligned_face)

                # Normalize embedding
                feature = feature.flatten()
                feature = feature / (np.linalg.norm(feature) + 1e-10)

                # ==============================
                # COMPARE WITH KNOWN EMBEDDINGS
                # ==============================

                similarities = []

                for known_embedding, label in zip(
                    known_embeddings,
                    known_labels
                ):

                    known_embedding = known_embedding.flatten()

                    known_embedding = known_embedding / (
                        np.linalg.norm(known_embedding) + 1e-10
                    )

                    similarity = float(
                        np.dot(feature, known_embedding)
                    )

                    similarities.append(
                        (similarity, label)
                    )

                # Sort highest similarity first
                similarities.sort(
                    key=lambda x: x[0],
                    reverse=True
                )

                # ==============================
                # BEST MATCH
                # ==============================

                best_similarity, best_label = similarities[0]

                player_name = class_names.get(
                    str(best_label),
                    str(best_label)
                )

                similarity_percent = max(
                    0,
                    min(100, best_similarity * 100)
                )

                # ==============================
                # DISPLAY RESULT
                # ==============================

                st.divider()

                st.subheader("🏏 Identification Result")

                st.success(
                    f"🏆 **{player_name}**"
                )

                st.metric(
                    "Similarity",
                    f"{similarity_percent:.2f}%"
                )

                # ==============================
                # TOP MATCHES
                # ==============================

                st.subheader("🔍 Top Matches")

                # Keep only best score for each player
                player_best = {}

                for score, label in similarities:

                    label = str(label)

                    if (
                        label not in player_best
                        or score > player_best[label]
                    ):
                        player_best[label] = score

                top_players = sorted(
                    player_best.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                for label, score in top_players:

                    name = class_names.get(
                        label,
                        label
                    )

                    st.write(
                        f"**{name}** — "
                        f"{max(0, min(100, score * 100)):.2f}%"
                    )