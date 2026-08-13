import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import io

# ==============================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================

st.set_page_config(
    page_title="Indian Cricket Face Identification AI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Glassmorphism CSS Inject
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Styling */
    .stApp {
        background-color: #0B0E14;
        background-image: 
            radial-gradient(at 20% 20%, rgba(14, 165, 233, 0.12) 0px, transparent 50%),
            radial-gradient(at 80% 80%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
            radial-gradient(at 50% 50%, rgba(16, 185, 129, 0.05) 0px, transparent 50%);
        background-attachment: fixed;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(18, 24, 38, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    .glass-card-sm {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* Header Styling */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 24px;
    }

    /* Badge & Status Styles */
    .status-badge-match {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%);
        border: 1px solid #10B981;
        color: #34D399;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }

    .status-badge-unknown {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.3) 100%);
        border: 1px solid #EF4444;
        color: #FCA5A5;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }

    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #38BDF8 !important;
    }

    /* Player Pill Tags */
    .player-tag {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #E2E8F0;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 3px;
        display: inline-block;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #090C12 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

</style>
""", unsafe_allow_html=True)


# ==============================================
# 2. PATHS & MODEL LOADING
# ==============================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
EMBEDDINGS_PATH = BASE_DIR / "models" / "face_embeddings.npz"
DETECTOR_PATH = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"

CLASS_NAMES = {
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

PLAYER_EMOJIS = {
    "Virat_Kohli": "👑",
    "MS_Dhoni": "🦁",
    "Rohit_Sharma": "💥",
    "Jasprit_Bumrah": "⚡",
    "Hardik_Pandya": "🔥",
    "KL_Rahul": "🎯",
    "Ravindra_Jadeja": "🗡️",
    "Shikhar_Dhawan": "🦅",
    "Mohammed_Shami": "🎯",
    "Bhuvneshwar_Kumar": "🎯",
    "Kuldeep_Yadav": "🌀",
    "Yuzvendra_Chahal": "♟️",
    "Dinesh_Karthik": "⚡",
    "Kedar_Jadhav": "🌟",
    "Vijay_Shankar": "🌟"
}


@st.cache_resource
def load_models():
    """Load SFace Recognizer and Pre-calculated Face Embeddings."""
    if not MODEL_PATH.exists() or not EMBEDDINGS_PATH.exists():
        st.error(f"Missing required model files in `{BASE_DIR / 'models'}`")
        st.stop()
        
    recognizer = cv2.FaceRecognizerSF.create(str(MODEL_PATH), "")
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    
    embeddings = data["embeddings"]
    labels = data["labels"]
    
    return recognizer, embeddings, labels


@st.cache_resource
def load_detector(score_threshold=0.5):
    """Load YuNet Face Detector."""
    if not DETECTOR_PATH.exists():
        st.error(f"Missing YuNet detector model at `{DETECTOR_PATH}`")
        st.stop()
        
    detector = cv2.FaceDetectorYN.create(
        str(DETECTOR_PATH),
        "",
        (320, 320),
        score_threshold,
        0.3,
        5000
    )
    return detector


recognizer, known_embeddings, known_labels = load_models()


# ==============================================
# 3. IDENTIFICATION ALGORITHM & PIPELINE
# ==============================================

def identify_face(aligned_face, known_embeddings, known_labels):
    """
    Generate 128D SFace embedding and match against database
    using top-3 cosine similarity aggregation per player.
    """
    feature = recognizer.feature(aligned_face).flatten()
    norm = np.linalg.norm(feature)
    if norm > 0:
        feature = feature / norm

    # Calculate cosine similarity with all stored embeddings
    similarities = np.dot(known_embeddings, feature)

    unique_players = sorted(list(set(str(label) for label in known_labels)))
    player_scores = {}

    for player in unique_players:
        indices = np.where(known_labels == player)[0]
        scores = similarities[indices]
        # Top 3 scores mean
        top_scores = np.sort(scores)[::-1][:3]
        player_scores[player] = float(np.mean(top_scores))

    # Rank players descending
    ranked = sorted(player_scores.items(), key=lambda x: x[1], reverse=True)
    best_player, best_score = ranked[0]
    
    return best_player, best_score, ranked


def process_image(image_bgr, conf_threshold, det_threshold):
    """
    Full pipeline: Detect faces, draw bounding boxes, align, and perform recognition.
    """
    height, width = image_bgr.shape[:2]
    detector = load_detector(score_threshold=det_threshold)
    detector.setInputSize((width, height))

    _, faces = detector.detect(image_bgr)
    
    if faces is None or len(faces) == 0:
        return None, []

    annotated_img = image_bgr.copy()
    face_results = []

    # Sort faces from left to right for intuitive ordering
    faces = sorted(faces, key=lambda f: f[0])

    for idx, face in enumerate(faces):
        # 1. Bounding box coordinates with safe clamping
        x, y, w, h = face[:4].astype(int)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)

        face_crop = image_bgr[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # 2. Align face
        aligned_face = recognizer.alignCrop(image_bgr, face)

        # 3. Recognize
        best_player, score, ranked = identify_face(aligned_face, known_embeddings, known_labels)

        is_match = score >= conf_threshold
        display_name = CLASS_NAMES.get(best_player, best_player) if is_match else "Unknown Person"
        emoji = PLAYER_EMOJIS.get(best_player, "🏏") if is_match else "❓"

        # 4. Draw bounding box and badge on image
        box_color = (129, 185, 16) if is_match else (68, 68, 239)  # Green if match, Red if unknown
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 3)

        label_text = f"#{idx+1}: {display_name} ({score*100:.1f}%)"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Label background box above head
        label_y1 = max(0, y1 - text_h - 10)
        cv2.rectangle(annotated_img, (x1, label_y1), (x1 + text_w + 12, label_y1 + text_h + 10), box_color, -1)
        cv2.putText(annotated_img, label_text, (x1 + 6, label_y1 + text_h + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        face_results.append({
            "face_index": idx + 1,
            "crop_rgb": cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB),
            "best_player": best_player,
            "display_name": display_name,
            "score": score,
            "score_percent": score * 100,
            "is_match": is_match,
            "emoji": emoji,
            "ranked": ranked
        })

    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    return annotated_rgb, face_results


# ==============================================
# 4. SIDEBAR - CONTROLS & ROSTER
# ==============================================

with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    
    conf_threshold = st.slider(
        "🎯 Match Confidence Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.45,
        step=0.05,
        help="Minimum similarity required to identify a face. Below this score, person is tagged as Unknown."
    )

    det_threshold = st.slider(
        "🔍 Face Detection Threshold",
        min_value=0.30,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="YuNet face detection confidence threshold."
    )

    st.markdown("---")
    st.markdown("### 🏏 Supported Squad (15 Players)")
    
    roster_html = "".join([
        f'<span class="player-tag">{PLAYER_EMOJIS.get(k, "🏏")} {v}</span>'
        for k, v in CLASS_NAMES.items()
    ])
    st.markdown(roster_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 Model Architecture")
    st.caption("**Face Detector**: YuNet (ONNX) - 320x320")
    st.caption("**Feature Extractor**: SFace (ONNX) - 128D Cosine Similarity")
    st.caption("**Aggregation**: Top-3 Mean Vector Matching")


# ==============================================
# 5. MAIN HEADER & HERO SECTION
# ==============================================

st.markdown('<h1 class="hero-title">🏏 Indian Cricket Team AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Real-Time Face Identification & Player Recognition System</p>', unsafe_allow_html=True)


# ==============================================
# 6. INPUT SOURCE SELECTION
# ==============================================

input_tab1, input_tab2, input_tab3 = st.tabs([
    "📁 Upload Image",
    "📷 Live Webcam Capture",
    "⚡ Quick Sample Presets"
])

selected_image_bgr = None

with input_tab1:
    uploaded_file = st.file_uploader(
        "Upload a photo of an Indian cricket player (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="uploader"
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
        selected_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

with input_tab2:
    st.markdown(
        "Click **Enable Camera** below to activate your webcam. "
        "Your browser will then ask for camera permission."
    )
    if "camera_enabled" not in st.session_state:
        st.session_state.camera_enabled = False

    if not st.session_state.camera_enabled:
        if st.button("📷 Enable Camera", key="enable_camera"):
            st.session_state.camera_enabled = True
            st.rerun()
    else:
        camera_file = st.camera_input("Take a snapshot for identification")
        if camera_file is not None:
            file_bytes = np.asarray(bytearray(camera_file.getvalue()), dtype=np.uint8)
            selected_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if st.button("🔇 Disable Camera", key="disable_camera"):
            st.session_state.camera_enabled = False
            st.rerun()

with input_tab3:
    st.write("Click any sample image below to run instant face identification:")
    
    # Uses the git-committed samples/ folder so images are available on Streamlit Cloud
    SAMPLES_DIR = BASE_DIR / "samples"
    samples = [
        {"name": "Virat Kohli",   "emoji": "👑", "path": SAMPLES_DIR / "Virat_Kohli.jpg"},
        {"name": "MS Dhoni",      "emoji": "🦁", "path": SAMPLES_DIR / "MS_Dhoni.jpg"},
        {"name": "Rohit Sharma",  "emoji": "💥", "path": SAMPLES_DIR / "Rohit_Sharma.jpg"},
        {"name": "Jasprit Bumrah","emoji": "⚡", "path": SAMPLES_DIR / "Jasprit_Bumrah.jpg"},
        {"name": "Hardik Pandya", "emoji": "🔥", "path": SAMPLES_DIR / "Hardik_Pandya.jpg"}
    ]
    
    if not SAMPLES_DIR.exists() or not any(SAMPLES_DIR.iterdir()):
        st.warning(
            "⚠️ Sample images folder (`samples/`) not found. "
            "Run `cp test_images/virat.jpg samples/Virat_Kohli.jpg` etc. "
            "or use the Upload tab instead."
        )
    else:
        cols = st.columns(len(samples))
        for i, sample in enumerate(samples):
            with cols[i]:
                if sample["path"].exists():
                    img_pil = Image.open(sample["path"])
                    st.image(img_pil, use_container_width=True, caption=f"{sample['emoji']} {sample['name']}")
                    if st.button(f"Identify {sample['name']}", key=f"btn_{i}", use_container_width=True):
                        selected_image_bgr = cv2.imread(str(sample["path"]))
                else:
                    st.info(f"{sample['emoji']} {sample['name']}\n\nImage not available")


# ==============================================
# 7. INFERENCE & RESULTS DISPLAY
# ==============================================

if selected_image_bgr is not None:
    st.markdown("---")
    
    with st.spinner("🔍 Detecting and analyzing faces..."):
        annotated_rgb, face_results = process_image(
            selected_image_bgr,
            conf_threshold=conf_threshold,
            det_threshold=det_threshold
        )

    if not face_results:
        st.error("❌ No faces detected in the image. Please try uploading a clearer front-facing photo.")
    else:
        # Results Header Layout
        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🎯 Detection Canvas")
            st.image(annotated_rgb, use_container_width=True, caption="Multi-Face Bounding Box Annotations")
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader(f"📊 Identification Summary ({len(face_results)} Face{'s' if len(face_results) > 1 else ''})")
            
            for res in face_results:
                st.markdown('<div class="glass-card-sm">', unsafe_allow_html=True)
                
                fcol1, fcol2 = st.columns([1, 2])
                with fcol1:
                    st.image(res["crop_rgb"], width=110, caption=f"Face #{res['face_index']}")
                
                with fcol2:
                    if res["is_match"]:
                        st.markdown(f'<span class="status-badge-match">🏆 {res["emoji"]} {res["display_name"]}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="status-badge-unknown">⚠️ {res["display_name"]}</span>', unsafe_allow_html=True)
                    
                    st.markdown(f"**Similarity Score**: `{res['score_percent']:.2f}%`")
                    st.progress(max(0.0, min(1.0, float(res["score"]))))
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)

        # Top Candidate Breakdown Expander
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🔬 Detailed Candidate Rankings")
        
        for res in face_results:
            with st.expander(f"Top Candidate Match Breakdown for Face #{res['face_index']} ({res['display_name']})", expanded=True):
                rank_cols = st.columns(5)
                for r_idx, (p_key, p_score) in enumerate(res["ranked"][:5]):
                    p_name = CLASS_NAMES.get(p_key, p_key)
                    p_emoji = PLAYER_EMOJIS.get(p_key, "🏏")
                    with rank_cols[r_idx]:
                        st.metric(
                            label=f"#{r_idx+1} {p_emoji} {p_name}",
                            value=f"{p_score*100:.1f}%"
                        )
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 Upload an image, capture with your webcam, or select a sample preset above to begin identification.")