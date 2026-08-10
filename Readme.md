# 🏏 Indian Cricket Face Identification System

A Deep Learning based face identification system that recognizes Indian cricket players from an uploaded image.

The system uses OpenCV Face Detection, SFace Face Recognition embeddings and similarity matching to identify the closest matching player.

---

# 📌 Project Overview

This project can identify Indian cricket players by analyzing their facial features.

User uploads an image of a player and the system:

1. Detects the face
2. Aligns the face
3. Generates face embedding
4. Compares embedding with stored player embeddings
5. Returns the closest matching player with similarity percentage


---

# ⭐ Supported Players

Currently the system supports 15 Indian cricket players:

- Virat Kohli
- Rohit Sharma
- MS Dhoni
- Jasprit Bumrah
- Hardik Pandya
- KL Rahul
- Bhuvneshwar Kumar
- Dinesh Karthik
- Kedar Jadhav
- Kuldeep Yadav
- Mohammed Shami
- Ravindra Jadeja
- Shikhar Dhawan
- Vijay Shankar
- Yuzvendra Chahal


---

# 🚀 Features

## Face Detection

Uses OpenCV YuNet model to detect faces from images.

## Face Recognition

Uses OpenCV SFace model to generate 128 dimensional face embeddings.

## Similarity Matching

Compares uploaded face embedding with stored embeddings using cosine similarity.

## Web Interface

Streamlit based user interface.

## Multiple Player Support

Can recognize multiple cricket players.


---

# 🛠 Technologies Used

## Programming Language

- Python 3


## Libraries

- OpenCV
- TensorFlow
- NumPy
- Streamlit
- Scikit-learn


## AI Models

### YuNet

Used for face detection.

File:


### SFace

Used for face recognition.

File:
models/face_recognition_sface_2021dec.onnx

# 📂 Project Structure


indian-cricket-face-identification
│
├── app.py
│
├── dataset
│   ├── Virat_Kohli
│   ├── Rohit_Sharma
│   ├── MS_Dhoni
│   └── Other Players
│
├── processed_dataset
│
├── models
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   ├── face_embeddings.npz
│
├── src
│
│   ├── prepare_dataset.py
│   ├── create_embeddings.py
│   ├── predict.py
│   ├── evaluate_embeddings.py
│
├── test_images
│
├── requirements.txt
│
└── README.md

---

# ⚙️ Installation


Clone repository:

git clone YOUR_GITHUB_LINK


Go inside project:

cd indian-cricket-face-identification


Create virtual environment:

python -m venv venv


Activate environment:


Mac/Linux:

source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


---

# 📸 Adding New Player Images

If you want to add a new cricket player:

Example:

Adding:

Shubman Gill


Create folder:


mkdir dataset/Shubman_Gill


Add images:

Example:

dataset/Shubman_Gill/
001.jpg
002.jpg
003.jpg
...


Recommended:

Minimum:
30 images

Better:

50-100 images


Images should have:

- Clear face
- Different angles
- Different lighting
- Single person

## Demo

![App Demo](screenshots/home.png)


---

# 🔄 After Adding New Images

Every time new players/images are added, follow these steps:


## Step 1

Update player list in:

src/prepare_dataset.py


Example:

```python
players=[
"Virat_Kohli",
"Rohit_Sharma",
"New_Player"
]
Step 2
Delete old processed data:
rm -rf processed_dataset
Step 3
Prepare dataset:
python src/prepare_dataset.py
What happens:
Reads original images
Detects faces
Crops faces
Saves processed images
Output:
DATASET PROCESSING COMPLETE
Step 4
Create new embeddings:
Delete old embeddings:
rm -f models/face_embeddings.npz
Run:
python src/create_embeddings.py
What happens:
Reads processed faces
Generates 128 dimension embeddings
Saves player information
Output:
models/face_embeddings.npz
Step 5
Update Streamlit player list
Open:
app.py
Add new player name inside:
class_names={}
🧠 Model Workflow
Input Image

      |
      |
      v

Face Detection (YuNet)

      |
      |
      v

Face Alignment

      |
      |
      v

SFace Embedding

      |
      |
      v

Cosine Similarity Matching

      |
      |
      v

Player Name + Confidence

▶️ Running Application
Start Streamlit:
streamlit run app.py
Open:
http://localhost:8501
Upload image:
Example:
dhoni.jpg
Output:
Identification Result

MS Dhoni

Similarity:

67.35%

🧪 Testing
Run prediction:
python src/predict.py test_images/player.jpg
Example output:
Best Match:

Virat_Kohli

Similarity:

76.23%

📊 Evaluation
Run:
python src/evaluate_embeddings.py
The system calculates:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
📈 Current Performance
Testing Result:
Accuracy: 100%

Face Detection:
Successful

Players:
15

Embedding Dimension:
128

📝 Important Notes
More images improve accuracy.
Clear face images give better results.
Similar looking players may reduce similarity score.
Minimum 30 images per player recommended.


👨‍💻 Author
Nishant Kumar

GitHub:
https://github.com/Nishant20361