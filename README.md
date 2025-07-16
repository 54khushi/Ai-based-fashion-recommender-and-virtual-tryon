
# 👗 AI-Based Fashion Recommender & Virtual Try-On System

An AI-powered fashion assistant that recommends visually similar clothing styles based on a user-uploaded image and enables users to virtually try them on using real-time pose detection and gesture control.

---

## 📌 Table of Contents

- [📖 About the Project](#about-the-project)  
- [✨ Features](#features)  
- [🛠 Tech Stack](#tech-stack)  
- [🚀 Installation](#installation)  
- [▶️ How to Run](#how-to-run)  
- [📁 Project Structure](#project-structure)  
- [📸 Screenshots](#screenshots)  
- [👩‍💻 Author](#author)  
- [📄 License](#license)

---

## 📖 About the Project

This intelligent fashion system combines computer vision and deep learning to:
- Recommend the most visually similar clothing items from a dataset
- Provide a live virtual try-on experience using a webcam
- Allow gesture-based outfit switching (no clicks required!)

---

## ✨ Features

- 🧠 ResNet50-based content similarity for fashion recommendations  
- 🖼 Real-time virtual try-on using MediaPipe Pose & Hands  
- 📤 Upload any clothing image to get top 5 similar suggestions  
- ✋ Hand gestures to switch try-on clothes (left/right)  
- ⚡ Precomputed embeddings for quick results  
- 📦 Clean and modular project structure  

---

## 🛠 Tech Stack

| Feature                  | Technology                        |
|--------------------------|------------------------------------|
| Image Feature Extraction | ResNet50 (TensorFlow / Keras)     |
| Similarity Matching      | scikit-learn Nearest Neighbors     |
| Virtual Try-On           | MediaPipe + OpenCV                 |
| Frontend UI              | Streamlit                          |
| Image Processing         | Pillow, NumPy                      |
| Storage                  | Pickle, Local Storage              |

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/54khushi/Ai-based-fashion-recommender-and-virtual-tryon
cd Ai-based-fashion-recommender-and-virtual-tryon

2. Create and Activate Virtual Environment
    python -m venv venv
    source venv/bin/activate        # On Linux/Mac
    venv\Scripts\activate           # On Windows
3. Install Dependencies
    pip install -r requirements.txt
    If requirements.txt is missing, install manually:
    pip install streamlit opencv-python mediapipe numpy scikit-learn tensorflow pillow tqdm

▶️ How to Run
🔹 Step 1: Generate Embeddings (Only Once)
    python app.py

    Extracts features from images/ folder
    Creates embeddings.pkl and filenames.pkl

🔹 Step 2: Run Recommender Interface (Streamlit App)
    streamlit run main.py
    Upload a clothing image

    View top 5 similar recommendations
    Click on 👚 Try On to select an item

🔹 Step 3: Start Virtual Try-On (Webcam)
    python trying.py

    Webcam opens with pose detection
    Hand gestures (left/right swipe) to switch outfits
    Shirt overlays are aligned on shoulder points   

📁 Project Structure
    ├── app.py                # Extracts image features
    ├── main.py               # Recommender system (Streamlit UI)
    ├── trying.py             # Real-time virtual try-on via webcam
    ├── test.py               # CLI-based image similarity test
    ├── images/               # Dataset of clothing images
    ├── tryon_images/         # PNG shirt overlays (transparent)
    ├── uploads/              # Uploaded images for recommendation
    ├── resources/            # UI buttons (left/right)
    ├── person/               # Optional: user photos
    ├── embeddings.pkl        # Saved feature vectors
    ├── filenames.pkl         # Saved image paths
    ├── selected_shirt.txt    # Stores try-on selection


 📸 Screenshots

### 🖼️ Recommender System (Streamlit UI)
![Recommender UI](screenshots/Screenshot1.png)
![Recommender UI](screenshots/Screenshot2.png)

### 👚 Virtual Try-On with Pose Detection

#### 📸 Try-On Example 1
![Try-On 1](screenshots/Screenshot3.png)




📄 License
Licensed under the MIT License
