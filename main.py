import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Ensure upload folder exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load feature list and filenames
try:
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
except FileNotFoundError:
    st.error("Embeddings or filenames not found. Make sure they are generated and present.")
    st.stop()

# Load ResNet50 model for feature extraction
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_model.trainable = False
model = tensorflow.keras.Sequential([
    resnet_model,
    GlobalMaxPooling2D()
])

st.title('👗 AI-Based Fashion Recommender & Try-On System')

# Save uploaded image to /uploads
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Feature extraction using pre-trained model
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Find top 5 similar items
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Streamlit interface
uploaded_file = st.file_uploader("📤 Upload a clothing image", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.subheader("✅ Uploaded Image")
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image')  # No resizing

        with st.spinner("🔍 Finding similar items..."):
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
            indices = recommend(features, feature_list)

        # Get recommended filenames
        recommended_image_names = [filenames[i] for i in indices[0]]

        st.subheader("🧠 Top 5 Recommendations")

        # Display recommended filenames
        st.write("🗂️ Recommended Image Names:")
        for name in recommended_image_names:
            st.text(name)

        # Show recommended images
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.image(filenames[indices[0][i]])  # original size
                if st.button(f"👚 Try On {i+1}", key=f"tryon_{i}"):
                    st.success(f"Try-On triggered for: {filenames[indices[0][i]]}")
    else:
        st.error("❌ Error occurred while saving the uploaded file.")

