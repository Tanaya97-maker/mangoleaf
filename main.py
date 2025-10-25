import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import gdown
import os

# Load model only once using Streamlit's cache
@st.cache_resource
def load_model():
    model_path = "densenet201_model.keras"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1p-V0imW_ORloHlWZgFnjyCdpf69r8KJX"
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

# Load the model
model = load_model()

# Function to make predictions
def model_prediction(test_image):
    try:
        image = Image.open(test_image).convert("RGB")
        image = image.resize((224, 224))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = input_arr / 255.0
        input_arr = np.expand_dims(input_arr, axis=0)

        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        confidence = prediction[0][result_index]
        return result_index, confidence
    except UnidentifiedImageError:
        return None, 0.0
    except Exception:
        return None, 0.0

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("SELECT PAGE", ["HOME", "DISEASE AND PESTICIDE"])

# Home Page
if app_mode == "HOME":
    st.image("2.png", use_container_width=True)
    st.markdown("""
    Welcome to the Leaf Disease Recognition System! 🌿🔍

    Our mission is to help in identifying mango leaf diseases efficiently. Upload an image of a mango leaf, and our system will analyze it to detect any signs of diseases.

    ### How It Works
    1. **Upload Image:** Go to the **Disease and Pesticide** page and upload an image of a mango leaf.
    2. **Analysis:** Deep learning algorithms analyze the image.
    3. **Results:** View predicted disease and suggested pesticide.

    ### Why Choose Us?
    - **Accuracy:** Deep learning-based model
    - **User-Friendly:** Simple interface
    - **Fast:** Results in seconds

    ### Start
    Click **Disease and Pesticide** from the sidebar to upload and analyze a leaf image.
    """)

# Disease Detection Page
elif app_mode == "DISEASE AND PESTICIDE":
    st.markdown("""
    <div style="
        background: linear-gradient(to right, #EBDF37, #ED6E61);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;">
        <h1 style="color: white; margin: 0;">DISEASE DETECTION AND PESTICIDE SUGGESTION</h1>
    </div>
    """, unsafe_allow_html=True)

    test_image = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        if st.button("Show Image", use_container_width=True):
            try:
                st.image(test_image, use_container_width=True)
            except:
                st.error("⚠️ Unable to display image. Please upload a valid image.")

        if st.button("Detect", use_container_width=True):
            st.markdown("<h3 style='text-align: center; font-weight: bold;'>Model prediction</h3>", unsafe_allow_html=True)

            result_index, confidence = model_prediction(test_image)

            class_names = ['Anthracnose', 'Grey Blight', 'Healthy', 'Red Rust', 'Sooty Mould']
            confidence_threshold = 0.80  

            if result_index is None or confidence < confidence_threshold:
                st.error("⚠️ Not a proper mango leaf image. Please upload a clear image of a mango leaf.")
            else:
                predicted_class = class_names[result_index]
                st.success(f"**Disease Detected** : {predicted_class}")

                pesticide_dict = {
                    'Anthracnose': (
                        "1. Spray Kavach or Chlorothalonil (2%) or Carbendazim.\n\n"
                        "OR\n\n"
                        "2. Hot water treatment: Dip in hot water at 55°C for 15 minutes."
                    ),
                    'Grey Blight': (
                        "Spray Mancozeb or Carbendazim (0.1% to 0.2%)."
                    ),
                    'Healthy': (
                        "No pesticide needed."
                    ),
                    'Red Rust': (
                        "Spray Bordeaux mixture or Copper Oxychloride (0.2%).\n\n"
                        "Preparation: Mix 2 g in 1 L of water."
                    ),
                    'Sooty Mould': (
                        "1. Spray insecticide like Monocrotophos or Methyl Demeton.\n\n"
                        "OR\n\n"
                        "2. Spray starch solution.\n\n"
                        "   Preparation: Boil 1 kg of starch in water and dilute with 2 L of water."
                    )
                }

                pesticide = pesticide_dict.get(predicted_class, "No recommendation available")
                st.info(f"**Suggested Pesticide** : {pesticide}")
