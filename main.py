import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model only once (optional: improve performance)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('densenet201_model.keras')
model = load_model()

# TensorFlow model prediction
def model_prediction(test_image):
    # Load and preprocess the uploaded image
    image = Image.open(test_image).convert("RGB")  # Ensure RGB
    image = image.resize((224, 224))               # Match model input size
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255.0                  # Normalize
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    # Predict
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = prediction[0][result_index]
    return result_index, confidence

#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("SELECT PAGE",["HOME","DISEASE AND PESTICIDE"])

#home
if(app_mode=="HOME"):
    image_path = "2.png"
    st.image(image_path,use_container_width=True) 
    st.write("")  # Adds a blank line 
    st.markdown("""
    Welcome to the Leaf Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying mango leaf diseases efficiently. Upload an image of a mango leaf, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease and Pesticide** page and upload an image of a leaf with suspected diseases.
    2. **Analysis:** Our system will process the image using deep learning algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes deep learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease and Pesticide** page in the sidebar to upload an image and experience the power of our Mango Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)
    
#prediction
elif app_mode == "DISEASE AND PESTICIDE":
    st.markdown(
    """
    <div style="
        background: linear-gradient(to right, #EBDF37, #ED6E61);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    ">
        <h1 style="color: white; margin: 0;">DISEASE DETECTION AND PESTICIDE SUGGESTION</h1>
    </div>
    """,
    unsafe_allow_html=True
    )
    st.write("")  # Adds a blank line
    test_image = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])
    if test_image is not None:
        if st.button("Show Image",use_container_width=True):
            st.image(test_image,use_container_width=True)
        if st.button("Detect",use_container_width=True):
            st.markdown(
                "<h3 style='text-align: center; font-weight: bold;'>Model prediction</h3>",
                unsafe_allow_html=True
            )
            result_index, confidence = model_prediction(test_image)
            class_names = ['Anthracnose', 'Grey Blight', 'Healthy', 'Red Rust', 'Sooty Mould']
            predicted_class = class_names[result_index]
            st.success(f"**Disease Detected** : {predicted_class}")
            # Pesticide recommendation
            pesticide_dict = {
                'Anthracnose': (
                    "1. Spraying of Kavach or Chlorothalonil (2%) or Carbendazim.\n\n"
                    "OR\n\n"
                    "2. Hot water treatment: Dip in hot water at 55°C for 15 minutes."
                ),
                'Grey Blight': (
                    "Spraying of Mancozeb or Carbendazim (0.1% to 0.2%)."
                ),
                'Healthy': (
                    "No pesticide needed."
                ),
                'Red Rust': (
                    "Spraying of Bordeaux mixture or Copper Oxychloride (0.2%).\n\n"
                    "Preparation: Mix 2 g in 1 L of water."
                ),
                'Sooty Mould': (
                    "1. Spraying of insecticide like Monocrotophos or Methyl Demeton.\n\n"
                    "OR\n\n"
                    "2. Spraying of starch solution.\n\n"
                    "   Preparation: Boil 1 kg of starch in water and dilute with 2 L of water."
                )
            }
            pesticide = pesticide_dict.get(predicted_class, 'No recommendation available')
            st.info(f"**Suggested Pesticide** : {pesticide}")
