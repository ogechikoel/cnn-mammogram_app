from google.colab import drive
drive.mount('/content/drive')


import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
model_path = '/content/drive/MyDrive/CNN/mammogram_h5_model.h5'

model = load_cnn_model()


@st.cache_resource
def load_cnn_model():
    return load_model(model_path)

st.title("Image Classification with CNN")
st.write("Upload an image to get predictions from the CNN model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')  # Grayscale as your model expects
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess image
    img_size = (256, 256)  
    img_resized = img.resize(img_size)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    # Make prediction
    prediction = model.predict(img_array)[0][0]  # single sigmoid output

    # Threshold for classification (usually 0.5)
    threshold = 0.5
    if prediction >= threshold:
        predicted_class = 'Malignant'
    else:
        predicted_class = 'Benign'

    # Display prediction
    st.subheader("Prediction")
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Prediction Probability (Malignant): {prediction:.2f}")
