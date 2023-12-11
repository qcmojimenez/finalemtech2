import os
import streamlit as st

# Check if the required packages are installed, and install them if not
try:
    import keras
except ImportError:
    os.system("pip install keras")
try:
    import tensorflow
except ImportError:
    os.system("pip install tensorflow")
try:
    import pillow
except ImportError:
    os.system("pip install pillow")
try:
    import numpy
except ImportError:
    os.system("pip install numpy")
try:
    import matplotlib
except ImportError:
    os.system("pip install matplotlib")

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model_path = '/best_model.h5'
model = load_model(model_path)

# Streamlit UI
st.title("Emtech2 - Emotion Prediction App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    prediction = model.predict(img_array)

    # Display results
    predicted_class = "Happy" if prediction[0] >= 0.5 else "Sad"
    confidence = prediction[0] if predicted_class == "Happy" else 1 - prediction[0]
    confidence_scalar = float(confidence)

    st.image(img, caption=f'Predicted Class: {predicted_class} (Confidence: {confidence_scalar:.2f})', use_column_width=True)
