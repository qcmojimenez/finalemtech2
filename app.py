import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image  # Explicit import for Keras image preprocessing
from tensorflow.keras.models import load_model  # Explicit import for Keras model loading

# PIL import for image manipulation
from PIL import Image

# Load the trained model with updated path
model_path = 'best_model.h5'  # Update with the correct path
model = load_model(model_path)

# Define the class labels
class_names = ['happy', 'sad']

# Streamlit app
st.title('Happy or Sad Detection')

# Description
st.markdown("""
The concept of the project is based on the midterm exam that identifies the weather.
We applied a CNN model to train the model for detecting if the face is happy or sad.
You can upload a photo we reserved from the Google Drive link that we also submitted.
""")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image for the model
    img = image.load_img(uploaded_file, target_size=(64, 64))  # Use Keras image preprocessing
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_index]
    probability = np.max(prediction)

    # Display the prediction
    st.write(f'The predicted class is: {predicted_class}')
    st.write(f'The probability is: {probability}')
