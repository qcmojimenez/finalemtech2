!pip install --upgrade pip
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.metrics import BinaryAccuracy
from keras.preprocessing import image
import numpy as np
import streamlit as st

model.save(os.path.join(save_dir, 'best_model.h5'))
loaded_model = load_model(os.path.join(save_dir, 'best_model.h5'))
st.title("Emotion Detection Streamlit App")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:

    img = image.load_img(uploaded_file, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = loaded_model.predict(img_array)
    st.write(f"Predicted Probability: {prediction[0]}")
    st.write(f"Predicted Class: {round(prediction[0][0])}")
