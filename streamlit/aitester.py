import json
from os import environ as env
from urllib.parse import quote_plus, urlencode
from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from flask import Flask, redirect, render_template, session, url_for, jsonify,request,send_from_directory, abort
import os
import base64
from pymongo import MongoClient
from datetime import datetime, timedelta
import cv2
import numpy as np

from keras.models import load_model
import streamlit as st

model = load_model('fire_detection_model.keras')

# Streamlit app
st.set_page_config(page_title="Fire Detection", page_icon="🔥")

st.title("🔥 Fire Detection 🔥")
st.write("Upload an image to check the probability of fire in it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Display the image
    st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img_resized = cv2.resize(img, (224, 224)) / (255.0 * 1.5)
    
    # Run prediction
    pred = model.predict(np.expand_dims(img_resized, axis=0))
    pred_prob = float(pred[0][0])
    
    # Display the prediction
    st.subheader("Prediction")
    st.write(f"**Probability of fire: {pred_prob * 100:.2f}%**")
    
    # Add a progress bar
    progress = st.progress(0)
    for i in range(int(pred_prob * 100) + 1):
        progress.progress(i)
else:
    st.info("Please upload an image file.")

