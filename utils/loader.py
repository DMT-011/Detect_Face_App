import os, pickle
import cv2
import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_haar_cascade():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(path)
    if face_cascade.empty():
        st.error("Không thể load Haar Cascade")
        return None
    return face_cascade

@st.cache_resource
def load_trained_cnn_model():
    model_file = "models/cnn_face_lfw_gray.keras"
    if os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file)
        return model, model_file
    return None, None
