import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import time

st.set_page_config(page_title="Hand Gesture Recognition", layout="centered")

# Load model and labels
model = tf.keras.models.load_model('model.h5')
with open("label_dict.pkl", "rb") as f:
    label_dict = pickle.load(f)
reverse_dict = {v: k for k, v in label_dict.items()}


# Function to preprocess image from webcam
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 64, 64, 1)
    return reshaped


# Predict gesture from image array
def predict_gesture(model, img: np.ndarray) -> str:
    pred = model.predict(img)
    class_id = np.argmax(pred)
    return reverse_dict[class_id]


# Predict gesture from uploaded image
def predict_image(img: np.ndarray) -> str:
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        pass
    else:
        raise ValueError("Unsupported image shape")

    img = cv2.resize(img, (64, 64)).reshape(1, 64, 64, 1) / 255.0
    pred = model.predict(img)
    class_id = np.argmax(pred)
    return reverse_dict[class_id]


# UI
st.title("🤖 Hand Gesture Recognition")
st.markdown("Upload a hand gesture image **or** use your webcam for real-time prediction.")

tab1, tab2 = st.tabs(["📷 Upload Image", "🎥 Webcam"])

# Tab 1: Upload Image
with tab1:
    uploaded = st.file_uploader("Upload a hand gesture image", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        label = predict_image(np.array(img))
        st.success(f"🧠 Predicted Gesture: **{label}**")

# Tab 2: Webcam Mode
with tab2:
    run = st.checkbox("✅ Start Webcam")
    FRAME_WINDOW = st.image([])
    label_placeholder = st.empty()

    cap = None

    if run:
        cap = cv2.VideoCapture(0)
        st.info("Webcam started. Showing live predictions...")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to access webcam.")
                break

            processed = preprocess_frame(frame)
            label = predict_gesture(model, processed)

            # Show prediction on frame
            cv2.putText(frame, f'Prediction: {label}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display frame and label
            FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)
            label_placeholder.markdown(
                f"<h3 style='text-align: center;'>🧠 Prediction: {label}</h3>",
                unsafe_allow_html=True
            )

            time.sleep(0.1)  # Slow down loop a little

    else:
        st.info("☝️ Click the checkbox to start webcam.")
        if cap:
            cap.release()
