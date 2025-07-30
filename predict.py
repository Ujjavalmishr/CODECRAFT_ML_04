import cv2
import numpy as np
import tensorflow as tf
import pickle

model = tf.keras.models.load_model('model.h5')
with open("label_dict.pkl", "rb") as f:
    label_dict = pickle.load(f)

reverse_dict = {v: k for k, v in label_dict.items()}

def predict_image(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64)).reshape(1, 64, 64, 1) / 255.0
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    print("Predicted Gesture:", reverse_dict[class_id])

# Example
predict_image('images/00/01/01_palm_0.png')
