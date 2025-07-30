import cv2
import numpy as np
import tensorflow as tf
import pickle

# ✅ Load model and label dictionary
print("[INFO] Loading model and label dictionary...")
model = tf.keras.models.load_model('model.h5')

with open('label_dict.pkl', 'rb') as f:
    label_dict = pickle.load(f)
reverse_dict = {v: k for k, v in label_dict.items()}

# 🔍 Prediction function
def predict_frame(frame: np.ndarray) -> str:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64)).reshape(1, 64, 64, 1) / 255.0
    prediction = model.predict(resized)
    class_id = np.argmax(prediction)
    return reverse_dict[class_id]

# 🎥 Initialize webcam
print("[INFO] Accessing webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Could not access webcam. Exiting.")
    exit()

print("[INFO] Press 'q' to quit.")

# 🔁 Live loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Failed to grab frame.")
        break

    # 🧠 Predict gesture
    try:
        gesture = predict_frame(frame)
    except Exception as e:
        gesture = "Error"
        print(f"Prediction error: {e}")

    # 🖼️ Display result on frame
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-Time Hand Gesture Recognition", frame)

    # ⏹️ Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🧹 Cleanup
cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam closed. Program ended.")
