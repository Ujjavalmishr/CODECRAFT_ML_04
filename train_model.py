from utils import load_data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle
import os

# 🔁 Path to your extracted folder
DATA_PATH = 'hand_gesture_recognition'

# 📦 Load Data
(X_train, X_test, y_train, y_test), label_dict = load_data(DATA_PATH)

# 🧠 Sanity Check
if len(X_train) == 0:
    raise ValueError("❌ No training samples found! Check your DATA_PATH.")

if len(label_dict) == 0:
    raise ValueError("❌ No labels found! Check dataset format.")

# 🔢 One-hot encode labels
y_train = to_categorical(y_train, num_classes=len(label_dict))
y_test = to_categorical(y_test, num_classes=len(label_dict))

print(f"✅ Loaded {len(X_train)} training samples and {len(X_test)} test samples.")
print(f"✅ Number of gesture classes: {len(label_dict)}")

# 🧠 Build Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_dict), activation='softmax')  # Dynamic output
])

# ⚙️ Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🚀 Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# 💾 Save model and label mapping
model.save('model.h5')
with open('label_dict.pkl', 'wb') as f:
    pickle.dump(label_dict, f)

print("✅ Model saved as 'model.h5' and labels as 'label_dict.pkl'")
