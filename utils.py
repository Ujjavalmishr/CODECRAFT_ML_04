import os
import cv2
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

def load_data(data_dir: str, img_size: Tuple[int, int] = (64, 64)):
    X, y = [], []
    label_dict = {}
    label_id = 0

    for person_dir in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_dir)
        if not os.path.isdir(person_path):
            continue
        for gesture_dir in os.listdir(person_path):
            gesture_path = os.path.join(person_path, gesture_dir)
            if not os.path.isdir(gesture_path):
                continue
            if gesture_dir not in label_dict:
                label_dict[gesture_dir] = label_id
                label_id += 1
            for img_name in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(label_dict[gesture_dir])

    X = np.array(X).reshape(-1, img_size[0], img_size[1], 1) / 255.0
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42), label_dict
