import sys
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf

from tensorflow.keras.preprocessing import image

CLASS_NAMES = [
    'Aster', 'Daisy', 'Iris', 'Lavender', 'Lily',
    'Marigold', 'Orchid', 'Poppy', 'Rose', 'Sunflower'
]

IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flower_model.keras")

def predict(img_path):
    if not os.path.exists(img_path):
        print(f"Loi: Khong tim thay anh '{img_path}'")
        return

    print("Dang load model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model da load thanh cong!")

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    print("Dang du doan...")
    preds = model.predict(img_array, verbose=0)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx] * 100

    print(f"Ket qua: {CLASS_NAMES[class_idx]} ({confidence:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Su dung: python predict.py <duong_dan_anh>")
        sys.exit(1)
    try:
        predict(sys.argv[1])
    except Exception as e:
        print(f"Loi: {e}")
