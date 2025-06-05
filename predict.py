import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from utils import CHARACTERS

def decode_prediction(preds):
    result = ''
    for pred in preds:
        index = np.argmax(pred)
        result += CHARACTERS[index]
    return result

# 모델 로드
model = load_model("captcha_model.h5")

# 예측할 이미지가 있는 폴더 경로
image_folder = "test_captcha"  # ← 이 폴더 안에 있는 이미지들 예측

# 폴더 안의 모든 JPG 이미지 처리
for filename in os.listdir(image_folder):
    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)

    if image is None:
        print("[오류] 이미지를 불러올 수 없음: {}".format(filename))
        continue

    image = cv2.resize(image, (100, 40))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    label = decode_prediction(preds)

    print("{} → 예측된 캡챠: {}".format(filename, label))
