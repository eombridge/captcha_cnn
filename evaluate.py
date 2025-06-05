import os
import numpy as np
import cv2
import random
from tensorflow.keras.models import load_model
from utils import CHARACTERS
from tqdm import tqdm

def decode_prediction(preds):
    result = ''
    for pred in preds:
        index = np.argmax(pred)
        result += CHARACTERS[index]
    return result

# 모델 로드
model = load_model("captcha_model.h5")

# 이미지 폴더
image_folder = "dataset"
all_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

# 무작위로 1만 개 선택
sample_files = random.sample(all_files, min(1000, len(all_files)))

# 정확도 집계
total = 0
correct = 0

for filename in tqdm(sample_files, desc="evaluating 1000 images"):
    label = filename.split(".")[0]
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)

    if image is None:
        continue

    image = cv2.resize(image, (100, 40))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    pred_label = decode_prediction(preds)

    total += 1
    if pred_label == label:
        correct += 1

# 최종 결과 출력
accuracy = (correct / total) * 100 if total > 0 else 0
print("\n평가한 이미지 수: {}".format(total))
print("정답 맞춘 수: {}".format(correct))
print("최종 정확도: {:.2f}%".format(accuracy))
