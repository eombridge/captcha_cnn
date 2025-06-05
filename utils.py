# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import string

# 대소문자 + 숫자 포함된 문자셋
CHARACTERS = string.ascii_letters + string.digits
CHAR_SET_LEN = len(CHARACTERS)

def encode_label(label):
    encoded = []
    for char in label:
        if char not in CHARACTERS:
            print("[무시됨] 알 수 없는 문자: {} (전체 라벨: {})".format(char, label))
            return None
        one_hot = np.zeros(CHAR_SET_LEN)
        idx = CHARACTERS.index(char)
        one_hot[idx] = 1.0
        encoded.append(one_hot)
    return np.array(encoded)

def load_data(data_dir, captcha_length=5, img_size=(100, 40)):
    print("[INFO] 데이터 로딩 시작...")

    X, y = [], []
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]

    for i, filename in enumerate(files):
        label = filename.split(".")[0]
        if len(label) != captcha_length:
            continue

        path = os.path.join(data_dir, filename)
        image = cv2.imread(path)
        if image is None:
            print("[경고] 이미지 로딩 실패: {}".format(filename))
            continue

        # BGR -> RGB 변환작업
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_size)
        image = image.astype("float32") / 255.0

        encoded = encode_label(label)
        if encoded is None:
            continue

        X.append(image)
        y.append(encoded)

        if i % 1000 == 0 and i != 0:
            print("[INFO] {}개 이미지 로딩 중...".format(i))

    print("[INFO] 데이터 로딩 완료! 총 {}개 이미지".format(len(X)))
    return np.array(X), np.array(y)

# CHARACTERS, CHAR_SET_LEN도 다른 곳에서 import할 수 있도록 노출
__all__ = ['load_data', 'encode_label', 'CHARACTERS', 'CHAR_SET_LEN']
