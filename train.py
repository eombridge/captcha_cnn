import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from utils import load_data, CHAR_SET_LEN
from tqdm.keras import TqdmCallback

# 설정
captcha_length = 5
img_size = (100, 40)

# 데이터 로딩
X, y = load_data("dataset", captcha_length=captcha_length, img_size=img_size)
y = [y[:, i, :] for i in range(captcha_length)]  # (N, 5, 62) → 5개 라벨로 분리

# 모델 생성
def build_improved_model(input_shape, captcha_length, char_set_len):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = [Dense(char_set_len, activation='softmax', name='char_{}'.format(i))(x) for i in range(captcha_length)]
    return Model(inputs=inputs, outputs=outputs)

model = build_improved_model(
    input_shape=(img_size[1], img_size[0], 3),
    captcha_length=captcha_length,
    char_set_len=CHAR_SET_LEN
)

# 모델 컴파일
model.compile(
    loss=['categorical_crossentropy'] * captcha_length,
    optimizer='adam',
    metrics=['accuracy'] * captcha_length
)

# 콜백 정의 (버전 호환성 대응: restore_best_weights 제거)
early_stop = EarlyStopping(patience=3, monitor='val_loss')  # <- 수정됨
reduce_lr = ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss', verbose=1)

# 학습
model.fit(
    X, y,
    batch_size=64,
    epochs=30,
    validation_split=0.2,
    callbacks=[TqdmCallback(verbose=1), early_stop, reduce_lr, checkpoint],
    verbose=0
)

# 모델 저장
model.save("captcha_model.h5")
print("[INFO] 모델 학습 완료 및 저장됨: captcha_model(simple).h5")
