# CAPTCHA 인식 CNN 모델

이 프로젝트는 이미지 기반의 CAPTCHA를 딥러닝으로 인식하는 간단한 예제입니다. 모델은 CNN 기반으로 구성했으며, 이미지의 각 글자를 다중 출력(Multi-Output Classification) 방식으로 예측합니다.

### 본 프로젝트는 [kaggle](https://www.kaggle.com/)의 [CAPTCHA Dataset](https://www.kaggle.com/datasets/parsasam/captcha-dataset)을 사용합니다.

---

## 📁 프로젝트 구조

```
captcha_cnn/
│
├── train.py                 # CNN 모델 학습 코드
├── predict.py               # 폴더 내 이미지 CAPTCHA 예측 코드
├── evaluate.py              # 랜덤 10,000개 CAPTCHA 정확도 평가 코드
├── utils.py                 # 데이터 전처리 및 문자셋 유틸리티
├── test_captcha/            # predict.py 에서 사용되는 평가용 이미지 저장 폴더
├── dataset/                 # 학습 및 평가용 이미지 저장 폴더
├── captcha_model.h5         # 학습 완료된 모델
```

---

## 💻 설치 및 환경

이 프로젝트는 **Anaconda 환경 사용을 권장**합니다.

### Conda 가상환경 생성 및 라이브러리 설치

```bash
python=3.5
install
tensorflow
opencv-python
numpy
tqdm
matplotlib
kreas
```
---

## 🧪 학습 방법

이미지 파일이 저장된 `dataset/` 폴더를 준비한 후, 다음 명령어로 학습을 시작하세요:

```bash
python train.py
```

- 학습 진행률은 `tqdm`로 표시됩니다.
- 학습이 완료되면 `captcha_model.h5` 파일로 저장됩니다.

---

## 🔍 CAPTCHA 예측

학습된 모델을 사용하여 `test_captcha/` 폴더 내 이미지를 예측하려면:

```bash
python predict.py
```

예시 출력:

```
1a1SZ.jpg → 예측된 캡챠: 1a1SZ
0rT9Q.jpg → 예측된 캡챠: 0rT9Q
...
```

---

## 📊 모델 정확도 평가

CAPTCHA 전체 문자열이 정확히 예측된 경우만 정답으로 판단하며, 평가에 사용되는 이미지는 `dataset/`에서 무작위로 10,000개 추출됩니다.

```bash
python eval.py
```

예시 출력:

```
[INFO] 평가용 이미지 10000개 로드 완료.
100%|██████████████████████████████████████| 10000/10000 [00:12<00:00, 802.34it/s]
[RESULT] 전체 정답률: 92.41%
```

---

## 🔤 문자셋 정보

- 지원 문자: `A-Z`, `a-z`, `0-9`
- 총 문자 클래스 수: 62개 (`CHARACTERS = string.ascii_letters + string.digits`)

---

## ⚠️ 주의사항

- 모든 이미지 파일 이름은 정답 문자열이어야 하며 `.jpg` 확장자를 가져야 합니다.
  - 예시: `a1B9Z.jpg` → 라벨은 `a1B9Z`
- 이미지 크기는 자동으로 `(100, 40)`으로 조정됩니다.
- 학습 데이터의 품질이 모델 정확도에 직접적인 영향을 줍니다.

---

## 🧾 각 파일 설명

| 파일명             | 설명 |
|------------------|------|
| `train.py`        | CNN 모델 구성 및 학습 실행 스크립트 |
| `predict.py`      | 폴더 내 `.jpg` 이미지에 대해 CAPTCHA 예측 수행 |
| `evaluate.py`         | 모델의 정답률 평가 (10,000개 랜덤 이미지) |
| `utils.py`        | 이미지 로딩, 라벨 인코딩 및 문자셋 정의 |

---

## 🧠 모델 구성 요약

```python
inputs = Input(shape=(40, 100, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

outputs = [Dense(62, activation='softmax', name=f'char_{i}')(x) for i in range(5)]
```

- 5개의 문자를 각각 Softmax로 예측
- 출력은 `[output1, output2, ..., output5]` 형태로 나옴

---

## 🧩 기타

- 본 프로젝트는 텍스트 CAPTCHA 인식에 적합하며, 회전, 왜곡, 배경이 복잡한 CAPTCHA에 대해서는 추가 전처리 및 augmentation이 필요할 수 있습니다.
- 더 나은 성능을 원한다면 CNN 구조를 deeper하게 변경하거나, LSTM/Attention을 추가해 hybrid 모델로 확장할 수도 있습니다.

---

## ❓ 문의

추가 문의는 [이슈](https://github.com/eombridge/captcha_cnn/issues)나 [연락처](https://eombridge.com)를 통해 남겨주세요.
