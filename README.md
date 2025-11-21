# can-ids
> 다양한 형태의 CAN CSV 데이터를 자동으로 처리하고, 정적 (통계 기반) + 시계열 기반 (LSTM, 1D-CNN) 분석 모델을 모두 지원하는 **완성형 IDS 파이프라인**

## 🔧 1. 개발 환경 설정
### 1️⃣ 저장소 클론
```bash
git clone <YOUR_REPOSITORY_URL>
cd can-ids
```

### 2️⃣ 가상환경 생성 (필수)
```bash
python -m venv .venv
```

### 3️⃣ 가상환경 활성화
**Linux / macOS / WSL**
```bash
python3 -m venv .venv
```

**Windows PowerShell**
```bash
.\.venv\Scripts\Activate.ps1
```

**Windows CMD**
```bash
.venv\Scripts\activate
```
> ⚠️ (중요) 명령어 실행 시 항상 (.venv) 표시가 떠야 합니다.

## 📦 2. 의존성 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📁 3. 프로젝트 구조
```markdown
can-ids/
│
├── data/
│   └── .gitkeep              # CSV 파일을 여기에 넣으세요
│
├── ids/                      # 전처리 및 Feature 추출 코드
│   ├── config.py             # CSV 컬럼명 후보 설정
│   ├── io_utils.py           # CSV 컬럼 자동 탐지
│   ├── payload_utils.py      # hex → byte 변환, entropy 계산
│   ├── windowing.py          # 슬라이딩 윈도우 생성
│   ├── features.py           # Δt/Freq/Entropy 기반 Feature 생성
│   └── sequence_builder.py   # Δt 시퀀스 생성 (LSTM/CNN용)
│
├── models/                   # ML/DL 모델 학습 코드
│   ├── train_baseline.py     # RandomForest/XGBoost 학습
│   ├── lstm_model.py         # LSTM 시퀀스 모델
│   ├── cnn1d_model.py        # 1D-CNN 시퀀스 모델
│   └── tune_rf_xgb.py        # Optuna 튜닝
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 🔍 4. CSV 데이터 준비
실제 환경에서 받은 CSV 파일을 : 
```bash
data/your_dataset.csv
```
이 위치에 넣습니다.
CSV 컬럼명이 어떤 형식이든 자동으로 인식됩니다.

## ✅ 5. 모델 실행
### 🔵 베이스라인 모델 실행
> RandomForest + XGBoost를 사용해 가장 빠르게 F1 점수를 확인할 수 있는 기본 파일입니다.

```bash
python -m models.train_baseline --csv data/your_dataset.csv --window-sec 1
```
이 스크립트는 다음을 자동으로 수행합니다 :
* CSV 컬럼 자동 탐지
* 슬라이딩 윈도우 생성
* Δt / ID 빈도 / Entropy Feature 생성
* RF & XGB 모델 학습
* F1 Score 출력

### 🟢 LSTM 기반 시퀀스 모델 실행
> 시간 흐름 (시계열 패턴)을 학습하기 위한 모델입니다.

**Δt 시퀀스용 LSTM 분류기**
```bash
python -m models.lstm_model \
  --csv data/your_dataset.csv \
  --window-size 32 \
  --epochs 10
```

**리플레이 공격 탐지 강화를 위한 멀티-피처 LSTM 모델**
```bash
python -m models.lstm_replay_model --csv data/my_can_log.csv
```

```bash
python -m models.lstm_replay_model \
    --csv data/my_can_log.csv \
    --window-size 32 \
    --step-size 16 \
    --epochs 10
```

### 🔴 1D-CNN 기반 시퀀스 모델 실행
> LSTM보다 더 빠르고 가벼운 모델입니다.

```bash
python -m models.cnn1d_model \
  --csv data/your_dataset.csv \
  --window-size 32 \
  --epochs 10
```

### 🟣 Optuna 하이퍼파라미터 튜닝
> Optuna 라이브러리를 활용한 튜닝 스크립트입니다.

**RandomForest 튜닝**
```bash
python -m models.tune_rf_xgb \
  --csv data/your_dataset.csv \
  --model rf \
  --n-trials 50
```

**XGBoost 튜닝**
```bash
python -m models.tune_rf_xgb \
  --csv data/your_dataset.csv \
  --model xgb \
  --n-trials 50
```
