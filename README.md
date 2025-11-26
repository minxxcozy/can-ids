# can-ids
> CAN 로그를 이용해 다음 두 단계를 통해 공격을 탐지하는 IDS를 구현

☑️ **1단계** : Binary Classification
* Normal vs Attack

☑️ **2단계** : Attack Classification
* Attack이 맞다고 판단된 window에 대해 공격 분류


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
source .venv/bin/activate
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
├── ids/
│   ├── io_utils.py          # CSV 컬럼 파서
│   ├── windowing.py         # Fixed-window 생성
│   ├── features.py          # Δt, entropy, rate 기반 feature 생성
│   ├── normal_profile.py    # Normal window 통계 생성
│
├── models/
│   ├── train_binary.py        # Step1: Normal vs Attack 학습
│   ├── train_attack_multi.py  # Step2: Attack 4-class 학습
│   ├── predict.py             # 최종 예측 수행
│
└── data/
    ├── ids_train.csv
    └── ids_test.csv
```

## 🔍 4. CSV 데이터 준비
실제 환경에서 받은 CSV 파일을 : 
```bash
data/your_dataset.csv
```
이 위치에 넣습니다.

## ✅ 5. 모델 실행
### 1️⃣ Normal vs Attack 이진 모델 학습
* 결과 파일: `models/binary.pkl`
```bash
python -m models.train_binary --csv data/autohack2025_train.csv --window-sec 0.2
```

### 2️⃣ Attack 전용 4-class 모델 학습
* 결과 파일: `models/attack.pkl`
* 라벨: Fuzzing / DoS / Spoofing / Replay
```bash
python -m models.train_attack_multi --csv data/autohack2025_train.csv --window-sec 0.2
```

### 3️⃣ 테스트 데이터 예측
**Linux / macOS / WSL**
```bash
python -m models.predict --csv data/autohack2025_test_data.csv --binary models/binary.pkl --attack models/attack.pkl --out data/autohack2025_test_pred.csv --window-sec 0.2
```
고정되어 있는 내부 값으로 자동 실행 : 
* window-sec = 0.2
* threshold = 0.55

**window-sec 값 변경 시**
```bash
python -m models.predict --csv data/autohack2025_test_data.csv --window-sec 0.5
```

**threshold 값 변경 시**
```bash
python -m models.predict --csv data/autohack2025_test_data.csv --threshold 0.4
```
