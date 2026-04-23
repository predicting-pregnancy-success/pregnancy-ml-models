### 📁 프로젝트 구조
```plaintext
PREGNANCY-ML-MODELS/
│
├── data/
│   ├── raw/                    # 원본 파일 그대로 보관
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── sample_submission.csv
│   │   └── 데이터_명세.xlsx
│   └── processed/              # 전처리 후 저장
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb            # 탐색적 데이터 분석
│   ├── 02_preprocessing.ipynb  # 전처리 실험
│   ├── 03_baseline.ipynb       # 베이스라인 모델
│   └── 04_tuning.ipynb         # 하이퍼파라미터 튜닝
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # 전처리 파이프라인
│   ├── features.py             # 피처 엔지니어링
│   ├── train.py                # 모델 학습 코드
│   ├── predict.py              # 예측 및 제출 생성
│   └── utils.py                # 공통 유틸리티
│
├── saved_models/               # 학습된 모델 저장 (.pkl, .joblib 등)
│
├── submissions/                # 제출 파일 버전 관리
│   └── submission_v1.csv
│
├── requirements.txt
└── README.md
```