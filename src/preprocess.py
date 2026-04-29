import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

HIGH_MISSING_COLS = [
    '난자 해동 경과일', 'PGS 시술 여부', 'PGD 시술 여부',
    '착상 전 유전 검사 사용 여부', '임신 시도 또는 마지막 임신 경과 연수',
    '배아 해동 경과일'
]

MEDIUM_MISSING_COLS = [
    '난자 채취 경과일', '난자 혼합 경과일', '배아 이식 경과일'
]

CATEGORICAL_COLS = [
    '시술 시기 코드',
    '시술 유형', '특정 시술 유형', '배란 유도 유형',
    '난자 출처', '정자 출처', '배아 생성 주요 이유'
]

TARGET_ENCODE_COLS = [
    '시술 유형', '특정 시술 유형', '배란 유도 유형',
    '난자 출처', '정자 출처', '배아 생성 주요 이유'
]

COUNT_COLS = [
    '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수',
    '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수',
    '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수'
]

COUNT_MAP = {
    '0회': 0, '1회': 1, '2회': 2, '3회': 3,
    '4회': 4, '5회': 5, '6회 이상': 6
}

DONOR_AGE_MAP = {
    '알 수 없음': -1,
    '만20세 이하': 0,
    '만21-25세': 1,
    '만26-30세': 2,
    '만31-35세': 3,
    '만36-40세': 4,
    '만41-45세': 5
}

# DI 시술 시 의미없는 배아/난자 관련 컬럼 → 0으로
DI_ZERO_COLS = [
    '총 생성 배아 수', '이식된 배아 수', '저장된 배아 수',
    '미세주입된 난자 수', '미세주입에서 생성된 배아 수',
    '미세주입 배아 이식 수', '미세주입 후 저장된 배아 수',
    '해동된 배아 수', '해동 난자 수', '수집된 신선 난자 수',
    '저장된 신선 난자 수', '혼합된 난자 수',
    '파트너 정자와 혼합된 난자 수', '기증자 정자와 혼합된 난자 수',
]

# MICE로 채울 컬럼 (배아/난자 수치, 서로 강하게 연관)
MICE_COLS = [
    '총 생성 배아 수', '이식된 배아 수', '저장된 배아 수',
    '미세주입된 난자 수', '미세주입에서 생성된 배아 수',
    '미세주입 배아 이식 수', '미세주입 후 저장된 배아 수',
    '해동된 배아 수', '해동 난자 수', '수집된 신선 난자 수',
    '저장된 신선 난자 수', '혼합된 난자 수',
    '파트너 정자와 혼합된 난자 수', '기증자 정자와 혼합된 난자 수',
]


def drop_high_missing(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in HIGH_MISSING_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def fill_di_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """DI 시술인 경우 배아/난자 관련 컬럼 결측치를 0으로"""
    di_mask = df['시술 유형'] == 'DI'
    for col in DI_ZERO_COLS:
        if col in df.columns:
            df.loc[di_mask, col] = df.loc[di_mask, col].fillna(0)
    return df


def encode_count_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].map(COUNT_MAP)
    return df


def encode_donor_age(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['난자 기증자 나이', '정자 기증자 나이']:
        if col in df.columns:
            df[col] = df[col].map(DONOR_AGE_MAP)
    return df


def add_domain_flags(df: pd.DataFrame) -> pd.DataFrame:
    if '배아 이식 경과일' in df.columns:
        df['배반포기_이식'] = (df['배아 이식 경과일'] == 5).astype(int)
    return df


def handle_medium_missing(df: pd.DataFrame, medians: dict = None) -> tuple:
    if medians is None:
        medians = {col: df[col].median() for col in MEDIUM_MISSING_COLS if col in df.columns}

    for col in MEDIUM_MISSING_COLS:
        if col not in df.columns:
            continue
        df[f'{col}_결측'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(medians[col])

    return df, medians


def handle_numeric_missing(df: pd.DataFrame, medians: dict = None) -> tuple:
    numeric_cols = [
        c for c in df.select_dtypes(include='number').columns
        if c != '임신 성공 여부'
    ]

    if medians is None:
        medians = {col: df[col].median() for col in numeric_cols}

    for col in numeric_cols:
        if col in medians:
            df[col] = df[col].fillna(medians[col])

    return df, medians


def encode_categoricals(df: pd.DataFrame, encoders: dict = None) -> tuple:
    if encoders is None:
        encoders = {}
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )

    return df, encoders


def apply_mice(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    mice_cols = [c for c in MICE_COLS if c in train.columns]

    imputer = IterativeImputer(random_state=42, max_iter=3)
    train_imputed = imputer.fit_transform(train[mice_cols])
    test_imputed  = imputer.transform(test[mice_cols])

    train[mice_cols] = train_imputed
    test[mice_cols]  = test_imputed

    return train, test


def scale_features(df: pd.DataFrame, scaler: StandardScaler = None,
                   exclude_cols: list = None) -> tuple:
    if exclude_cols is None:
        exclude_cols = ['임신 성공 여부', 'ID']

    scale_cols = [
        c for c in df.select_dtypes(include='number').columns
        if c not in exclude_cols and '_결측' not in c
    ]

    if scaler is None:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
    else:
        df[scale_cols] = scaler.transform(df[scale_cols])

    return df, scaler


def preprocess(
    train: pd.DataFrame,
    test: pd.DataFrame,
    medians: dict = None,
    encoders: dict = None,
    scaler: StandardScaler = None,
    scale: bool = False,
) -> tuple:
    """
    train, test 동시에 받아서 처리.
    MICE를 train 기준으로 학습 후 test에 적용하기 위해 두 개를 같이 받음.
    """
    train = train.copy()
    test  = test.copy()

    # 1. 고결측 컬럼 제거
    train = drop_high_missing(train)
    test  = drop_high_missing(test)

    # 2. DI 시술 결측치 0으로
    train = fill_di_zeros(train)
    test  = fill_di_zeros(test)

    # 3. MICE로 배아/난자 결측치 처리
    train, test = apply_mice(train, test)

    # 4. 횟수 컬럼 변환
    train = encode_count_cols(train)
    test  = encode_count_cols(test)

    # 5. 기증자 나이 변환
    train = encode_donor_age(train)
    test  = encode_donor_age(test)

    # 6. 도메인 플래그 추가
    train = add_domain_flags(train)
    test  = add_domain_flags(test)

    # 7. 중간 결측 처리 (median + 결측 지시 변수)
    train, medians = handle_medium_missing(train)
    test,  _       = handle_medium_missing(test, medians)

    # 8. 나머지 수치형 결측 처리
    train, medians = handle_numeric_missing(train)
    test,  _       = handle_numeric_missing(test, medians)

    # 9. 범주형 인코딩
    train, encoders = encode_categoricals(train)
    test,  _        = encode_categoricals(test, encoders)

    if scale:
        train, scaler = scale_features(train, scaler)
        test,  _      = scale_features(test, scaler)

    return train, test