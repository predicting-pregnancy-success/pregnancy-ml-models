import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 상수
HIGH_MISSING_COLS = [
    '난자 해동 경과일', 'PGS 시술 여부', 'PGD 시술 여부',
    '착상 전 유전 검사 사용 여부', '임신 시도 또는 마지막 임신 경과 연수',
    '배아 해동 경과일'
]

MEDIUM_MISSING_COLS = [
    '난자 채취 경과일', '난자 혼합 경과일', '배아 이식 경과일'
]

CATEGORICAL_COLS = [
    '시술 시기 코드', '시술 유형', '특정 시술 유형', '배란 유도 유형',
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


def drop_high_missing(df: pd.DataFrame) -> pd.DataFrame:
    """결측률 80% 이상 컬럼 제거"""
    cols_to_drop = [c for c in HIGH_MISSING_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def encode_count_cols(df: pd.DataFrame) -> pd.DataFrame:
    """횟수 컬럼 숫자로 변환"""
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].map(COUNT_MAP)
    return df


def encode_donor_age(df: pd.DataFrame) -> pd.DataFrame:
    """기증자 나이 컬럼 변환"""
    for col in ['난자 기증자 나이', '정자 기증자 나이']:
        if col in df.columns:
            df[col] = df[col].map(DONOR_AGE_MAP)
    return df


def handle_medium_missing(df: pd.DataFrame, medians: dict = None) -> tuple:
    """중간 결측 컬럼: 결측 지시 변수 추가 후 중앙값 대체"""
    if medians is None:
        medians = {col: df[col].median() for col in MEDIUM_MISSING_COLS if col in df.columns}

    for col in MEDIUM_MISSING_COLS:
        if col not in df.columns:
            continue
        df[f'{col}_결측'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(medians[col])

    return df, medians


def handle_numeric_missing(df: pd.DataFrame, medians: dict = None) -> tuple:
    """나머지 수치형 컬럼 결측 중앙값 대체"""
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
    """범주형 컬럼 LabelEncoding"""
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


def scale_features(df: pd.DataFrame, scaler: StandardScaler = None,
                   exclude_cols: list = None) -> tuple:
    """수치형 컬럼 표준화 (선형 모델 사용 시)"""
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


def preprocess(df: pd.DataFrame, medians: dict = None, encoders: dict = None,
               scaler: StandardScaler = None, scale: bool = False) -> pd.DataFrame:
    df = df.copy()
    df = drop_high_missing(df)
    df = encode_count_cols(df)
    df = encode_donor_age(df)
    df, medians = handle_medium_missing(df, medians)
    df, medians = handle_numeric_missing(df, medians)
    df, encoders = encode_categoricals(df, encoders)

    if scale:
        df, scaler = scale_features(df, scaler)

    return df