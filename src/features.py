import pandas as pd
import numpy as np


AGE_MAP = {
    '만18-34세': 0,
    '만35-37세': 1,
    '만38-39세': 2,
    '만40-42세': 3,
    '만43-44세': 4,
    '만45-50세': 5,
    '알 수 없음': -1
}


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    def safe_div(a, b):
        return np.where(b == 0, 0, a / b)

    df['배아_이식률'] = safe_div(df['이식된 배아 수'], df['총 생성 배아 수'])
    df['배아_저장률'] = safe_div(df['저장된 배아 수'], df['총 생성 배아 수'])
    df['난자_수정률'] = safe_div(df['혼합된 난자 수'], df['수집된 신선 난자 수'])
    df['미세주입_성공률'] = safe_div(df['미세주입에서 생성된 배아 수'], df['미세주입된 난자 수'])
    df['신선_난자_활용률'] = safe_div(df['혼합된 난자 수'], df['수집된 신선 난자 수'] + df['해동 난자 수'])

    return df


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    df['연령_그룹'] = df['시술 당시 나이'].map(AGE_MAP)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    # 연령_그룹이 추가된 이후에 호출되므로 수치형인 연령_그룹 사용
    df['연령_x_시술횟수'] = df['연령_그룹'] * df['총 시술 횟수']
    df['연령_x_이식배아수'] = df['연령_그룹'] * df['이식된 배아 수']
    df['연령_x_생성배아수'] = df['연령_그룹'] * df['총 생성 배아 수']

    return df


def add_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    df['임신_성공률_이력'] = df['총 임신 횟수'] / (df['총 시술 횟수'] + eps)
    df['출산_성공률_이력'] = df['총 출산 횟수'] / (df['총 임신 횟수'] + eps)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_ratio_features(df)
    df = add_age_features(df)
    df = add_interaction_features(df)
    df = add_cumulative_features(df)
    return df