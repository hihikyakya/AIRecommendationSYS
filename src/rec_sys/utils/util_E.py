
import numpy as np
import pandas as pd

from .util_D import get_density, get_gini_index


def k_core_filtering(df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, int]:
    """반복 제거 방식의 k-core 필터링을 수행하여 (필터링된 DataFrame, 반복 횟수) 반환."""
    df_curr = df[["user", "item", "rating"]].copy()
    iterations = 0

    while True:
        iterations += 1
        prev_len = len(df_curr)

        # 유저별 평가 수 계산
        user_counts = df_curr.groupby("user")["item"].transform("count")
        # 영화별 평가 수 계산
        item_counts = df_curr.groupby("item")["user"].transform("count")

        mask = (user_counts >= k) & (item_counts >= k)
        df_curr = df_curr[mask]

        if len(df_curr) == prev_len:
            break

    return df_curr, iterations


def reduction_filtering(
    user_stats: pd.DataFrame,
    global_mean: float,
    lambdas: list[float] = [5.0, 10.0, 25.0],
) -> pd.DataFrame:
    """축소 보정(Shrinkage / Empirical Bayes Smoothing)을 적용하여 보정 전후 평균 평점을 계산."""
    res_df = user_stats.copy()
    n = res_df["eval_count"]
    r_mean = res_df["user_mean"]

    for lmb in lambdas:
        col_name = f"보정후(λ={int(lmb) if lmb.is_integer() else lmb})"
        res_df[col_name] = (n / (n + lmb)) * r_mean + (lmb / (n + lmb)) * global_mean

    return res_df