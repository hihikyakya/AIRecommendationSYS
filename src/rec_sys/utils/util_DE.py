import numpy as np
import pandas as pd

def get_density(matrix_or_df: pd.DataFrame | np.ndarray) -> float:
    """유저-영화 행렬의 밀도(|Ω| / (|U| * |I|))를 계산."""
    if isinstance(matrix_or_df, pd.DataFrame):
        # 0이 아니거나 결측치가 아닌 유효 평가 칸 수
        if matrix_or_df.isna().any().any():
            observed = matrix_or_df.notna().sum().sum()
        else:
            observed = (matrix_or_df > 0).sum().sum()
        total_cells = matrix_or_df.shape[0] * matrix_or_df.shape[1]
    else:
        arr = np.array(matrix_or_df)
        observed = np.count_nonzero(~np.isnan(arr) & (arr > 0))
        total_cells = arr.size

    return float(observed / total_cells) if total_cells > 0 else 0.0


def get_gini_index(values: np.ndarray | pd.Series | list) -> float:
    """지니 계수(Gini Index)를 계산 (0 = 완전 평등, 1 = 완전 불평등)."""
    arr = np.array(values, dtype=np.float64).flatten()
    arr = arr[~np.isnan(arr)]
    arr = arr[arr >= 0]
    n = len(arr)
    if n == 0 or np.sum(arr) == 0:
        return 0.0
    arr = np.sort(arr)
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * arr)) / (n * np.sum(arr)) - (n + 1.0) / n
    return float(gini)