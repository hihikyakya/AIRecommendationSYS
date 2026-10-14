import pandas as pd
from sklearn.model_selection import train_test_split


def random_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """sklearn의 train_test_split을 활용한 무작위 8:2 분할."""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df.copy(), test_df.copy()


def sequential_train_test_split(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """전체 평가를 시간순(ts)으로 정렬 후 80% 지점을 경계로 분할 (전역 시간 분할)."""
    df_sorted = df.sort_values("ts").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    boundary_ts = df_sorted.iloc[split_idx]["ts"]

    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    return train_df, test_df, boundary_ts


def sequential_train_test_split_groupby(
    df: pd.DataFrame, by: str = "user", train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """유저마다 시간순(ts) 앞 80% / 뒤 20%로 분할 (유저별 시간 분할)."""
    df_sorted = df.sort_values([by, "ts"]).copy()
    user_rank = df_sorted.groupby(by).cumcount()
    user_total = df_sorted.groupby(by)["item"].transform("count")

    is_train = user_rank < (user_total * train_ratio).astype(int)
    train_df = df_sorted[is_train].copy()
    test_df = df_sorted[~is_train].copy()
    return train_df, test_df


def leave_one_out_split(
    df: pd.DataFrame, by: str = "user"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """유저별 시간순 마지막 1개 test, 직전 1개 valid, 나머지 train 분할."""
    df_sorted = df.sort_values([by, "ts"]).copy()
    user_total = df_sorted.groupby(by)["item"].transform("count")
    user_rank = df_sorted.groupby(by).cumcount()

    is_test = user_rank == (user_total - 1)
    is_valid = user_rank == (user_total - 2)
    is_train = ~(is_test | is_valid)

    test_df = df_sorted[is_test].copy()
    valid_df = df_sorted[is_valid].copy()
    train_df = df_sorted[is_train].copy()
    return train_df, valid_df, test_df