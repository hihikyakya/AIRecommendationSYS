import pandas as pd
from pathlib import Path
from typing import Literal

def parse_df_movie_len(dataset: str | Path, choice: Literal["movies", "ratings", "users"]) -> pd.DataFrame:
    path = Path(dataset)
    if choice == "ratings":
        return pd.read_csv(
            path / "ratings.dat",
            sep="::",
            engine="python",
            names=["user", "item", "rating", "ts"],
        )
    elif choice == "users":
        return pd.read_csv(
            path / "users.dat",
            sep="::",
            engine="python",
            names=["user", "gender", "age", "occupation", "zip"],
        )
    elif choice == "movies":
        return pd.read_csv(
            path / "movies.dat",
            sep="::",
            engine="python",
            names=["item", "title", "genres"],
            encoding="latin-1",
        )
    else:
        raise ValueError(f"Unknown choice: {choice}")

def attach_to_user(df_users: pd.DataFrame, df_ratings: pd.DataFrame, df_movies: pd.DataFrame) -> pd.DataFrame:
    # 유저 정보를 중심으로 상호작용(ratings) 및 영화(movies) 정보를 결합 (double join)
    df_merged = df_ratings.merge(df_users, on="user", how="left").merge(df_movies, on="item", how="left")
    return df_merged

def attach_to_item(df_users: pd.DataFrame, df_ratings: pd.DataFrame, df_movies: pd.DataFrame) -> pd.DataFrame:
    # 아이템 정보를 중심으로 상호작용(ratings) 및 유저(users) 정보를 결합
    df_merged = df_movies.merge(df_ratings, on="item", how="left").merge(df_users, on="user", how="left")
    return df_merged
