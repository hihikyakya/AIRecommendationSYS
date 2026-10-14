import pandas as pd
from pathlib import Path

from ..utils import util_A


def solve(dataset:str|Path):
    # A1-1
    df_movies = util_A.parse_df_movie_len(dataset,"movies")
    df_ratings = util_A.parse_df_movie_len(dataset,"ratings")
    df_users = util_A.parse_df_movie_len(dataset,"users")

    df_users_data=util_A.attach_to_user(df_users, df_ratings, df_movies)
    print(f"유저 데이터 컬럼 목록: {df_users_data.columns}")
    df_items_metadata=util_A.attach_to_item(df_users, df_ratings, df_movies)
    print(f"아이템 메타데이터 컬럼 목록: {df_items_metadata.columns}")

    # A1-2
    print(df_ratings.head(n=5))

    return df_users_data, df_items_metadata