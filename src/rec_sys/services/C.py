import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _solve1(df_items_metadata:pd.DataFrame):

    # TODO: C1-1은 대충 이런 느낌?
    df_product_movie_genres=df_items_metadata.groupby("Movies")["Genres"].sum()
    print(df_product_movie_genres.shape[0] * df_product_movie_genres.shape[1])

    # TODO: C1-2
    df_items_metadata["description"]=

def _solve2(df_items_metadata:pd.DataFrame):
    # TODO: C2
    df_items_metadata=parse_timestamp(df_items_metadata)
    # TODO more...

def solve(df_items_metadata: pd.DataFrame):
    _solve1(df_items_metadata)
    _solve2(df_items_metadata)