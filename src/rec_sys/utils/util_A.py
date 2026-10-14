import pandas as pd
from pathlib import Path
from typing import Literal

def parse_df_movie_len(dataset:str|Path, choice: Literal["movies", "ratings","users"]) -> pd.DataFrame:
    pass

def attach_to_user(df_users:pd.DataFrame, df_ratings:pd.DataFrame, df_movies:pd.DataFrame)->pd.DataFrame:
    pass

def attach_to_item(df_users:pd.DataFrame, df_ratings:pd.DataFrame, df_movies:pd.DataFrame)->pd.DataFrame:
    pass