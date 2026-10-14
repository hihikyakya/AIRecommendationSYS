import pandas as pd
from pathlib import Path

from ..utils import util_A


def solve(dataset: str | Path):
    print("\n" + "=" * 60)
    print(" [Part A] 데이터의 전체 지도와 이벤트 로그")
    print("=" * 60)

    # A1-1: 컬럼 범주 매핑 딕셔너리
    column_mapping = {
        "① 상호작용 (Interaction)": ["rating"],
        "② 아이템 메타데이터 (Item Metadata)": ["item", "title", "genres"],
        "③ 유저 정보 (User Information)": ["user", "gender", "age", "occupation", "zip"],
        "④ 컨텍스트 (Context)": ["ts"],
    }
    print("[A1] 컬럼 범주 매핑 딕셔너리:")
    for category, cols in column_mapping.items():
        print(f"  - {category}: {cols}")

    df_movies = util_A.parse_df_movie_len(dataset, "movies")
    df_ratings = util_A.parse_df_movie_len(dataset, "ratings")
    df_users = util_A.parse_df_movie_len(dataset, "users")

    df_users_data = util_A.attach_to_user(df_users, df_ratings, df_movies)
    print(f"\n유저 결합 데이터 컬럼 목록: {list(df_users_data.columns)}")
    df_items_metadata = util_A.attach_to_item(df_users, df_ratings, df_movies)
    print(f"아이템 결합 메타데이터 컬럼 목록: {list(df_items_metadata.columns)}")

    # A1-2: 이벤트 로그 형식 DataFrame 변환 및 앞 5행 출력
    print("\n[A1-2] 이벤트 로그 (Event Log) 상위 5행:")
    df_event_log = df_ratings.rename(
        columns={"user": "user_id", "item": "item_id", "rating": "rating", "ts": "timestamp"}
    )[["timestamp", "user_id", "item_id", "rating"]]
    print(df_event_log.head(n=5).to_string(index=False))

    return df_users_data, df_items_metadata