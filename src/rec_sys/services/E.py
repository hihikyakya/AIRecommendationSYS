import numpy as np
import pandas as pd

from ..utils import util_E
from ..utils.util_DE import get_density, get_gini_index


def _solve1(df_users_data: pd.DataFrame):
    print("\n" + "-" * 50)
    print(" [E1] 정제와 k-core 필터링")
    print("-" * 50)

    # E1-1: 정제 규칙 두 가지 (중복 평가, 비정상적으로 빠른 연속 평가) (pass)
    total_ratings = len(df_users_data)
    total_users = df_users_data["user"].nunique()

    # 규칙 1: 중복 평가 (동일 유저-영화)
    dup_mask = df_users_data.duplicated(subset=["user", "item"], keep="first")
    dup_eval_count = dup_mask.sum()
    dup_user_count = df_users_data[dup_mask]["user"].nunique()

    # 규칙 2: 비정상적으로 빠른 연속 평가 (봇 난사 입력, 연속 평가 시간차 <= 1초)
    df_sorted = df_users_data.sort_values(["user", "ts"]).copy()
    time_diff = df_sorted.groupby("user")["ts"].diff()
    fast_mask = (time_diff.notna()) & (time_diff <= 1)
    fast_eval_count = fast_mask.sum()
    fast_user_count = df_sorted[fast_mask]["user"].nunique()

    clean_summary = pd.DataFrame(
        {
            "정제 규칙": ["규칙 1: 중복 평가 (Duplicated)", "규칙 2: 비정상 고속 평가 (<=1초)"],
            "제거되는 평가 수": [dup_eval_count, fast_eval_count],
            "영향받는 유저 수": [dup_user_count, fast_user_count],
            "전체 평가 대비 비율(%)": [
                (dup_eval_count / total_ratings) * 100,
                (fast_eval_count / total_ratings) * 100,
            ],
            "전체 유저 대비 비율(%)": [
                (dup_user_count / total_users) * 100,
                (fast_user_count / total_users) * 100,
            ],
        }
    )
    print("[E1-1] 정제 규칙별 필터링 효과 표:")
    print(clean_summary.to_string(index=False))

    # E1-2: 반복 제거 방식의 k-core 필터링 (k = 5, 10, 20, 30)
    print("\n[E1-2] k-core 필터링 (k = 5, 10, 20, 30):")
    k_values = [5, 10, 20, 30]
    k_core_results = []

    user_min_evals = df_users_data.groupby("user")["item"].count().min()
    item_min_evals = df_users_data.groupby("item")["user"].count().min()
    print(f"* 필터링 전 최소 평가 수 - 유저별: {user_min_evals}건, 영화별: {item_min_evals}건")

    for k in k_values:
        filtered_df, iters = util_E.k_core_filtering(df_users_data, k=k)
        rem_users = filtered_df["user"].nunique()
        rem_items = filtered_df["item"].nunique()
        rem_evals = len(filtered_df)

        density = (rem_evals / (rem_users * rem_items)) if rem_users * rem_items > 0 else 0.0
        item_eval_counts = filtered_df.groupby("item")["user"].count()
        item_gini = get_gini_index(item_eval_counts)

        k_core_results.append(
            {
                "k-core": f"k={k}",
                "반복 횟수": iters,
                "남은 유저 수": rem_users,
                "남은 영화 수": rem_items,
                "남은 평가 수": rem_evals,
                "밀도(%)": density * 100,
                "아이템 지니계수": item_gini,
            }
        )

    res_k_df = pd.DataFrame(k_core_results)
    print(res_k_df.to_string(index=False))

    # k = 20 적용 시 제거되는 유저 수와 영화 수
    k20_row = res_k_df[res_k_df["k-core"] == "k=20"].iloc[0]
    dropped_users_k20 = total_users - k20_row["남은 유저 수"]
    dropped_items_k20 = df_users_data["item"].nunique() - k20_row["남은 영화 수"]
    print(f"\n* k=20 필터링 적용 시 제거되는 유저 수: {dropped_users_k20}명, 영화 수: {dropped_items_k20}편")


def _solve2(df_users_data: pd.DataFrame):
    print("\n" + "-" * 50)
    print(" [E2] 유저 중심 정렬과 축소 보정")
    print("-" * 50)


    # E2-1: 유저별 시간순 정렬 앞 80%를 train으로 두고 유저별 평균을 빼는 중심화 함수
    df_sorted = df_users_data.sort_values(["user", "ts"]).copy()
    df_sorted["user_rank"] = df_sorted.groupby("user").cumcount()
    df_sorted["user_total"] = df_sorted.groupby("user")["item"].transform("count")
    df_sorted["is_train"] = df_sorted["user_rank"] < (df_sorted["user_total"] * 0.8).astype(int)

    train_df = df_sorted[df_sorted["is_train"]].copy()

    # 중심화 함수 적용
    user_means = train_df.groupby("user")["rating"].transform("mean")
    train_df["user_mean"] = user_means
    train_df["centered_rating"] = train_df["rating"] - user_means

    print(f"[E2-1] Train 데이터 크기: {len(train_df):,}건 (전체의 {len(train_df)/len(df_users_data)*100:.1f}%)")
    print("  - 중심화된 평점(centered_rating = rating - user_mean) 상위 5행:")
    print(train_df[["user", "item", "rating", "user_mean", "centered_rating"]].head(5).round(3).to_string(index=False))

    # E2-2: 공통 평가 영화 >= 3편 유저 쌍 중 평균 평점 차이가 가장 큰 쌍 ('후한 민지 / 짠돌이 서준')
    user_avg = train_df.groupby("user")["rating"].mean()
    high_users = user_avg.sort_values(ascending=False).index[:100]
    low_users = user_avg.sort_values(ascending=True).index[:100]

    # 유저별 영화 set
    train_user_movies = train_df.groupby("user")["item"].apply(set).to_dict()

    best_pair = None
    max_diff = -1.0
    common_movies_found = set()

    for u_high in high_users:
        m_high = train_user_movies.get(u_high, set())
        for u_low in low_users:
            m_low = train_user_movies.get(u_low, set())
            common_m = m_high & m_low
            if len(common_m) >= 3:
                diff = user_avg[u_high] - user_avg[u_low]
                if diff > max_diff:
                    max_diff = diff
                    best_pair = (u_high, u_low)
                    common_movies_found = common_m

    u_generous, u_strict = best_pair
    print(f"\n[E2-2] 공통 영화 3편 이상 유저 쌍 중 평균 차이가 가장 큰 쌍:")
    print(f"  - 후한 유저 (User {u_generous}): 평균 {user_avg[u_generous]:.2f}점")
    print(f"  - 짠돌이 유저 (User {u_strict}): 평균 {user_avg[u_strict]:.2f}점 (평균 차이: {max_diff:.2f}점)")

    # 영화 제목 맵
    movie_title_map = df_users_data.drop_duplicates(subset=["item"]).set_index("item")["title"].to_dict()

    pair_rows = []
    for m in common_movies_found:
        r_gen = train_df[(train_df["user"] == u_generous) & (train_df["item"] == m)]["rating"].values[0]
        c_gen = train_df[(train_df["user"] == u_generous) & (train_df["item"] == m)]["centered_rating"].values[0]

        r_str = train_df[(train_df["user"] == u_strict) & (train_df["item"] == m)]["rating"].values[0]
        c_str = train_df[(train_df["user"] == u_strict) & (train_df["item"] == m)]["centered_rating"].values[0]

        pair_rows.append(
            {
                "MovieID": m,
                "영화 제목": movie_title_map.get(m, f"Movie {m}"),
                "후한 유저 원점수": r_gen,
                "후한 유저 중심화점수": round(c_gen, 2),
                "짠돌이 유저 원점수": r_str,
                "짠돌이 유저 중심화점수": round(c_str, 2),
            }
        )

    print("\n강의의 '후한 민지 / 짠돌이 서준' 비교 표:")
    print(pd.DataFrame(pair_rows).to_string(index=False))

    # E2-3: 축소 보정 함수 (평가 수가 적은 유저의 평균을 전체 평균으로 당김)
    global_mean = float(train_df["rating"].mean())
    user_train_counts = train_df.groupby("user")["rating"].count()
    user_train_means = train_df.groupby("user")["rating"].mean()

    user_stats = pd.DataFrame(
        {"user": user_train_counts.index, "eval_count": user_train_counts.values, "user_mean": user_train_means.values}
    )
    user_stats = user_stats.sort_values(by=["eval_count", "user"]).reset_index(drop=True)

    bottom_10_users = user_stats.head(10).copy()
    shrinkage_df = util_E.reduction_filtering(bottom_10_users, global_mean=global_mean, lambdas=[5.0, 10.0, 25.0])

    print(f"\n[E2-3] 축소 보정 (전체 Train 평균 μ = {global_mean:.3f}):")
    print("  Train 평가 수가 가장 적은 유저 10명의 보정 전후 평균 평점:")
    print(shrinkage_df.round(3).to_string(index=False))


def solve(df_users_data: pd.DataFrame):
    print("\n" + "=" * 60)
    print(" [Part E] 전처리: 원료를 다듬는 공정")
    print("=" * 60)
    _solve1(df_users_data)
    _solve2(df_users_data=df_users_data)
