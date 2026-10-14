import numpy as np
import pandas as pd

from ..utils.util_B import draw_histogram


def _solve1(df_users_data: pd.DataFrame):
    print("\n" + "-" * 50)
    print(" [B1] Explicit 평점의 해부")
    print("-" * 50)

    # 1. 평점 1–5의 빈도와 비율 출력 및 히스토그램
    rating_counts = df_users_data["rating"].value_counts().sort_index()
    rating_ratios = df_users_data["rating"].value_counts(normalize=True).sort_index()
    rating_summary = pd.DataFrame({"빈도 (Count)": rating_counts, "비율 (Ratio)": rating_ratios})
    print("[B1-1] 평점 1~5 빈도 및 비율:")
    print(rating_summary.to_string())

    df_ratings_rate = df_users_data["rating"]
    draw_histogram(
        df_ratings_rate,
        title="B1-1 Rating Distribution",
        xlabel="Rating",
        ylabel="Count",
        bins=np.arange(0.5, 6.5, 1),
    )

    # 2. 유저별 평균 평점 히스토그램 및 극단 유저 비율
    df_users_rate = df_users_data.groupby("user")["rating"].mean()
    high_user_ratio = (df_users_rate >= 4.2).mean()
    low_user_ratio = (df_users_rate <= 3.0).mean()

    print("\n[B1-2] 유저별 평균 평점 극단 유저 비율:")
    print(f"  - 평균 평점 4.2 이상 유저 비율: {high_user_ratio * 100:.2f}%")
    print(f"  - 평균 평점 3.0 이하 유저 비율: {low_user_ratio * 100:.2f}%")

    draw_histogram(
        df_users_rate,
        title="B1-2 User Mean Rating Distribution",
        xlabel="User Mean Rating",
        ylabel="User Count",
        bins=25,
    )

    # 3. 영화를 평가 수 기준 5개 구간(분위)으로 나누고 구간별 영화 수와 평균 평점 표 출력
    item_stats = df_users_data.groupby("item")["rating"].agg(eval_count="count", avg_rating="mean")
    item_stats["quantile_bin"] = pd.qcut(
        item_stats["eval_count"], q=5, labels=["1분위(하위 20%)", "2분위", "3분위", "4분위", "5분위(상위 20%)"]
    )
    bin_summary = item_stats.groupby("quantile_bin", observed=False).agg(
        영화수=("avg_rating", "count"),
        평균평점=("avg_rating", "mean"),
        최소평가수=("eval_count", "min"),
        최대평가수=("eval_count", "max"),
    )
    print("\n[B1-3] 영화 평가 수 5개 분위별 영화 수 및 평균 평점:")
    print(bin_summary.to_string())


def _solve2(df_users_data: pd.DataFrame):
    print("\n" + "-" * 50)
    print(" [B2] Implicit 안경과 Negative Sampling")
    print("-" * 50)

    # 1. 평가를 '평가함 = 1'로 이진화했을 때 원래 1–2였던 칸의 비율
    # 전체 관측된 칸(평가함=1) 중 원래 평점이 1 또는 2였던 칸의 비율
    low_rating_ratio = (df_users_data["rating"] <= 2).mean()
    print(f"[B2-1] 관측된 평가(1) 중 원래 평점이 1-2점(부정적 평가)이었던 칸의 비율: {low_rating_ratio * 100:.2f}%")

    # 2. 각 유저의 평가를 시간순 앞 80%(과거) / 뒤 20%(미래) 분할
    df_sorted = df_users_data.sort_values(["user", "ts"]).copy()
    df_sorted["user_rank"] = df_sorted.groupby("user").cumcount()
    df_sorted["user_total"] = df_sorted.groupby("user")["item"].transform("count")
    df_sorted["is_past"] = df_sorted["user_rank"] < (df_sorted["user_total"] * 0.8).astype(int)

    past_df = df_sorted[df_sorted["is_past"]]
    future_df = df_sorted[~df_sorted["is_past"]]

    # 고유 아이템 풀 및 과거 평가 빈도(인기도)
    all_items = np.array(df_sorted["item"].unique())
    past_item_counts = past_df["item"].value_counts().to_dict()
    total_item_counts = df_sorted["item"].value_counts().to_dict()

    weights = np.array([past_item_counts.get(it, 1e-8) for it in all_items], dtype=np.float64)

    user_past = past_df.groupby("user")["item"].apply(set).to_dict()
    user_future = future_df.groupby("user")["item"].apply(set).to_dict()

    # 과거 양성 1개당 음성 4개 샘플링 (균등 & 인기 기반)
    np.random.seed(42)

    unif_eval_counts = []
    unif_future_eval_hits = []

    pop_eval_counts = []
    pop_future_eval_hits = []

    print("[B2-2 & B2-3] 전체 유저 대상 음성 샘플링(과거 양성 1개당 4개) 진행 중...")
    for user_id, past_items in user_past.items():
        future_items = user_future.get(user_id, set())
        num_neg = len(past_items) * 4
        if num_neg == 0:
            continue

        cand_mask = np.array([it not in past_items for it in all_items])
        cand_items = all_items[cand_mask]

        # (1) 균등 샘플링 (Uniform Sampling)
        neg_unif = np.random.choice(cand_items, size=num_neg, replace=True)
        unif_eval_counts.extend([total_item_counts.get(it, 0) for it in neg_unif])
        unif_future_eval_hits.extend([it in future_items for it in neg_unif])

        # (2) 인기 기반 샘플링 (Popularity-based Sampling)
        cand_w = weights[cand_mask]
        cand_p = cand_w / cand_w.sum()
        neg_pop = np.random.choice(cand_items, size=num_neg, replace=True, p=cand_p)
        pop_eval_counts.extend([total_item_counts.get(it, 0) for it in neg_pop])
        pop_future_eval_hits.extend([it in future_items for it in neg_pop])

    # 3. 두 방식 비교 결과 표 출력
    res_df = pd.DataFrame(
        {
            "샘플링 방식": ["균등 샘플링 (Uniform)", "인기 기반 샘플링 (Popularity)"],
            "뽑힌 음성 영화의 평균 평가 수": [np.mean(unif_eval_counts), np.mean(pop_eval_counts)],
            "미래 구간에서 실제로 평가한 비율(%)": [
                np.mean(unif_future_eval_hits) * 100,
                np.mean(pop_future_eval_hits) * 100,
            ],
        }
    )
    print("\n[B2-3] 균등 샘플링 vs 인기 기반 샘플링 비교 결과:")
    print(res_df.to_string(index=False))


def solve(df_users_data: pd.DataFrame):
    print("\n" + "=" * 60)
    print(" [Part B] 상호작용 데이터: 두 가지 피드백")
    print("=" * 60)
    _solve1(df_users_data)
    _solve2(df_users_data)