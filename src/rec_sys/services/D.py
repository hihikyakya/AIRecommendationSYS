from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..utils.util_D import draw_lorenz_curve
from ..utils.util_DE import get_density, get_gini_index


def solve(df_users_data: pd.DataFrame):
    print("\n" + "=" * 60)
    print(" [Part D] 표현과 측정")
    print("=" * 60)

    # ========================================================
    # D1-1: 세 희소 행렬 구성 및 유저 이탈율 분석
    # ========================================================
    print("\n[D1-1] 세 희소 행렬 구성 및 유저 이탈율 분석")

    # 고유 유저 및 영화 ID 매핑
    unique_users = sorted(df_users_data["user"].unique())
    unique_items = sorted(df_users_data["item"].unique())
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    item_to_idx = {it: i for i, it in enumerate(unique_items)}

    num_users = len(unique_users)
    num_items = len(unique_items)

    u_indices = df_users_data["user"].map(user_to_idx).values
    i_indices = df_users_data["item"].map(item_to_idx).values
    ratings_val = df_users_data["rating"].values

    # 행렬 ①: 전체 기간, 값은 평점
    mat1 = csr_matrix((ratings_val, (u_indices, i_indices)), shape=(num_users, num_items))

    # 행렬 ②: 전체 기간, 값은 1 (평가 여부)
    mat2 = csr_matrix((np.ones_like(ratings_val), (u_indices, i_indices)), shape=(num_users, num_items))

    # 행렬 ③: 데이터 마지막 시점 기준 최근 180일, 값은 평점
    last_timestamp = df_users_data["ts"].max()
    cutoff_timestamp = last_timestamp - (180 * 24 * 3600)
    recent_180_df = df_users_data[df_users_data["ts"] >= cutoff_timestamp].copy()

    u_recent = recent_180_df["user"].map(user_to_idx).values
    i_recent = recent_180_df["item"].map(item_to_idx).values
    r_recent = recent_180_df["rating"].values

    mat3 = csr_matrix((r_recent, (u_recent, i_recent)), shape=(num_users, num_items))

    # ①에 있던 유저 중 ③에서 사라진 유저 비율
    active_users_1 = set(df_users_data["user"])
    active_users_3 = set(recent_180_df["user"])
    churned_users = active_users_1 - active_users_3
    churn_ratio = len(churned_users) / len(active_users_1)

    mat_summary = pd.DataFrame(
        {
            "행렬 구분": [
                "① 전체 기간 (값: 평점)",
                "② 전체 기간 (값: 1, 평가여부)",
                "③ 최근 180일 (값: 평점)",
            ],
            "관측된 칸 수(nnz)": [mat1.nnz, mat2.nnz, mat3.nnz],
            "유저 수": [df_users_data["user"].nunique(), df_users_data["user"].nunique(), recent_180_df["user"].nunique()],
            "영화 수": [df_users_data["item"].nunique(), df_users_data["item"].nunique(), recent_180_df["item"].nunique()],
        }
    )
    print(mat_summary.to_string(index=False))
    print(f"\n* 마지막 시점 기준 180일 경계: {pd.to_datetime(cutoff_timestamp, unit='s')} ~ {pd.to_datetime(last_timestamp, unit='s')}")
    print(f"* ①에 있던 전체 {len(active_users_1)}명 중 ③에서 사라진 유저: {len(churned_users)}명 ({churn_ratio * 100:.2f}%)")

    # E 파트에서 활용하기 위한 유저 x 영화 피벗 행렬
    df_product_user_movie = df_users_data.pivot(index="user", columns="item", values="rating").fillna(0)

    # ========================================================
    # D1-2: 행렬 ①의 밀도 및 관측 수 통계
    # ========================================================
    print("\n[D1-2] 행렬 ①의 밀도 및 관측 수 통계")
    density = get_density(df_product_user_movie)
    print(f"행렬 ① 밀도 (|Ω| / (|U| × |I|)): {density * 100:.4f}% ({density:.6f})")

    user_eval_counts = df_users_data.groupby("user")["item"].count()
    item_eval_counts = df_users_data.groupby("item")["user"].count()

    eval_stats_df = pd.DataFrame(
        {
            "구분": ["유저당 관측 수", "영화당 관측 수"],
            "평균 (Mean)": [user_eval_counts.mean(), item_eval_counts.mean()],
            "중앙값 (Median)": [user_eval_counts.median(), item_eval_counts.median()],
            "최솟값 (Min)": [user_eval_counts.min(), item_eval_counts.min()],
            "최댓값 (Max)": [user_eval_counts.max(), item_eval_counts.max()],
        }
    )
    print(eval_stats_df.round(2).to_string(index=False))

    # ========================================================
    # D1-3: 두 유저가 함께 평가한 영화 수의 기댓값 (어림 계산)
    # ========================================================
    print("\n[D1-3] 두 유저가 함께 평가한 영화 수의 기댓값 (강의 어림 계산)")
    # 강의 어림 계산: E[공통 평가 수] = (평균 유저 평가 수)^2 / 전체 영화 수
    avg_user_ratings = user_eval_counts.mean()
    approx_expected_coeval = (avg_user_ratings**2) / num_items

    # 정확한 기댓값 (모든 영화 i에 대해 n_i*(n_i-1) / (|U|*(|U|-1)))
    exact_expected_coeval = float(np.sum(item_eval_counts.values * (item_eval_counts.values - 1))) / (num_users * (num_users - 1))

    print(f"  - 평균 유저 평가 수 N_u: {avg_user_ratings:.2f}건, 전체 영화 수 |I|: {num_items}편")
    print(f"  - 강의 어림 계산 기댓값 (N_u^2 / |I|): {approx_expected_coeval:.4f}편")
    print(f"  - 실제 데이터 기준 기대 공통 평가 수: {exact_expected_coeval:.4f}편")

    # ========================================================
    # D1-4: 순위-평가 수 그래프, 상위 점유율, 지니계수 및 로렌츠 곡선
    # ========================================================
    print("\n[D1-4] 영화 평가 순위-평가 수 분석 및 지니계수")

    sorted_items = item_eval_counts.sort_values(ascending=False).reset_index(drop=True)
    total_ratings = sorted_items.sum()
    cumsum_ratings = sorted_items.cumsum()

    # 상위 1%, 5%, 10%, 20% 점유율 계산
    percentages = [1, 5, 10, 20]
    top_share_data = []
    for pct in percentages:
        k = max(1, int(np.ceil(num_items * (pct / 100.0))))
        share = cumsum_ratings.iloc[k - 1] / total_ratings
        top_share_data.append({"상위 영화 비율(%)": f"상위 {pct}%", "영화 편수": f"{k}편", "전체 평가 점유율(%)": f"{share * 100:.2f}%"})

    print("상위 영화들의 평가 점유율:")
    print(pd.DataFrame(top_share_data).to_string(index=False))

    # 순위-평가 수 그래프 시각화 및 저장
    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(1, num_items + 1), sorted_items.values, color="crimson", linewidth=2)
    plt.title("Movie Rank vs Evaluation Count (Long Tail)", fontsize=14, fontweight="bold")
    plt.xlabel("Movie Rank", fontsize=12)
    plt.ylabel("Number of Ratings", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_save_path = out_dir / "d1_4_rank_eval_count.png"
    plt.tight_layout()
    plt.savefig(rank_save_path, dpi=150)
    plt.close()
    print(f"  [차트 저장 완료] {rank_save_path}")

    # 지니계수 계산 및 로렌츠 곡선
    item_gini = get_gini_index(item_eval_counts)
    user_gini = get_gini_index(user_eval_counts)

    print(f"\n지니계수(Gini Index) 계산 결과:")
    print(f"  - 영화 축(Item Axis) 지니계수: {item_gini:.4f}")
    print(f"  - 유저 축(User Axis) 지니계수: {user_gini:.4f}")

    draw_lorenz_curve(item_eval_counts, user_eval_counts, save_path=out_dir / "d1_4_lorenz_curves.png")

    return df_product_user_movie
