from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import util_F


def _solve1(df_users_data: pd.DataFrame) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    print("\n" + "-" * 50)
    print(" [F1] 네 가지 분할과 누출 (Data Split & Leakage)")
    print("-" * 50)

    # 1. 랜덤 8:2 분할과 시간 역행 누출
    train_rand, test_rand = util_F.random_train_test_split(df_users_data, test_size=0.2, random_state=42)

    train_max_ts = train_rand.groupby("user")["ts"].max()
    is_time_travel = test_rand["ts"] < test_rand["user"].map(train_max_ts)
    time_travel_ratio = is_time_travel.mean()

    # 같은 유저가 같은 날 매긴 평가가 train과 test로 나뉜 비율
    train_rand["date"] = pd.to_datetime(train_rand["ts"], unit="s").dt.date
    test_rand["date"] = pd.to_datetime(test_rand["ts"], unit="s").dt.date

    train_user_dates = set(zip(train_rand["user"], train_rand["date"]))
    test_user_dates = set(zip(test_rand["user"], test_rand["date"]))
    same_day_overlap = train_user_dates & test_user_dates

    test_same_day_mask = test_rand.apply(lambda r: (r["user"], r["date"]) in same_day_overlap, axis=1)
    same_day_split_ratio = test_same_day_mask.mean()

    print("[F1-1] 랜덤 8:2 분할:")
    print(f"  - 시간 역행 누출(test 중 같은 유저의 더 늦은 평가가 train에 있는 비율): {time_travel_ratio * 100:.2f}%")
    print(f"  - 같은 유저가 같은 날 매긴 평가가 train과 test로 나뉜 비율: {same_day_split_ratio * 100:.2f}%")

    # 2. 전역 시간 분할 (80% 경계)
    train_seq, test_seq, boundary_ts = util_F.sequential_train_test_split(df_users_data, train_ratio=0.8)
    boundary_dt = pd.to_datetime(boundary_ts, unit="s")

    train_users = set(train_seq["user"])
    test_users = set(test_seq["user"])
    test_only_users = test_users - train_users

    # test 유저 중 train 평가가 5건 미만인 유저 비율
    train_user_counts = train_seq.groupby("user").size()
    test_users_under_5 = [u for u in test_users if train_user_counts.get(u, 0) < 5]
    under_5_ratio = len(test_users_under_5) / len(test_users) if test_users else 0.0

    print("\n[F1-2] 전역 시간 분할 (Global Time Split):")
    print(f"  - 80% 경계 일시: {boundary_dt}")
    print(f"  - test에만 존재하는 유저 수 (콜드 스타트 유저): {len(test_only_users)}명")
    print(f"  - test 유저 중 train 평가 건수가 5건 미만인 비율: {under_5_ratio * 100:.2f}%")

    # 월별 평가 수 그래프 및 경계 세로선 표시
    df_plot = df_users_data.copy()
    df_plot["year_month"] = pd.to_datetime(df_plot["ts"], unit="s").dt.to_period("M")
    monthly_counts = df_plot["year_month"].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    x_labels = [str(period) for period in monthly_counts.index]
    plt.bar(x_labels, monthly_counts.values, color="cornflowerblue", edgecolor="black", alpha=0.85)

    boundary_period = boundary_dt.to_period("M")
    if boundary_period in monthly_counts.index:
        boundary_idx = list(monthly_counts.index).index(boundary_period)
        plt.axvline(x=boundary_idx, color="red", linestyle="--", linewidth=2, label=f"Boundary (80%): {boundary_period}")

    plt.title("Monthly Evaluation Count with Global Time Split Boundary", fontsize=14, fontweight="bold")
    plt.xlabel("Year-Month", fontsize=12)
    plt.ylabel("Rating Count", fontsize=12)
    plt.xticks(rotation=60)
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    f1_save_path = out_dir / "f1_2_monthly_eval_boundary.png"
    plt.tight_layout()
    plt.savefig(f1_save_path, dpi=150)
    plt.close()
    print(f"  [차트 저장 완료] {f1_save_path}")

    # 3. 유저별 시간 분할 (80% / 20%) 및 약한 누출 분석
    train_user_seq, test_user_seq = util_F.sequential_train_test_split_groupby(df_users_data, by="user", train_ratio=0.8)
    test_user_seq["is_before_global_boundary"] = test_user_seq["ts"] < boundary_ts
    weak_leakage_ratio = test_user_seq["is_before_global_boundary"].mean()

    print("\n[F1-3] 유저별 시간 분할 (User-wise Sequential Split):")
    print(f"  - test 평가 중 전역 시간 경계({boundary_dt})보다 이전인 비율 (약한 누출): {weak_leakage_ratio * 100:.2f}%")

    # 4. leave-one-out 분할
    loo_train, loo_valid, loo_test = util_F.leave_one_out_split(df_users_data, by="user")
    loo_user_count = len(loo_test)
    loo_high_rating_ratio = (loo_test["rating"] >= 4).mean()

    print("\n[F1-4] Leave-One-Out 분할:")
    print(f"  - 평가 대상 유저 수: {loo_user_count}명")
    print(f"  - test 정답 중 ★4 이상인 비율: {loo_high_rating_ratio * 100:.2f}%")

    # 네 가지 분할 결과 종합 표 출력
    split_summary_df = pd.DataFrame(
        {
            "분할 방식": [
                "1. 랜덤 8:2",
                "2. 전역 시간 분할",
                "3. 유저별 시간 분할",
                "4. Leave-One-Out",
            ],
            "핵심 측정 항목": [
                f"시간역행 누출 {time_travel_ratio*100:.2f}%, 같은날 분할 {same_day_split_ratio*100:.2f}%",
                f"경계일({boundary_dt.strftime('%Y-%m-%d')}), test전용 {len(test_only_users)}명, 5건미만 {under_5_ratio*100:.2f}%",
                f"약한 누출 비율 {weak_leakage_ratio*100:.2f}%",
                f"대상 {loo_user_count}명, ★4 이상 비율 {loo_high_rating_ratio*100:.2f}%",
            ],
        }
    )
    print("\n[F1 종합] 네 가지 데이터 분할 비교표:")
    print(split_summary_df.to_string(index=False))

    return boundary_ts, loo_train, loo_test


def _solve2(df_users_data: pd.DataFrame, loo_train: pd.DataFrame, loo_test: pd.DataFrame):
    print("\n" + "-" * 50)
    print(" [F2] Chance Level: 분할과 채점이 바꾸는 기준선")
    print("-" * 50)

    # 전체 영화 풀
    all_movies = set(df_users_data["item"].unique())
    num_total_movies = len(all_movies)

    # 유저별 train 평가 영화 풀
    train_movies_by_user = loo_train.groupby("user")["item"].apply(set).to_dict()

    # 유저별 leave-one-out test 아이템 및 평점
    test_rows = loo_test.set_index("user")[["item", "rating"]]

    user_chance_probabilities = []
    has_high_rating_target = []

    k_recommend = 10  # 무작위 10편 추천

    for user_id, row in test_rows.iterrows():
        test_item = row["item"]
        test_rating = row["rating"]

        rated_in_train = train_movies_by_user.get(user_id, set())
        unrated_candidates = all_movies - rated_in_train
        n_candidates = len(unrated_candidates)

        if test_rating >= 4 and (test_item in unrated_candidates):
            has_high_rating_target.append(True)
            # 숨긴 정답 1편이 무작위 10편 추천에 포함될 확률 (초기하분포 / 조합)
            # P(Hit) = 1 - ((N-1) choose 10) / (N choose 10) = 10 / N
            p_hit = k_recommend / n_candidates if n_candidates >= k_recommend else 1.0
            user_chance_probabilities.append(p_hit)
        else:
            has_high_rating_target.append(False)
            user_chance_probabilities.append(0.0)

    chance_level_mean = np.mean(user_chance_probabilities)
    high_rating_user_ratio = np.mean(has_high_rating_target)

    # 몬테카를로 시뮬레이션 검증 (10회 평균)
    np.random.seed(42)
    sim_hits = []
    sample_users = list(test_rows.index)

    for user_id in sample_users:
        test_item = test_rows.loc[user_id, "item"]
        test_rating = test_rows.loc[user_id, "rating"]
        if test_rating < 4:
            sim_hits.append(0)
            continue
        rated_in_train = train_movies_by_user.get(user_id, set())
        candidates = list(all_movies - rated_in_train)
        if len(candidates) < k_recommend:
            sim_hits.append(int(test_item in candidates))
            continue
        rec = np.random.choice(candidates, size=k_recommend, replace=False)
        sim_hits.append(int(test_item in rec))

    sim_chance_level = np.mean(sim_hits)

    f2_df = pd.DataFrame(
        {
            "지표": [
                "평가 대상 총 유저 수",
                "숨긴 정답(★4 이상)을 보유한 유저 비율(%)",
                "이론적 기대 성적 (Chance Level)",
                "시뮬레이션 실측 성적 (Chance Level)",
            ],
            "값": [
                f"{len(test_rows)}명",
                f"{high_rating_user_ratio * 100:.2f}%",
                f"{chance_level_mean * 100:.4f}% ({chance_level_mean:.6f})",
                f"{sim_chance_level * 100:.4f}% ({sim_chance_level:.6f})",
            ],
        }
    )
    print("[F2] 무작위 10편 추천 시 Chance Level (Hit@10 유저 비율):")
    print(f2_df.to_string(index=False))


def solve(df_users_data: pd.DataFrame):
    print("\n" + "=" * 60)
    print(" [Part F] 데이터 분할: 시간을 지켜라")
    print("=" * 60)
    boundary_ts, loo_train, loo_test = _solve1(df_users_data)
    _solve2(df_users_data, loo_train=loo_train, loo_test=loo_test)