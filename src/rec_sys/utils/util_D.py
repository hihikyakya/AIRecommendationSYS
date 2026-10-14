from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_density(matrix_or_df: pd.DataFrame | np.ndarray) -> float:
    """유저-영화 행렬의 밀도(|Ω| / (|U| * |I|))를 계산."""
    if isinstance(matrix_or_df, pd.DataFrame):
        # 0이 아니거나 결측치가 아닌 유효 평가 칸 수
        if matrix_or_df.isna().any().any():
            observed = matrix_or_df.notna().sum().sum()
        else:
            observed = (matrix_or_df > 0).sum().sum()
        total_cells = matrix_or_df.shape[0] * matrix_or_df.shape[1]
    else:
        arr = np.array(matrix_or_df)
        observed = np.count_nonzero(~np.isnan(arr) & (arr > 0))
        total_cells = arr.size

    return float(observed / total_cells) if total_cells > 0 else 0.0


def get_gini_index(values: np.ndarray | pd.Series | list) -> float:
    """지니 계수(Gini Index)를 계산 (0 = 완전 평등, 1 = 완전 불평등)."""
    arr = np.array(values, dtype=np.float64).flatten()
    arr = arr[~np.isnan(arr)]
    arr = arr[arr >= 0]
    n = len(arr)
    if n == 0 or np.sum(arr) == 0:
        return 0.0
    arr = np.sort(arr)
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * arr)) / (n * np.sum(arr)) - (n + 1.0) / n
    return float(gini)


def draw_lorenz_curve(
    movie_counts: pd.Series | np.ndarray,
    user_counts: pd.Series | np.ndarray,
    save_path: str | Path | None = None,
):
    """영화 축과 유저 축 각각의 로렌츠 곡선(Lorenz Curve)을 시각화."""
    plt.figure(figsize=(8, 7))

    for counts, label, color in [
        (movie_counts, "Movies (Item axis)", "royalblue"),
        (user_counts, "Users (User axis)", "darkorange"),
    ]:
        arr = np.sort(np.array(counts, dtype=np.float64))
        cum_arr = np.cumsum(arr)
        cum_arr_pct = cum_arr / cum_arr[-1]
        x_pct = np.linspace(0, 1, len(arr) + 1)
        y_pct = np.insert(cum_arr_pct, 0, 0.0)
        plt.plot(x_pct, y_pct, label=label, color=color, linewidth=2)

    # 45도 대각선 (완전 평등선)
    plt.plot([0, 1], [0, 1], "k--", label="Equality line (Gini = 0)", alpha=0.7)

    plt.title("Lorenz Curves: Item vs User Axis", fontsize=14, fontweight="bold")
    plt.xlabel("Cumulative Proportion of Population", fontsize=12)
    plt.ylabel("Cumulative Proportion of Ratings", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path is None:
        out_dir = Path("outputs/figures")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "d1_4_lorenz_curves.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [차트 저장 완료] {save_path}")