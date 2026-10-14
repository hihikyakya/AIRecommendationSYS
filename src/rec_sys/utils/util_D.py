from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



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
