from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_histogram(
    data: pd.Series | pd.DataFrame | np.ndarray,
    title: str = "Histogram",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    bins: int | list = 20,
    save_path: str | Path | None = None,
):
    """데이터의 히스토그램을 그리고 이미지 파일로 저장하는 함수."""
    plt.figure(figsize=(8, 5))
    if isinstance(data, pd.DataFrame):
        data_to_plot = data.iloc[:, 0].dropna()
    elif isinstance(data, pd.Series):
        data_to_plot = data.dropna()
    else:
        data_to_plot = np.array(data)

    plt.hist(data_to_plot, bins=bins, color="skyblue", edgecolor="black", alpha=0.85)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if save_path is None:
        out_dir = Path("outputs/figures")
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = "".join(c if c.isalnum() else "_" for c in title.lower()) + ".png"
        save_path = out_dir / filename
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [차트 저장 완료] {save_path}")