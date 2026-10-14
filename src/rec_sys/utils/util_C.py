from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """타임스탬프에서 datetime, 시간대(hour), 요일(dayofweek) 컬럼을 파생."""
    df_res = df.copy()
    dt_series = pd.to_datetime(df_res["ts"], unit="s")
    df_res["datetime"] = dt_series
    df_res["hour"] = dt_series.dt.hour
    # 요일 이름 (Monday ~ Sunday) 및 요일 인덱스
    df_res["dayofweek"] = dt_series.dt.day_name()
    return df_res


def draw_bar_chart(
    series: pd.Series,
    title: str = "Bar Chart",
    xlabel: str = "Category",
    ylabel: str = "Count",
    save_path: str | Path | None = None,
    color: str = "steelblue",
):
    """시리즈 데이터를 바탕으로 막대그래프를 그리고 파일로 저장."""
    plt.figure(figsize=(9, 5))
    series.plot(kind="bar", color=color, edgecolor="black", alpha=0.85)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45 if len(series) > 10 else 0)
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