import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils.util_C import draw_bar_chart, parse_timestamp


def _solve1(df_items_metadata: pd.DataFrame):
    print("\n" + "-" * 50)
    print(" [C1] 아이템 메타데이터의 수치화")
    print("-" * 50)

    # 영화 메타데이터 고유 행 추출
    df_movies = df_items_metadata[["item", "title", "genres"]].drop_duplicates(subset=["item"]).copy()

    # C1-1: 영화 x 장르의 0/1 표 변환
    df_genres_dummy = df_movies["genres"].str.get_dummies(sep="|")
    df_product_movie_genres = pd.concat([df_movies[["item", "title"]], df_genres_dummy], axis=1)

    print(f"[C1-1] 영화 × 장르 0/1 행렬 크기: {df_genres_dummy.shape} (영화 수: {df_genres_dummy.shape[0]}, 장르 수: {df_genres_dummy.shape[1]})")

    # 장르별 영화 수 분포 (Index 값 '자체'가 장르 이름 문자열인 범주형 데이터프레임)
    genre_counts = df_genres_dummy.sum(axis=0).sort_values(ascending=False)
    genre_distribution_df = pd.DataFrame(
        {
            "영화 수": genre_counts.values,
            "전체 대비 비율(%)": (genre_counts.values / len(df_movies)) * 100,
        },
        index=genre_counts.index,  # index 값 자체가 'Drama', 'Comedy', 'Action' 등의 장르 문자
    )
    print("\n장르별 영화 수 분포 (Index = 범주형 장르 문자):")
    print(genre_distribution_df.round(2).to_string())

    # 가로축(X축)에 장르 문자를 범주형 데이터로 살려 막대그래프 시각화 및 저장
    draw_bar_chart(
        genre_counts,
        title="C1-1 Genre Distribution of Movies",
        xlabel="Genre (Categorical)",
        ylabel="Number of Movies",
        color="mediumpurple",
    )

    # 영화당 장르 수 기술통계 및 요약
    genres_per_movie = df_genres_dummy.sum(axis=1)
    print("\n영화당 보유 장르 수 기술통계 요약:")
    print(genres_per_movie.describe().to_string())

    # 고유 장르 조합 수
    unique_genre_combos = df_movies["genres"].nunique()
    print(f"\n고유 장르 조합 수: {unique_genre_combos}개")


    # C1-2: TF-IDF 행렬 계산 (제목 단어 + 장르 + 개봉연도)
    # 제목에서 연도 추출 및 제목 단어 분리
    df_movies["year"] = df_movies["title"].str.extract(r"\((\d{4})\)")[0].fillna("")
    df_movies["title_words"] = df_movies["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)
    df_movies["genre_words"] = df_movies["genres"].str.replace("|", " ", regex=False)

    df_movies["description"] = (
        df_movies["title_words"] + " " + df_movies["genre_words"] + " " + df_movies["year"]
    )

    tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_movies["description"])
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

    print(f"\n[C1-2] TF-IDF 행렬 생성 완료! 크기: {tfidf_matrix.shape}")

    # 임의의 영화 1편에 대해 TF-IDF 상위 5개 단어 출력
    sample_idx = 0  # 1번 인덱스 영화 (예: Toy Story)
    sample_movie = df_movies.iloc[sample_idx]
    sample_vec = tfidf_matrix[sample_idx].toarray().flatten()
    top5_indices = sample_vec.argsort()[-5:][::-1]

    print(f"\n샘플 영화: '{sample_movie['title']}' (장르: {sample_movie['genres']})")
    print(f"문서 내용: '{sample_movie['description']}'")
    print("TF-IDF 상위 5개 단어 및 가중치:")
    for rank, idx in enumerate(top5_indices, 1):
        print(f"  {rank}위: {feature_names[idx]} ({sample_vec[idx]:.4f})")


def _solve2(df_items_metadata: pd.DataFrame):
    print("\n" + "-" * 50)
    print(" [C2] 컨텍스트 (시간대 및 요일 분석)")
    print("-" * 50)

    # 평가 상호작용 로그가 존재하는 행만 필터링 (평가가 없는 영화 행의 NaN 제외)
    df_valid_ratings = df_items_metadata.dropna(subset=["ts"]).copy()

    # 타임스탬프 파싱
    df_parsed = parse_timestamp(df_valid_ratings)
    # 시간대를 명시적 int로 변환
    df_parsed["hour"] = df_parsed["hour"].astype(int)


    # (a) 시간대별·요일별 평가 건수 막대그래프
    hourly_counts = df_parsed["hour"].value_counts().sort_index()
    hourly_counts.index = hourly_counts.index.astype(int)
    hourly_counts.index.name = "시간대(시)"

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_counts = df_parsed["dayofweek"].value_counts().reindex(day_order)
    daily_counts.index.name = "요일"

    print("[C2-a] 시간대별 평가 건수:")
    print(hourly_counts.to_string())
    print("\n[C2-a] 요일별 평가 건수:")
    print(daily_counts.to_string())

    draw_bar_chart(
        hourly_counts,
        title="C2-a Hourly Evaluation Count",
        xlabel="Hour of Day (0-23)",
        ylabel="Number of Ratings",
        color="royalblue",
    )
    draw_bar_chart(
        daily_counts,
        title="C2-a Daily Evaluation Count",
        xlabel="Day of Week",
        ylabel="Number of Ratings",
        color="darkseagreen",
    )

    # (b) 'Horror', "Children's", 'Drama'의 시간대별 비중 표
    target_genres = ["Horror", "Children's", "Drama"]
    for g in target_genres:
        df_parsed[f"is_{g}"] = df_parsed["genres"].str.contains(g, na=False, regex=False)

    hourly_group = df_parsed.groupby("hour")
    total_ratings_by_hour = hourly_group.size().astype(int)

    genre_ratio_by_hour = pd.DataFrame(index=total_ratings_by_hour.index.astype(int))
    genre_ratio_by_hour.index.name = "시간대(시)"
    genre_ratio_by_hour["전체 평가 수"] = total_ratings_by_hour

    for g in target_genres:
        genre_count_hour = hourly_group[f"is_{g}"].sum()
        genre_ratio_by_hour[f"{g} 비중(%)"] = (genre_count_hour / total_ratings_by_hour) * 100

    print("\n[C2-b] 시간대별 주요 장르('Horror', 'Children\\'s', 'Drama') 비중 표:")
    # .0이 나오지 않도록 전체 평가 수는 int로, 비중은 소수점 2자리로 포맷
    formatters = {"전체 평가 수": "{:d}".format, **{f"{g} 비중(%)": "{:.2f}%".format for g in target_genres}}
    print(genre_ratio_by_hour.to_string(formatters=formatters))



def solve(df_items_metadata: pd.DataFrame):
    print("\n" + "=" * 60)
    print(" [Part C] 행렬 바깥의 데이터")
    print("=" * 60)
    _solve1(df_items_metadata)
    _solve2(df_items_metadata)
