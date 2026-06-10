from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RAW_INPUT_GLOB = "classroom-metrics-*.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = PROJECT_ROOT / "history"
OUTPUT_DIR = PROJECT_ROOT / "clustering"
AUGMENTED_OUTPUT_PATH = OUTPUT_DIR / "kmeans_student_features.csv"
ELBOW_OUTPUT_PATH = OUTPUT_DIR / "kmeans_elbow.png"
CLUSTER_OUTPUT_PATH = OUTPUT_DIR / "kmeans_clusters.png"
CLUSTER_3D_OUTPUT_PATH = OUTPUT_DIR / "kmeans_clusters_3d.png"
CLUSTER_CMAP = "viridis"
ANALYSIS_OUTPUT_PATH = OUTPUT_DIR / "cluster_analysis.md"
TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 15
TICK_FONT_SIZE = 13
LEGEND_FONT_SIZE = 13
CHINESE_FONT_FAMILY = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "KaiTi", "Arial Unicode MS"]
RELATIVE_NORMAL_THRESHOLD = 0.8

plt.rcParams["font.sans-serif"] = CHINESE_FONT_FAMILY + plt.rcParams["font.sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

TOTAL_STUDENTS = 200
RANDOM_SEED = 42

ID_COLUMNS = [
    "\u8ab2\u7a0b\u4ee3\u78bc",
    "\u8ab2\u7a0b\u540d\u7a31",
    "\u5b78\u751f\u5b78\u865f",
    "\u5b78\u751f\u59d3\u540d",
]

REALTIME_METRICS = [
    "focus-ratio",
    "head-stability",
    "fatigue",
    "posture-angle",
    "desk-distance",
    "stillness",
    "hand-raise",
    "shared-attention",
]

OUTPUT_COLUMN_NAMES = {
    "course_id": "\u8ab2\u7a0b\u4ee3\u78bc",
    "course_name": "\u8ab2\u7a0b\u540d\u7a31",
    "student_id": "\u5b78\u751f\u5b78\u865f",
    "student_name": "\u5b78\u751f\u59d3\u540d",
    "focus-ratio_mean": "\u5c08\u6ce8\u5ea6\u5e73\u5747",
    "head-stability_mean": "\u982d\u90e8\u7a69\u5b9a\u5ea6\u5e73\u5747",
    "fatigue_mean": "\u75b2\u52de\u5ea6\u5e73\u5747",
    "posture-angle_mean": "\u8eab\u9ad4\u524d\u50be\u6295\u5165\u5ea6\u5e73\u5747",
    "desk-distance_mean": "\u982d\u8207\u684c\u8ddd\u96e2\u5e73\u5747",
    "stillness_mean": "\u767c\u5446\u6307\u6578\u5e73\u5747",
    "hand-raise_mean": "\u8209\u624b\u6bd4\u4f8b\u5e73\u5747",
    "shared-attention_mean": "\u5171\u540c\u6ce8\u610f\u529b\u5e73\u5747",
    "assignment_score_mean": "\u4f5c\u696d\u6210\u7e3e\u5e73\u5747",
}

SCORE_COLUMN = "\u4f5c\u696d\u6210\u7e3e\u5e73\u5747"

FEATURE_COLUMNS = [
    "\u5c08\u6ce8\u5ea6\u5e73\u5747",
    "\u982d\u90e8\u7a69\u5b9a\u5ea6\u5e73\u5747",
    "\u75b2\u52de\u5ea6\u5e73\u5747",
    "\u8eab\u9ad4\u524d\u50be\u6295\u5165\u5ea6\u5e73\u5747",
    "\u982d\u8207\u684c\u8ddd\u96e2\u5e73\u5747",
    "\u767c\u5446\u6307\u6578\u5e73\u5747",
    "\u8209\u624b\u6bd4\u4f8b\u5e73\u5747",
    "\u5171\u540c\u6ce8\u610f\u529b\u5e73\u5747",
]

FEATURE_PLOT_LABELS = [
    "專注度",
    "姿態穩定度",
    "疲勞度",
    "上課投入度",
    "專心距離",
    "發呆指數",
    "參與度",
    "互動模式",
]

SYNTHETIC_NUMERIC_COLUMNS = [*FEATURE_COLUMNS, SCORE_COLUMN]

VALUE_LIMITS = {
    "\u5c08\u6ce8\u5ea6\u5e73\u5747": (0, 100),
    "\u982d\u90e8\u7a69\u5b9a\u5ea6\u5e73\u5747": (0, 45),
    "\u75b2\u52de\u5ea6\u5e73\u5747": (0, 70),
    "\u8eab\u9ad4\u524d\u50be\u6295\u5165\u5ea6\u5e73\u5747": (0, 50),
    "\u982d\u8207\u684c\u8ddd\u96e2\u5e73\u5747": (20, 100),
    "\u767c\u5446\u6307\u6578\u5e73\u5747": (0, 60),
    "\u8209\u624b\u6bd4\u4f8b\u5e73\u5747": (0, 0.08),
    "\u5171\u540c\u6ce8\u610f\u529b\u5e73\u5747": (0, 1),
    "\u4f5c\u696d\u6210\u7e3e\u5e73\u5747": (60, 100),
}

FIRST_NAMES = [
    "\u9673",
    "\u6797",
    "\u9ec3",
    "\u5f35",
    "\u674e",
    "\u738b",
    "\u5433",
    "\u5289",
    "\u8521",
    "\u694a",
    "\u8a31",
    "\u912d",
    "\u8b1d",
    "\u90ed",
    "\u8cf4",
    "\u66fe",
    "\u5468",
    "\u8449",
    "\u6c5f",
    "\u8607",
]
GIVEN_NAMES = [
    "\u5b50\u8ed2",
    "\u5b87\u7fd4",
    "\u5bb6\u8c6a",
    "\u627f\u6069",
    "\u54c1\u777f",
    "\u51a0\u5ef7",
    "\u5b9c\u84c1",
    "\u82b7\u6674",
    "\u601d\u59a4",
    "\u5fc3\u6021",
    "\u4f73\u7a4e",
    "\u5ead\u7444",
    "\u5955\u8fb0",
    "\u67cf\u8c6a",
    "\u5b89\u5ef7",
    "\u96c5\u5a77",
    "\u660e\u8ed2",
    "\u5b8f\u5049",
    "\u54f2\u7dad",
    "\u80b2\u5ead",
]

PROFILE_NAMES = ["high_engagement", "steady", "at_risk", "active_but_unstable"]
PROFILE_CENTERS = {
    "high_engagement": {
        "\u5c08\u6ce8\u5ea6\u5e73\u5747": 82,
        "\u982d\u90e8\u7a69\u5b9a\u5ea6\u5e73\u5747": 24,
        "\u75b2\u52de\u5ea6\u5e73\u5747": 8,
        "\u8eab\u9ad4\u524d\u50be\u6295\u5165\u5ea6\u5e73\u5747": 28,
        "\u982d\u8207\u684c\u8ddd\u96e2\u5e73\u5747": 38,
        "\u767c\u5446\u6307\u6578\u5e73\u5747": 4,
        "\u8209\u624b\u6bd4\u4f8b\u5e73\u5747": 0.02,
        "\u5171\u540c\u6ce8\u610f\u529b\u5e73\u5747": 0.96,
        "\u4f5c\u696d\u6210\u7e3e\u5e73\u5747": 91,
        "\u6700\u65b0\u4f5c\u696d\u6210\u7e3e": 94,
    },
    "steady": {
        "\u5c08\u6ce8\u5ea6\u5e73\u5747": 68,
        "\u982d\u90e8\u7a69\u5b9a\u5ea6\u5e73\u5747": 13,
        "\u75b2\u52de\u5ea6\u5e73\u5747": 22,
        "\u8eab\u9ad4\u524d\u50be\u6295\u5165\u5ea6\u5e73\u5747": 7,
        "\u982d\u8207\u684c\u8ddd\u96e2\u5e73\u5747": 65,
        "\u767c\u5446\u6307\u6578\u5e73\u5747": 10,
        "\u8209\u624b\u6bd4\u4f8b\u5e73\u5747": 0.01,
        "\u5171\u540c\u6ce8\u610f\u529b\u5e73\u5747": 0.9,
        "\u4f5c\u696d\u6210\u7e3e\u5e73\u5747": 84,
        "\u6700\u65b0\u4f5c\u696d\u6210\u7e3e": 87,
    },
    "at_risk": {
        "\u5c08\u6ce8\u5ea6\u5e73\u5747": 42,
        "\u982d\u90e8\u7a69\u5b9a\u5ea6\u5e73\u5747": 16,
        "\u75b2\u52de\u5ea6\u5e73\u5747": 36,
        "\u8eab\u9ad4\u524d\u50be\u6295\u5165\u5ea6\u5e73\u5747": 7,
        "\u982d\u8207\u684c\u8ddd\u96e2\u5e73\u5747": 82,
        "\u767c\u5446\u6307\u6578\u5e73\u5747": 38,
        "\u8209\u624b\u6bd4\u4f8b\u5e73\u5747": 0,
        "\u5171\u540c\u6ce8\u610f\u529b\u5e73\u5747": 0.78,
        "\u4f5c\u696d\u6210\u7e3e\u5e73\u5747": 78,
        "\u6700\u65b0\u4f5c\u696d\u6210\u7e3e": 80,
    },
    "active_but_unstable": {
        "\u5c08\u6ce8\u5ea6\u5e73\u5747": 58,
        "\u982d\u90e8\u7a69\u5b9a\u5ea6\u5e73\u5747": 27,
        "\u75b2\u52de\u5ea6\u5e73\u5747": 18,
        "\u8eab\u9ad4\u524d\u50be\u6295\u5165\u5ea6\u5e73\u5747": 18,
        "\u982d\u8207\u684c\u8ddd\u96e2\u5e73\u5747": 55,
        "\u767c\u5446\u6307\u6578\u5e73\u5747": 16,
        "\u8209\u624b\u6bd4\u4f8b\u5e73\u5747": 0.04,
        "\u5171\u540c\u6ce8\u610f\u529b\u5e73\u5747": 0.88,
        "\u4f5c\u696d\u6210\u7e3e\u5e73\u5747": 86,
        "\u6700\u65b0\u4f5c\u696d\u6210\u7e3e": 88,
    },
}


def load_base_data(input_paths: list[Path] | None = None) -> pd.DataFrame:
    input_paths = input_paths or resolve_raw_input_paths()
    df = format_history_for_kmeans(input_paths)
    missing_columns = [column for column in [*ID_COLUMNS, *FEATURE_COLUMNS, SCORE_COLUMN] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")
    return df


def resolve_raw_input_paths() -> list[Path]:
    candidates = sorted(HISTORY_DIR.glob(RAW_INPUT_GLOB))
    if not candidates:
        raise FileNotFoundError(f"No input CSV matched {HISTORY_DIR / RAW_INPUT_GLOB}")
    return candidates


def format_history_for_kmeans(input_paths: list[Path]) -> pd.DataFrame:
    df = pd.concat((pd.read_csv(path) for path in input_paths), ignore_index=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    student_columns = ["course_id", "course_name", "student_id", "student_name"]
    students = (
        df[student_columns]
        .drop_duplicates(subset=["student_id"])
        .set_index("student_id")
    )

    realtime = df[df["metric_key"].isin(REALTIME_METRICS)]
    realtime_features = realtime.pivot_table(
        index="student_id",
        columns="metric_key",
        values="value",
        aggfunc="mean",
    )
    realtime_features = realtime_features.reindex(columns=REALTIME_METRICS)
    realtime_features.columns = [f"{column}_mean" for column in realtime_features.columns]

    scores = df[df["metric_key"] == "assignment-score"].copy()
    score_mean = scores.groupby("student_id")["value"].mean().rename("assignment_score_mean")

    formatted = students.join([realtime_features, score_mean]).reset_index()
    ordered_columns = [
        "course_id",
        "course_name",
        "student_id",
        "student_name",
        *[f"{metric}_mean" for metric in REALTIME_METRICS],
        "assignment_score_mean",
    ]
    formatted = formatted[ordered_columns].sort_values("student_id")
    return formatted.rename(columns=OUTPUT_COLUMN_NAMES)


def generate_similar_students(base_df: pd.DataFrame, total_students: int = TOTAL_STUDENTS) -> pd.DataFrame:
    if len(base_df) >= total_students:
        return base_df.head(total_students).copy()

    rng = np.random.default_rng(RANDOM_SEED)
    synthetic_count = total_students - len(base_df)
    used_student_ids = set(base_df["\u5b78\u751f\u5b78\u865f"].astype(str))
    synthetic_rows = []
    for index in range(synthetic_count):
        source = base_df.iloc[index % len(base_df)].copy()
        new_row = source.copy()
        profile = PROFILE_NAMES[index % len(PROFILE_NAMES)]
        center = PROFILE_CENTERS[profile]

        for column in SYNTHETIC_NUMERIC_COLUMNS:
            low, high = VALUE_LIMITS[column]
            noise_scale = (high - low) * 0.11
            if column in {
                "\u8209\u624b\u6bd4\u4f8b\u5e73\u5747",
                "\u5171\u540c\u6ce8\u610f\u529b\u5e73\u5747",
            }:
                noise_scale = (high - low) * 0.065
            value = center[column] + rng.normal(0, noise_scale)
            new_row[column] = round(float(np.clip(value, low, high)), 2)

        new_row["\u5b78\u751f\u5b78\u865f"] = random_student_id(rng, used_student_ids)
        new_row["\u5b78\u751f\u59d3\u540d"] = random_student_name(rng)
        synthetic_rows.append(new_row)

    return pd.concat([base_df, pd.DataFrame(synthetic_rows)], ignore_index=True)


def random_student_name(rng: np.random.Generator) -> str:
    first_name = FIRST_NAMES[int(rng.integers(0, len(FIRST_NAMES)))]
    given_name = GIVEN_NAMES[int(rng.integers(0, len(GIVEN_NAMES)))]
    return f"{first_name}{given_name}"


def random_student_id(rng: np.random.Generator, used_student_ids: set[str]) -> str:
    while True:
        prefix = str(rng.choice(["112", "144"]))
        suffix = int(rng.integers(3000, 7999))
        student_id = f"D{prefix}{suffix:04d}"
        if student_id not in used_student_ids:
            used_student_ids.add(student_id)
            return student_id


def choose_best_k(inertias: list[float], k_values: list[int]) -> int:
    points = np.column_stack([k_values, inertias])
    start = points[0]
    end = points[-1]
    line = end - start
    line_norm = np.linalg.norm(line)
    if line_norm == 0:
        return 3

    vectors = points - start
    distances = np.abs(line[0] * vectors[:, 1] - line[1] * vectors[:, 0]) / line_norm
    return int(k_values[int(np.argmax(distances))])


def run_kmeans(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int, list[int], list[float], np.ndarray, np.ndarray]:
    features = df[FEATURE_COLUMNS].astype(float)
    scaled_features = StandardScaler().fit_transform(features)

    max_k = min(10, len(df) - 1)
    k_values = list(range(1, max_k + 1))
    inertias = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
        model.fit(scaled_features)
        inertias.append(float(model.inertia_))

    best_k = choose_best_k(inertias, k_values)
    final_model = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=20)
    clustered = df.copy()
    clustered["cluster"] = final_model.fit_predict(scaled_features)

    pca_points_2d = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(scaled_features)
    pca_points_3d = PCA(n_components=3, random_state=RANDOM_SEED).fit_transform(scaled_features)
    return clustered, best_k, k_values, inertias, pca_points_2d, pca_points_3d


def save_elbow_plot(k_values: list[int], inertias: list[float], best_k: int) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o", linewidth=2)
    plt.axvline(best_k, color="tab:red", linestyle="--", label=f"chosen k = {best_k}")
    plt.xlabel("Number of clusters (k)", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Inertia", fontsize=LABEL_FONT_SIZE)
    plt.xticks(k_values)
    plt.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(ELBOW_OUTPUT_PATH, dpi=160)
    plt.close()


def save_cluster_plot(clustered: pd.DataFrame, pca_points: np.ndarray, best_k: int) -> None:
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        pca_points[:, 0],
        pca_points[:, 1],
        c=clustered["cluster"],
        cmap=CLUSTER_CMAP,
        s=82,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.35,
    )
    plt.xlabel("PCA 1", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("PCA 2", fontsize=LABEL_FONT_SIZE)
    plt.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    plt.grid(alpha=0.25)
    colorbar = plt.colorbar(scatter, label="Cluster")
    colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    colorbar.set_label("Cluster", fontsize=LABEL_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(CLUSTER_OUTPUT_PATH, dpi=160)
    plt.close()


def save_cluster_3d_plot(clustered: pd.DataFrame, pca_points_3d: np.ndarray, best_k: int) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        pca_points_3d[:, 0],
        pca_points_3d[:, 1],
        pca_points_3d[:, 2],
        c=clustered["cluster"],
        cmap=CLUSTER_CMAP,
        s=58,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.35,
    )
    ax.set_xlabel("Component 1", fontsize=LABEL_FONT_SIZE, labelpad=10)
    ax.set_ylabel("Component 2", fontsize=LABEL_FONT_SIZE, labelpad=10)
    ax.set_zlabel("Component 3", fontsize=LABEL_FONT_SIZE, labelpad=10)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.view_init(elev=24, azim=-58)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08, label="Cluster")
    colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    colorbar.set_label("Cluster", fontsize=LABEL_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(CLUSTER_3D_OUTPUT_PATH, dpi=170)
    plt.close()


def save_cluster_score_plots(clustered: pd.DataFrame) -> list[Path]:
    output_paths = []
    for cluster_id in sorted(clustered["cluster"].unique()):
        cluster_scores = clustered.loc[clustered["cluster"] == cluster_id, SCORE_COLUMN].astype(float)
        mean_score = cluster_scores.mean()
        median_score = cluster_scores.median()
        std_score = cluster_scores.std(ddof=0)

        output_path = OUTPUT_DIR / f"score_cluster_{cluster_id}.png"
        plt.figure(figsize=(8, 5))
        plt.hist(
            cluster_scores,
            bins=10,
            color=plt.get_cmap(CLUSTER_CMAP)(cluster_id / max(clustered["cluster"].max(), 1)),
            edgecolor="white",
            alpha=0.88,
        )
        plt.axvline(mean_score, color="tab:red", linestyle="--", linewidth=2, label=f"mean = {mean_score:.1f}")
        plt.axvline(median_score, color="black", linestyle=":", linewidth=2, label=f"median = {median_score:.1f}")
        plt.xlabel("Assignment score average", fontsize=LABEL_FONT_SIZE)
        plt.ylabel("Student count", fontsize=LABEL_FONT_SIZE)
        plt.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
        plt.grid(axis="y", alpha=0.25)
        plt.legend(fontsize=LEGEND_FONT_SIZE)
        plt.text(
            0.98,
            0.92,
            f"Cluster {cluster_id}\nn = {len(cluster_scores)}\nstd = {std_score:.1f}",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            fontsize=LABEL_FONT_SIZE,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        output_paths.append(output_path)

    return output_paths


def save_cluster_feature_plots(clustered: pd.DataFrame) -> list[Path]:
    output_paths = []
    feature_means = clustered.groupby("cluster")[FEATURE_COLUMNS].mean()
    global_means = clustered[FEATURE_COLUMNS].mean()
    global_stds = clustered[FEATURE_COLUMNS].std(ddof=0).replace(0, 1)

    for cluster_id, means in feature_means.iterrows():
        raw_output_path = OUTPUT_DIR / f"features_raw_cluster_{cluster_id}.png"
        relative_output_path = OUTPUT_DIR / f"features_relative_cluster_{cluster_id}.png"
        relative_mask_output_path = OUTPUT_DIR / f"features_relative_mask_cluster_{cluster_id}.png"
        color = plt.get_cmap(CLUSTER_CMAP)(cluster_id / max(clustered["cluster"].max(), 1))

        plt.figure(figsize=(11, 6))
        x = np.arange(len(FEATURE_COLUMNS))
        bar_width = 0.36
        overall_bars = plt.bar(
            x - bar_width / 2,
            global_means.values,
            color="#9ca3af",
            alpha=0.55,
            width=bar_width,
            edgecolor="white",
            label="Overall mean",
        )
        cluster_bars = plt.bar(
            x + bar_width / 2,
            means.values,
            color=color,
            alpha=0.92,
            width=bar_width,
            edgecolor="white",
            label=f"Cluster {cluster_id}",
        )
        for bars in (overall_bars, cluster_bars):
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + max(global_means.max(), means.max()) * 0.018,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        plt.xticks(x, FEATURE_PLOT_LABELS, rotation=28, ha="right", fontsize=12)
        plt.yticks(fontsize=TICK_FONT_SIZE)
        plt.ylabel("Average value", fontsize=LABEL_FONT_SIZE)
        plt.ylim(0, max(global_means.max(), means.max()) * 1.28)
        plt.grid(axis="y", alpha=0.25)
        plt.legend(fontsize=LEGEND_FONT_SIZE)
        plt.tight_layout()
        plt.savefig(raw_output_path, dpi=170)
        plt.close()
        output_paths.append(raw_output_path)

        differences = ((means - global_means) / global_stds).reindex(FEATURE_COLUMNS)
        colors = ["#2f80ed" if value >= 0 else "#eb5757" for value in differences.values]

        plt.figure(figsize=(9, 6))
        y = np.arange(len(FEATURE_COLUMNS))
        max_abs = max(abs(differences.min()), abs(differences.max()), 1)
        limit = max_abs * 1.25
        normal_threshold = RELATIVE_NORMAL_THRESHOLD
        plt.barh(y, differences.values, color=colors, alpha=0.88, edgecolor="white")
        for index, value in enumerate(differences.values):
            offset = 0.04 if value >= 0 else -0.04
            plt.text(
                value + offset,
                index,
                f"{value:+.2f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=11,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            )
        plt.axvline(0, color="#333333", linewidth=1.5)
        plt.yticks(y, FEATURE_PLOT_LABELS, fontsize=17)
        plt.xticks(fontsize=TICK_FONT_SIZE)
        plt.xlabel("Difference from overall average (standard deviations)", fontsize=LABEL_FONT_SIZE)
        plt.grid(axis="x", alpha=0.25)
        plt.xlim(-limit, limit)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(relative_output_path, dpi=170)
        plt.close()
        output_paths.append(relative_output_path)

        plt.figure(figsize=(9, 6))
        plt.axvspan(-limit, -normal_threshold, color="#eb5757", alpha=0.28, zorder=0)
        plt.axvspan(-normal_threshold, normal_threshold, color="#9ca3af", alpha=0.32, zorder=0)
        plt.axvspan(normal_threshold, limit, color="#2f80ed", alpha=0.28, zorder=0)
        plt.barh(y, differences.values, color=colors, alpha=0.9, edgecolor="white")
        for index, value in enumerate(differences.values):
            offset = 0.04 if value >= 0 else -0.04
            plt.text(
                value + offset,
                index,
                f"{value:+.2f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=11,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            )
        plt.axvline(0, color="#333333", linewidth=1.5)
        plt.yticks(y, FEATURE_PLOT_LABELS, fontsize=17)
        plt.xticks(fontsize=TICK_FONT_SIZE)
        plt.xlabel("Difference from overall average (standard deviations)", fontsize=17)
        plt.grid(axis="x", alpha=0.25)
        plt.xlim(-limit, limit)
        plt.gca().invert_yaxis()
        y_top = -0.72
        plt.text(-limit * 0.62, y_top, "過低", ha="center", va="center", fontsize=15, color="#9b2c2c")
        plt.text(0, y_top, "正常", ha="center", va="center", fontsize=15, color="#4b5563")
        plt.text(limit * 0.62, y_top, "過高", ha="center", va="center", fontsize=15, color="#1f4e8c")
        plt.tight_layout()
        plt.savefig(relative_mask_output_path, dpi=170)
        plt.close()
        output_paths.append(relative_mask_output_path)

    return output_paths


def save_cluster_analysis_report(clustered: pd.DataFrame) -> Path:
    feature_means = clustered.groupby("cluster")[FEATURE_COLUMNS].mean()
    global_means = clustered[FEATURE_COLUMNS].mean()
    global_stds = clustered[FEATURE_COLUMNS].std(ddof=0).replace(0, 1)
    score_summary = clustered.groupby("cluster")[SCORE_COLUMN].agg(["count", "mean", "median"])
    score_ranks = score_summary["mean"].rank(ascending=False, method="min").astype(int)

    lines = [
        "# K-means Cluster Analysis",
        "",
        f"- K-means features: {len(FEATURE_COLUMNS)} classroom performance indicators",
        f"- Score reference: {SCORE_COLUMN}",
        f"- Too low / normal / too high threshold: +/- {RELATIVE_NORMAL_THRESHOLD:.1f} standard deviations",
        "",
    ]

    for cluster_id in sorted(clustered["cluster"].unique()):
        scores = score_summary.loc[cluster_id]
        differences = ((feature_means.loc[cluster_id] - global_means) / global_stds).reindex(FEATURE_COLUMNS)
        too_high = differences[differences > RELATIVE_NORMAL_THRESHOLD]
        too_low = differences[differences < -RELATIVE_NORMAL_THRESHOLD]
        lines.extend(
            [
                f"## Cluster {cluster_id}",
                "",
                "### Score Summary",
                f"- Student count: {int(scores['count'])}",
                f"- Score rank: {score_ranks.loc[cluster_id]}",
                f"- Mean score: {scores['mean']:.2f}",
                f"- Median score: {scores['median']:.2f}",
                "",
                "### Too High Indicators",
            ]
        )

        if too_high.empty:
            lines.append("- None")
        else:
            for feature, value in too_high.sort_values(ascending=False).items():
                lines.append(f"- {feature}: {value:+.2f} std")

        lines.append("")
        lines.append("### Too Low Indicators")
        if too_low.empty:
            lines.append("- None")
        else:
            for feature, value in too_low.sort_values().items():
                lines.append(f"- {feature}: {value:+.2f} std")

        lines.append("")
        lines.append("### Analysis")
        lines.append(build_cluster_analysis_paragraph(cluster_id, scores, score_ranks.loc[cluster_id], too_high, too_low))
        lines.extend(["", ""])

    ANALYSIS_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")
    return ANALYSIS_OUTPUT_PATH


def build_cluster_analysis_paragraph(
    cluster_id: int,
    scores: pd.Series,
    score_rank: int,
    too_high: pd.Series,
    too_low: pd.Series,
) -> str:
    too_high_text = "、".join(too_high.index.tolist()) if not too_high.empty else "沒有明顯過高指標"
    too_low_text = "、".join(too_low.index.tolist()) if not too_low.empty else "沒有明顯過低指標"
    return (
        f"Cluster {cluster_id} 共有 {int(scores['count'])} 位學生，平均成績 {scores['mean']:.2f}，"
        f"中位數 {scores['median']:.2f}，在四群中排名第 {score_rank}。"
        f"本群過高指標為 {too_high_text}；過低指標為 {too_low_text}。"
        "整體來看，這些差異可作為理解該群學習型態的主要線索。"
        "若過高的是投入或專注相關指標，通常代表學習參與較佳；若過高的是疲勞、發呆或距離相關指標，"
        "則可能表示學習狀態需要追蹤。過低指標則可作為後續輔導或課堂提醒的切入點。"
    )


def main() -> None:
    input_paths = resolve_raw_input_paths()
    base_df = load_base_data(input_paths)
    augmented_df = generate_similar_students(base_df)
    clustered, best_k, k_values, inertias, pca_points, pca_points_3d = run_kmeans(augmented_df)

    AUGMENTED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(AUGMENTED_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    save_elbow_plot(k_values, inertias, best_k)
    save_cluster_plot(clustered, pca_points, best_k)
    save_cluster_3d_plot(clustered, pca_points_3d, best_k)
    score_plot_paths = save_cluster_score_plots(clustered)
    feature_plot_paths = save_cluster_feature_plots(clustered)
    analysis_report_path = save_cluster_analysis_report(clustered)

    print("Loaded source data:")
    for path in input_paths:
        print(f"  {path}")
    print(f"Generated {len(augmented_df)} students.")
    print(f"Best k by elbow heuristic: {best_k}")
    print(f"Saved augmented data: {AUGMENTED_OUTPUT_PATH}")
    print(f"Saved elbow plot: {ELBOW_OUTPUT_PATH}")
    print(f"Saved cluster plot: {CLUSTER_OUTPUT_PATH}")
    print(f"Saved 3D cluster plot: {CLUSTER_3D_OUTPUT_PATH}")
    print("Saved score analysis plots:")
    for path in score_plot_paths:
        print(f"  {path}")
    print("Saved feature analysis plots:")
    for path in feature_plot_paths:
        print(f"  {path}")
    print(f"Saved analysis report: {analysis_report_path}")


if __name__ == "__main__":
    main()
