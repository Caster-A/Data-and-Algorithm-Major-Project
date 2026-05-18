# 数据清洗、异常值处理与时间窗口聚合
# 负责人：成员二

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "data" / "submission"

TRAIN_VOLUME_PATH = RAW_DIR / "training" / "volume(table 6)_training.csv"
TEST_VOLUME_PATH = RAW_DIR / "testing_phase1" / "volume(table 6)_test1.csv"

WINDOW_FREQ = "20min"
VALID_COMBOS = [(1, 0), (1, 1), (2, 0), (3, 0), (3, 1)]

TRAIN_START = "2016-09-19 00:00:00"
TRAIN_END_EXCLUSIVE = "2016-10-18 00:00:00"
TEST_START = "2016-10-18 00:00:00"
TEST_END_EXCLUSIVE = "2016-10-25 00:00:00"
PREDICT_DATES = pd.date_range("2016-10-18", "2016-10-24", freq="D")


def read_volume(path: Path, dataset: str) -> pd.DataFrame:
    """读取逐车 volume 原始记录，并做基础字段清洗。"""
    df = pd.read_csv(path)
    df.columns = [col.strip().strip('"') for col in df.columns]

    required_cols = ["time", "tollgate_id", "direction"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} 缺少必要字段: {missing_cols}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["tollgate_id"] = pd.to_numeric(df["tollgate_id"], errors="coerce")
    df["direction"] = pd.to_numeric(df["direction"], errors="coerce")
    df = df.dropna(subset=["time", "tollgate_id", "direction"]).copy()

    df["tollgate_id"] = df["tollgate_id"].astype(int)
    df["direction"] = df["direction"].astype(int)
    df["dataset"] = dataset

    valid_combo_index = pd.MultiIndex.from_tuples(
        VALID_COMBOS, names=["tollgate_id", "direction"]
    )
    observed_combo_index = pd.MultiIndex.from_frame(df[["tollgate_id", "direction"]])
    df = df[observed_combo_index.isin(valid_combo_index)].copy()
    return df


def build_complete_index(
    start: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
    combos: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """生成收费站-方向组合和 20 分钟窗口的完整笛卡尔索引。"""
    combos = combos or VALID_COMBOS
    windows = pd.date_range(
        start=pd.Timestamp(start),
        end=pd.Timestamp(end_exclusive) - pd.Timedelta(WINDOW_FREQ),
        freq=WINDOW_FREQ,
    )

    window_df = pd.DataFrame({"time_window_start": np.repeat(windows, len(combos))})
    combo_df = pd.DataFrame(combos, columns=["tollgate_id", "direction"])
    combo_df = pd.concat([combo_df] * len(windows), ignore_index=True)
    return pd.concat([window_df, combo_df], axis=1)


def aggregate_volume(
    volume_df: pd.DataFrame,
    start: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
    dataset: str,
) -> pd.DataFrame:
    """将逐车通行记录聚合为 20 分钟粒度的流量表，并补齐空窗口。"""
    df = volume_df[
        (volume_df["time"] >= pd.Timestamp(start))
        & (volume_df["time"] < pd.Timestamp(end_exclusive))
    ].copy()
    df["time_window_start"] = df["time"].dt.floor(WINDOW_FREQ)

    grouped = (
        df.groupby(["time_window_start", "tollgate_id", "direction"])
        .size()
        .rename("volume")
        .reset_index()
    )

    complete = build_complete_index(start, end_exclusive)
    result = complete.merge(
        grouped,
        on=["time_window_start", "tollgate_id", "direction"],
        how="left",
    )
    result["volume"] = result["volume"].fillna(0).astype(int)
    result["dataset"] = dataset
    return add_time_features(result)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """补充后续特征工程常用的时间字段。"""
    result = df.copy()
    result["time_window_end"] = result["time_window_start"] + pd.Timedelta(WINDOW_FREQ)
    result["time_window"] = (
        "["
        + result["time_window_start"].dt.strftime("%Y-%m-%d %H:%M:%S")
        + ","
        + result["time_window_end"].dt.strftime("%Y-%m-%d %H:%M:%S")
        + ")"
    )
    result["date"] = result["time_window_start"].dt.date.astype(str)
    result["hour"] = result["time_window_start"].dt.hour
    result["minute"] = result["time_window_start"].dt.minute
    result["window_index"] = result["hour"] * 3 + result["minute"] // 20
    result["weekday"] = result["time_window_start"].dt.weekday
    result["is_weekend"] = result["weekday"].isin([5, 6]).astype(int)
    result["is_morning_peak"] = result["hour"].isin([8, 9]).astype(int)
    result["is_evening_peak"] = result["hour"].isin([17, 18]).astype(int)
    result["is_peak"] = (
        (result["is_morning_peak"] == 1) | (result["is_evening_peak"] == 1)
    ).astype(int)
    result["is_lead_window"] = (
        result["hour"].isin([6, 7]) | result["hour"].isin([15, 16])
    ).astype(int)
    return result


def mark_and_smooth_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """按组合和日内窗口标记异常流量，并给出只基于过去窗口的平滑参考列。"""
    result = df.sort_values(
        ["tollgate_id", "direction", "time_window_start"]
    ).copy()
    group_cols = ["tollgate_id", "direction", "window_index"]

    stats = (
        result.groupby(group_cols)["volume"]
        .agg(q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
        .reset_index()
    )
    stats["iqr"] = stats["q3"] - stats["q1"]
    stats["lower_bound"] = (stats["q1"] - 1.5 * stats["iqr"]).clip(lower=0)
    stats["upper_bound"] = stats["q3"] + 1.5 * stats["iqr"]

    result = result.merge(
        stats[group_cols + ["lower_bound", "upper_bound"]],
        on=group_cols,
        how="left",
    )
    result["is_outlier"] = (
        (result["volume"] < result["lower_bound"])
        | (result["volume"] > result["upper_bound"])
    ).astype(int)

    past_rolling_median = (
        result.groupby(["tollgate_id", "direction"])["volume"]
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).median())
    )
    past_rolling_median = (
        past_rolling_median.fillna(result["volume"])
        .round()
        .astype(int)
    )
    result["volume_smooth"] = np.where(
        result["is_outlier"] == 1, past_rolling_median, result["volume"]
    ).astype(int)
    result = result.drop(columns=["lower_bound", "upper_bound"])
    return result


def build_prediction_template() -> pd.DataFrame:
    """生成 2016-10-18 至 2016-10-24 高峰时段提交模板。"""
    starts = []
    for day in PREDICT_DATES:
        for hour in [8, 9, 17, 18]:
            for minute in [0, 20, 40]:
                starts.append(day + pd.Timedelta(hours=hour, minutes=minute))

    template = pd.DataFrame({"time_window_start": np.repeat(starts, len(VALID_COMBOS))})
    combo_df = pd.DataFrame(VALID_COMBOS, columns=["tollgate_id", "direction"])
    combo_df = pd.concat([combo_df] * len(starts), ignore_index=True)
    template = pd.concat([template, combo_df], axis=1)
    template = add_time_features(template)
    template["volume"] = np.nan
    return template[["tollgate_id", "time_window", "direction", "volume"]]


def save_outputs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = read_volume(TRAIN_VOLUME_PATH, dataset="train")
    test_raw = read_volume(TEST_VOLUME_PATH, dataset="test_phase1")

    train_20min = aggregate_volume(
        train_raw,
        start=TRAIN_START,
        end_exclusive=TRAIN_END_EXCLUSIVE,
        dataset="train",
    )
    train_20min = mark_and_smooth_outliers(train_20min)

    test_known_20min = aggregate_volume(
        test_raw,
        start=TEST_START,
        end_exclusive=TEST_END_EXCLUSIVE,
        dataset="test_phase1_known",
    )
    test_known_20min = test_known_20min[test_known_20min["is_lead_window"] == 1].copy()

    observed_20min = pd.concat([train_20min, test_known_20min], ignore_index=True)
    prediction_template = build_prediction_template()

    train_20min.to_csv(PROCESSED_DIR / "volume_train_20min.csv", index=False)
    test_known_20min.to_csv(PROCESSED_DIR / "volume_test_known_20min.csv", index=False)
    observed_20min.to_csv(PROCESSED_DIR / "volume_observed_20min.csv", index=False)
    prediction_template.to_csv(
        SUBMISSION_DIR / "submission_template_phase1.csv", index=False
    )

    print("Volume data processed.")
    print(f"train_20min: {len(train_20min)} rows")
    print(f"test_known_20min: {len(test_known_20min)} rows")
    print(f"prediction_template: {len(prediction_template)} rows")


def main() -> None:
    save_outputs()


if __name__ == "__main__":
    main()
