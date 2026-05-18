# 特征工程与滑动窗口样本构造
# 负责人：成员三

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "data" / "submission"

TRAIN_VOLUME_PATH = PROCESSED_DIR / "volume_train_20min.csv"
TEST_KNOWN_VOLUME_PATH = PROCESSED_DIR / "volume_test_known_20min.csv"
SUBMISSION_TEMPLATE_PATH = SUBMISSION_DIR / "submission_template_phase1.csv"

TRAIN_WEATHER_PATH = RAW_DIR / "training" / "weather (table 7)_training.csv"
TEST_WEATHER_PATH = RAW_DIR / "testing_phase1" / "weather (table 7)_test1.csv"
ROUTES_PATH = RAW_DIR / "training" / "routes (table 4).csv"
LINKS_PATH = RAW_DIR / "training" / "links (table 3).csv"

VALID_COMBOS = [(1, 0), (1, 1), (2, 0), (3, 0), (3, 1)]
WINDOW_FREQ = "20min"
LEAD_WINDOW_COUNT = 6
HORIZON_COUNT = 6
SESSION_CONFIGS = [
    {
        "session": "morning",
        "lead_hour": 6,
        "target_hour": 8,
    },
    {
        "session": "evening",
        "lead_hour": 15,
        "target_hour": 17,
    },
]


def load_volume(path: Path) -> pd.DataFrame:
    """读取 20 分钟流量表并规范时间字段。"""
    df = pd.read_csv(path)
    df["time_window_start"] = pd.to_datetime(df["time_window_start"])
    df["time_window_end"] = pd.to_datetime(df["time_window_end"])
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df


def make_time_window(start: pd.Timestamp) -> str:
    """生成提交文件要求的左闭右开时间窗口字符串。"""
    end = start + pd.Timedelta(WINDOW_FREQ)
    return f"[{start:%Y-%m-%d %H:%M:%S},{end:%Y-%m-%d %H:%M:%S})"


def build_session_samples(
    volume_df: pd.DataFrame,
    start_dates: pd.Series | list[str],
    include_target: bool,
) -> pd.DataFrame:
    """用高峰前 2 小时的 6 个窗口构造高峰期 6 个预测步长样本。"""
    volume_lookup = volume_df.set_index(
        ["time_window_start", "tollgate_id", "direction"]
    )["volume"]
    rows = []

    for date_value in sorted(pd.to_datetime(start_dates).unique()):
        day = pd.Timestamp(date_value).normalize()
        for session_config in SESSION_CONFIGS:
            lead_start = day + pd.Timedelta(hours=session_config["lead_hour"])
            target_start = day + pd.Timedelta(hours=session_config["target_hour"])
            lead_times = [
                lead_start + pd.Timedelta(minutes=20 * idx)
                for idx in range(LEAD_WINDOW_COUNT)
            ]

            for tollgate_id, direction in VALID_COMBOS:
                lead_values = []
                missing_leads = 0
                for lead_time in lead_times:
                    key = (lead_time, tollgate_id, direction)
                    value = volume_lookup.get(key, np.nan)
                    if pd.isna(value):
                        missing_leads += 1
                    lead_values.append(value)

                if missing_leads:
                    continue

                lag_features = {
                    f"lag_{LEAD_WINDOW_COUNT - idx}_volume": lead_values[idx]
                    for idx in range(LEAD_WINDOW_COUNT)
                }
                lead_array = np.asarray(lead_values, dtype=float)
                lead_stats = {
                    "lead_2h_sum": lead_array.sum(),
                    "lead_2h_mean": lead_array.mean(),
                    "lead_2h_max": lead_array.max(),
                    "lead_2h_min": lead_array.min(),
                    "lead_2h_std": lead_array.std(ddof=0),
                    "lead_1h_sum": lead_array[-3:].sum(),
                    "lead_1h_mean": lead_array[-3:].mean(),
                    "lead_trend": lead_array[-1] - lead_array[0],
                }

                for horizon_step in range(1, HORIZON_COUNT + 1):
                    target_time = target_start + pd.Timedelta(
                        minutes=20 * (horizon_step - 1)
                    )
                    row = {
                        "tollgate_id": tollgate_id,
                        "direction": direction,
                        "target_time": target_time,
                        "time_window": make_time_window(target_time),
                        "session": session_config["session"],
                        "horizon_step": horizon_step,
                        "date": target_time.date().isoformat(),
                        "hour": target_time.hour,
                        "minute": target_time.minute,
                        "window_index": target_time.hour * 3 + target_time.minute // 20,
                        "weekday": target_time.weekday(),
                        "is_weekend": int(target_time.weekday() in [5, 6]),
                        "is_morning_peak": int(session_config["session"] == "morning"),
                        "is_evening_peak": int(session_config["session"] == "evening"),
                        **lag_features,
                        **lead_stats,
                    }

                    if include_target:
                        target_key = (target_time, tollgate_id, direction)
                        target_volume = volume_lookup.get(target_key, np.nan)
                        if pd.isna(target_volume):
                            continue
                        row["target_volume"] = target_volume

                    rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["target_time", "tollgate_id", "direction"]
        ).reset_index(drop=True)
    return result


def load_weather_features() -> pd.DataFrame:
    """读取训练和测试天气，并转成可按时间向后匹配的特征表。"""
    weather_frames = []
    for path, dataset in [
        (TRAIN_WEATHER_PATH, "train"),
        (TEST_WEATHER_PATH, "test_phase1"),
    ]:
        df = pd.read_csv(path)
        df.columns = [col.strip().strip('"') for col in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        df["weather_time"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
        numeric_cols = [
            "pressure",
            "sea_pressure",
            "wind_direction",
            "wind_speed",
            "temperature",
            "rel_humidity",
            "precipitation",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["weather_dataset"] = dataset
        weather_frames.append(df[["weather_time", "weather_dataset"] + numeric_cols])

    weather = pd.concat(weather_frames, ignore_index=True)
    weather = (
        weather.sort_values("weather_time")
        .drop_duplicates(subset=["weather_time"], keep="last")
        .reset_index(drop=True)
    )
    return weather


def merge_weather(samples: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """为目标窗口匹配不晚于目标时间的最近一次天气观测。"""
    if samples.empty:
        return samples

    samples_sorted = samples.sort_values("target_time").reset_index(drop=True)
    weather_sorted = weather.sort_values("weather_time").reset_index(drop=True)
    merged = pd.merge_asof(
        samples_sorted,
        weather_sorted,
        left_on="target_time",
        right_on="weather_time",
        direction="backward",
    )
    merged["weather_observation_lag_hours"] = (
        (merged["target_time"] - merged["weather_time"]).dt.total_seconds() / 3600
    )
    return merged


def build_route_features() -> pd.DataFrame:
    """从 routes 和 links 构造收费站级静态拓扑聚合特征。"""
    routes = pd.read_csv(ROUTES_PATH)
    links = pd.read_csv(LINKS_PATH)
    routes.columns = [col.strip().strip('"') for col in routes.columns]
    links.columns = [col.strip().strip('"') for col in links.columns]

    links["link_id"] = pd.to_numeric(links["link_id"], errors="coerce").astype(int)
    for col in ["length", "width", "lanes", "lane_width"]:
        links[col] = pd.to_numeric(links[col], errors="coerce")
    link_lookup = links.set_index("link_id")

    route_rows = []
    for _, route in routes.iterrows():
        link_ids = [
            int(link_id)
            for link_id in str(route["link_seq"]).split(",")
            if str(link_id).strip()
        ]
        route_links = link_lookup.loc[link_ids]
        route_rows.append(
            {
                "intersection_id": route["intersection_id"],
                "tollgate_id": int(route["tollgate_id"]),
                "route_link_count": len(link_ids),
                "route_total_length": route_links["length"].sum(),
                "route_mean_width": route_links["width"].mean(),
                "route_min_width": route_links["width"].min(),
                "route_mean_lanes": route_links["lanes"].mean(),
                "route_min_lanes": route_links["lanes"].min(),
                "route_total_lanes": route_links["lanes"].sum(),
                "route_mean_lane_width": route_links["lane_width"].mean(),
            }
        )

    route_features = pd.DataFrame(route_rows)
    tollgate_features = (
        route_features.groupby("tollgate_id")
        .agg(
            route_count=("intersection_id", "count"),
            route_total_length_mean=("route_total_length", "mean"),
            route_total_length_min=("route_total_length", "min"),
            route_total_length_max=("route_total_length", "max"),
            route_link_count_mean=("route_link_count", "mean"),
            route_mean_width_mean=("route_mean_width", "mean"),
            route_min_width_min=("route_min_width", "min"),
            route_mean_lanes_mean=("route_mean_lanes", "mean"),
            route_min_lanes_min=("route_min_lanes", "min"),
            route_total_lanes_mean=("route_total_lanes", "mean"),
            route_mean_lane_width_mean=("route_mean_lane_width", "mean"),
        )
        .reset_index()
    )
    return tollgate_features


def finalize_features(samples: pd.DataFrame, route_features: pd.DataFrame) -> pd.DataFrame:
    """融合静态路线特征并整理类别字段。"""
    if samples.empty:
        return samples

    result = samples.merge(route_features, on="tollgate_id", how="left")
    result["combo_id"] = (
        result["tollgate_id"].astype(str) + "_" + result["direction"].astype(str)
    )
    result["session_id"] = result["session"].map({"morning": 0, "evening": 1})
    result = result.sort_values(["target_time", "tollgate_id", "direction"]).reset_index(
        drop=True
    )
    return result


def save_feature_tables() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_volume = load_volume(TRAIN_VOLUME_PATH)
    test_known_volume = load_volume(TEST_KNOWN_VOLUME_PATH)
    submission_template = pd.read_csv(SUBMISSION_TEMPLATE_PATH)

    weather = load_weather_features()
    route_features = build_route_features()

    train_samples = build_session_samples(
        train_volume,
        start_dates=train_volume["date"].drop_duplicates(),
        include_target=True,
    )
    predict_dates = (
        pd.to_datetime(
            submission_template["time_window"].str.extract(r"\[(.*?),")[0]
        )
        .dt.date.astype(str)
        .drop_duplicates()
    )
    predict_samples = build_session_samples(
        test_known_volume,
        start_dates=predict_dates,
        include_target=False,
    )

    train_features = finalize_features(
        merge_weather(train_samples, weather), route_features
    )
    predict_features = finalize_features(
        merge_weather(predict_samples, weather), route_features
    )

    train_features.to_csv(PROCESSED_DIR / "train_features.csv", index=False)
    predict_features.to_csv(PROCESSED_DIR / "predict_features_phase1.csv", index=False)
    route_features.to_csv(PROCESSED_DIR / "route_topology_features.csv", index=False)

    print("Feature tables built.")
    print(f"train_features: {len(train_features)} rows")
    print(f"predict_features_phase1: {len(predict_features)} rows")
    print(f"route_topology_features: {len(route_features)} rows")


def main() -> None:
    save_feature_tables()


if __name__ == "__main__":
    main()
