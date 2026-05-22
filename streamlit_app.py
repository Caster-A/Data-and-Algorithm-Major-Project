from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
OBSERVED_PATH = PROJECT_ROOT / "data" / "processed" / "volume_observed_20min.csv"
SUBMISSION_PATH = PROJECT_ROOT / "data" / "submission" / "submission_phase1.csv"

VALID_COMBOS = [(1, 0), (1, 1), (2, 0), (3, 0), (3, 1)]
DIRECTION_LABELS = {0: "Entry", 1: "Exit"}
SOURCE_LABELS = {
    "observed": "历史观测",
    "prediction": "预测结果",
}
SESSION_OPTIONS = {
    "全部": None,
    "上午 08:00-10:00": "morning_peak",
    "下午 17:00-19:00": "evening_peak",
    "先导窗口 06:00-08:00 / 15:00-17:00": "lead",
}


def parse_time_window_start(series: pd.Series) -> pd.Series:
    """Parse the left boundary from strings like [start,end)."""
    return pd.to_datetime(series.astype(str).str.extract(r"\[(.*?),")[0])


def make_time_window(start: pd.Series) -> pd.Series:
    end = start + pd.Timedelta(minutes=20)
    return (
        "["
        + start.dt.strftime("%Y-%m-%d %H:%M:%S")
        + ","
        + end.dt.strftime("%Y-%m-%d %H:%M:%S")
        + ")"
    )


def normalize_volume_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    result = df.copy()
    if "time_window_start" in result.columns:
        result["time_window_start"] = pd.to_datetime(result["time_window_start"])
    elif "time_window" in result.columns:
        result["time_window_start"] = parse_time_window_start(result["time_window"])
    else:
        raise ValueError("数据缺少 time_window_start 或 time_window 字段。")

    result["time_window_end"] = result["time_window_start"] + pd.Timedelta(minutes=20)
    if "time_window" not in result.columns:
        result["time_window"] = make_time_window(result["time_window_start"])

    result["tollgate_id"] = pd.to_numeric(result["tollgate_id"], errors="coerce")
    result["direction"] = pd.to_numeric(result["direction"], errors="coerce")
    result["volume"] = pd.to_numeric(result["volume"], errors="coerce")
    result = result.dropna(subset=["tollgate_id", "direction", "volume"])
    result["tollgate_id"] = result["tollgate_id"].astype(int)
    result["direction"] = result["direction"].astype(int)
    result = result[result[["tollgate_id", "direction"]].apply(tuple, axis=1).isin(VALID_COMBOS)]

    result["source"] = source
    result["source_label"] = SOURCE_LABELS[source]
    result["date"] = result["time_window_start"].dt.date.astype(str)
    result["hour"] = result["time_window_start"].dt.hour
    result["minute"] = result["time_window_start"].dt.minute
    result["time_label"] = result["time_window_start"].dt.strftime("%H:%M")
    result["direction_label"] = result["direction"].map(DIRECTION_LABELS)
    result["combo_label"] = (
        "Tollgate "
        + result["tollgate_id"].astype(str)
        + " - "
        + result["direction_label"]
    )
    return result[
        [
            "time_window_start",
            "time_window_end",
            "time_window",
            "tollgate_id",
            "direction",
            "volume",
            "source",
            "source_label",
            "date",
            "hour",
            "minute",
            "time_label",
            "direction_label",
            "combo_label",
        ]
    ]


@st.cache_data(show_spinner=False)
def load_traffic_data() -> pd.DataFrame:
    missing = [path for path in [OBSERVED_PATH, SUBMISSION_PATH] if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {path.relative_to(PROJECT_ROOT)}" for path in missing)
        raise FileNotFoundError(
            "缺少可视化所需数据文件：\n"
            f"{missing_text}\n\n"
            "请先按顺序运行：python src/data_cleaning.py、"
            "python src/feature_engineering.py、python src/train_model.py。"
        )

    observed = normalize_volume_frame(pd.read_csv(OBSERVED_PATH), "observed")
    prediction = normalize_volume_frame(pd.read_csv(SUBMISSION_PATH), "prediction")
    return (
        pd.concat([observed, prediction], ignore_index=True)
        .sort_values(["time_window_start", "tollgate_id", "direction", "source"])
        .reset_index(drop=True)
    )


def filter_session(df: pd.DataFrame, session_key: str | None) -> pd.DataFrame:
    if session_key is None:
        return df
    if session_key == "morning_peak":
        return df[df["hour"].isin([8, 9])]
    if session_key == "evening_peak":
        return df[df["hour"].isin([17, 18])]
    if session_key == "lead":
        return df[df["hour"].isin([6, 7, 15, 16])]
    return df


def volume_to_width(volume: float, max_volume: float) -> float:
    if max_volume <= 0:
        return 3
    return 3 + 10 * max(float(volume), 0) / max_volume


def add_curved_route(
    fig: go.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    width: float,
    name: str,
    label: str,
    bend: float,
) -> None:
    x0, y0 = start
    x1, y1 = end
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2 + bend
    fig.add_trace(
        go.Scatter(
            x=[x0, mid_x, x1],
            y=[y0, mid_y, y1],
            mode="lines+markers",
            line={"color": color, "width": width, "shape": "spline"},
            marker={
                "size": [0, 0, 12],
                "symbol": "triangle-right",
                "color": color,
                "line": {"width": 0},
            },
            name=name,
            text=[label, label, label],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )


def build_network_figure(current_df: pd.DataFrame, global_max: float) -> go.Figure:
    fig = go.Figure()
    background = "#30343d"
    entry_color = "#53c7f0"
    exit_color = "#f06d76"

    intersections = {
        "Intersection A": (0.4, 2.2),
        "Intersection B": (9.6, 1.7),
        "Intersection C": (9.4, 4.7),
    }
    tollgates = {
        1: (3.2, 3.65),
        2: (3.0, 1.55),
        3: (6.3, 2.2),
    }
    route_specs = {
        (1, 0): ((9.4, 4.7), tollgates[1], entry_color, 0.35),
        (1, 1): (tollgates[1], (0.4, 2.2), exit_color, 0.55),
        (2, 0): ((0.4, 2.2), tollgates[2], entry_color, -0.15),
        (3, 0): ((0.4, 2.2), tollgates[3], entry_color, -0.55),
        (3, 1): (tollgates[3], (9.6, 1.7), exit_color, -0.15),
    }
    current_lookup = {
        (int(row.tollgate_id), int(row.direction)): float(row.volume)
        for row in current_df.itertuples()
    }

    for (tollgate_id, direction), (start, end, color, bend) in route_specs.items():
        volume = current_lookup.get((tollgate_id, direction), 0)
        direction_label = DIRECTION_LABELS[direction]
        add_curved_route(
            fig,
            start=start,
            end=end,
            color=color,
            width=volume_to_width(volume, global_max),
            name=f"Tollgate {tollgate_id} {direction_label}",
            label=f"Tollgate {tollgate_id} {direction_label}: {volume:.1f}",
            bend=bend,
        )

    for name, (x, y) in intersections.items():
        fig.add_shape(
            type="rect",
            x0=x - 0.18,
            x1=x + 0.18,
            y0=y - 0.7,
            y1=y + 0.7,
            fillcolor="#f4d774",
            line={"color": "#f4d774"},
        )
        fig.add_annotation(
            x=x,
            y=y - 0.9 if name != "Intersection C" else y + 0.9,
            text=name,
            showarrow=False,
            font={"color": "#f4d774", "size": 15},
        )

    for tollgate_id, (x, y) in tollgates.items():
        fig.add_shape(
            type="circle",
            x0=x - 0.45,
            x1=x + 0.45,
            y0=y - 0.45,
            y1=y + 0.45,
            fillcolor="rgba(210, 214, 219, 0.62)",
            line={"color": "rgba(210, 214, 219, 0.2)"},
        )
        fig.add_annotation(
            x=x - 0.72,
            y=y + 0.68,
            text=f"Tollgate {tollgate_id}",
            showarrow=False,
            font={"color": "#e5e7eb", "size": 17},
        )

    label_offsets = {
        (1, 0): (3.95, 3.95),
        (1, 1): (2.45, 3.15),
        (2, 0): (3.65, 1.25),
        (3, 0): (6.9, 2.75),
        (3, 1): (6.9, 1.55),
    }
    for combo, (x, y) in label_offsets.items():
        volume = current_lookup.get(combo, 0)
        tollgate_id, direction = combo
        fig.add_annotation(
            x=x,
            y=y,
            text=f"T{tollgate_id} {DIRECTION_LABELS[direction]}<br><b>{volume:.0f}</b>",
            showarrow=False,
            font={"color": "#f9fafb", "size": 13},
            bgcolor="rgba(17, 24, 39, 0.68)",
            bordercolor="rgba(255,255,255,0.18)",
            borderwidth=1,
            borderpad=4,
        )

    fig.add_trace(
        go.Scatter(
            x=[0.25, 0.85],
            y=[0.45, 0.45],
            mode="lines",
            line={"color": entry_color, "width": 4},
            name="Highway Entry",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.25, 0.85],
            y=[0.2, 0.2],
            mode="lines",
            line={"color": exit_color, "width": 4},
            name="Highway Exit",
        )
    )
    fig.add_annotation(
        x=1.55,
        y=0.45,
        text="Highway Entry",
        showarrow=False,
        font={"color": entry_color, "size": 15},
    )
    fig.add_annotation(
        x=1.48,
        y=0.2,
        text="Highway Exit",
        showarrow=False,
        font={"color": exit_color, "size": 15},
    )
    fig.add_annotation(
        x=0.55,
        y=4.95,
        text="<b>IN</b>",
        showarrow=False,
        font={"color": "#f9fafb", "size": 24},
    )

    fig.update_layout(
        height=520,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor=background,
        plot_bgcolor=background,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.02,
            "xanchor": "left",
            "x": 0.02,
            "font": {"color": "#e5e7eb"},
        },
        xaxis={"visible": False, "range": [0, 10]},
        yaxis={"visible": False, "range": [0, 5.3], "scaleanchor": "x", "scaleratio": 1},
    )
    return fig


def render_metric_cards(current_df: pd.DataFrame) -> None:
    total_volume = current_df["volume"].sum()
    max_row = current_df.sort_values("volume", ascending=False).head(1)
    max_combo = "无数据"
    max_value = 0.0
    if not max_row.empty:
        max_combo = max_row.iloc[0]["combo_label"]
        max_value = float(max_row.iloc[0]["volume"])

    col1, col2, col3 = st.columns(3)
    col1.metric("当前窗口总流量", f"{total_volume:.1f}")
    col2.metric("最高组合", max_combo)
    col3.metric("最高流量", f"{max_value:.1f}")


def build_bar_chart(current_df: pd.DataFrame) -> go.Figure:
    plot_df = current_df.sort_values(["tollgate_id", "direction"]).copy()
    fig = px.bar(
        plot_df,
        x="combo_label",
        y="volume",
        color="direction_label",
        color_discrete_map={"Entry": "#53c7f0", "Exit": "#f06d76"},
        text="volume",
        labels={"combo_label": "收费站-方向组合", "volume": "车流量", "direction_label": "方向"},
    )
    fig.update_traces(
        texttemplate="%{y:.1f}",
        textposition="outside",
        hovertemplate="%{x}<br>车流量=%{y:.1f}<extra></extra>",
    )
    fig.update_layout(height=360, margin={"l": 10, "r": 10, "t": 25, "b": 10})
    return fig


def build_trend_chart(day_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        day_df.sort_values("time_window_start"),
        x="time_window_start",
        y="volume",
        color="combo_label",
        line_dash="source_label",
        markers=True,
        labels={
            "time_window_start": "时间窗口",
            "volume": "车流量",
            "combo_label": "收费站-方向组合",
            "source_label": "数据来源",
        },
    )
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d %H:%M}<br>车流量=%{y:.1f}<extra></extra>")
    fig.update_layout(height=430, margin={"l": 10, "r": 10, "t": 25, "b": 10})
    return fig


def build_heatmap(day_df: pd.DataFrame) -> go.Figure:
    heatmap_df = (
        day_df.groupby(["combo_label", "time_label"], as_index=False)["volume"]
        .mean()
        .pivot(index="combo_label", columns="time_label", values="volume")
        .sort_index()
    )
    fig = px.imshow(
        heatmap_df,
        aspect="auto",
        color_continuous_scale="YlGnBu",
        labels={"x": "20 分钟窗口", "y": "收费站-方向组合", "color": "平均车流量"},
    )
    fig.update_layout(height=360, margin={"l": 10, "r": 10, "t": 25, "b": 10})
    return fig


def select_default_date(df: pd.DataFrame, available_dates: list[str]) -> str:
    prediction_dates = sorted(df.loc[df["source"] == "prediction", "date"].unique())
    if prediction_dates:
        return prediction_dates[0]
    return available_dates[0]


def main() -> None:
    st.set_page_config(page_title="高速收费站车流量可视化", layout="wide")
    st.title("高速收费站车流量可视化")
    st.caption("历史观测 + XGBoost 预测结果，按 20 分钟窗口展示收费站方向组合流量。")

    try:
        traffic_df = load_traffic_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"数据读取失败：{exc}")
        return

    with st.sidebar:
        st.header("筛选条件")
        source_labels = st.multiselect(
            "数据来源",
            options=list(SOURCE_LABELS.values()),
            default=list(SOURCE_LABELS.values()),
        )
        source_values = [
            source for source, label in SOURCE_LABELS.items() if label in source_labels
        ]
        source_df = traffic_df[traffic_df["source"].isin(source_values)]
        if source_df.empty:
            st.warning("当前数据来源筛选无结果。")
            return

        available_dates = sorted(source_df["date"].unique())
        default_date = select_default_date(source_df, available_dates)
        selected_date = st.selectbox(
            "日期",
            options=available_dates,
            index=available_dates.index(default_date),
        )
        selected_session_label = st.selectbox("时段", options=list(SESSION_OPTIONS.keys()), index=0)
        session_key = SESSION_OPTIONS[selected_session_label]

        filtered_base = filter_session(source_df[source_df["date"] == selected_date], session_key)
        if filtered_base.empty:
            st.warning("当前日期和时段没有可展示数据。")
            return

        available_times = sorted(filtered_base["time_label"].unique())
        selected_time = st.selectbox("20 分钟窗口", options=available_times, index=0)

        combo_labels = sorted(filtered_base["combo_label"].unique())
        selected_combos = st.multiselect("收费站-方向组合", combo_labels, default=combo_labels)

    display_df = filtered_base[filtered_base["combo_label"].isin(selected_combos)].copy()
    if display_df.empty:
        st.info("当前筛选条件没有数据，请调整日期、时段或收费站方向组合。")
        return

    current_df = display_df[display_df["time_label"] == selected_time].copy()
    current_df = (
        current_df.sort_values(["source", "tollgate_id", "direction"])
        .drop_duplicates(subset=["tollgate_id", "direction"], keep="last")
    )
    if current_df.empty:
        st.info("当前时间窗口没有数据，请选择其他窗口。")
        return

    window_title = current_df.iloc[0]["time_window"]
    source_title = "、".join(sorted(current_df["source_label"].unique()))
    st.subheader(f"{selected_date} {selected_time} 窗口车流量")
    st.caption(f"时间窗口：{window_title} | 数据来源：{source_title}")
    render_metric_cards(current_df)

    global_max = float(traffic_df["volume"].max())
    st.plotly_chart(build_network_figure(current_df, global_max), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 当前窗口组合流量")
        st.plotly_chart(build_bar_chart(current_df), use_container_width=True)
    with right:
        st.markdown("#### 当前筛选数据")
        table_df = current_df[
            ["source_label", "time_window", "tollgate_id", "direction_label", "volume"]
        ].rename(
            columns={
                "source_label": "数据来源",
                "time_window": "时间窗口",
                "tollgate_id": "收费站",
                "direction_label": "方向",
                "volume": "车流量",
            }
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.markdown("#### 选中日期内趋势")
    st.plotly_chart(build_trend_chart(display_df), use_container_width=True)

    st.markdown("#### 选中日期内时间窗口热力图")
    st.plotly_chart(build_heatmap(display_df), use_container_width=True)


if __name__ == "__main__":
    main()
