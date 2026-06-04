# 基于图神经网络的道路流量预测
# 独立于现有 XGBoost 流水线，输出单独的 GNN 验证结果和提交文件。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils import calculate_mape, calculate_rmse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "data" / "submission"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "train_features.csv"
PREDICT_FEATURES_PATH = PROCESSED_DIR / "predict_features_phase1.csv"
TRAIN_VOLUME_PATH = PROCESSED_DIR / "volume_train_20min.csv"
SUBMISSION_TEMPLATE_PATH = SUBMISSION_DIR / "submission_template_phase1.csv"

VALIDATION_START = pd.Timestamp("2016-10-11")
VALIDATION_END = pd.Timestamp("2016-10-18")
COMBOS = [(1, 0), (1, 1), (2, 0), (3, 0), (3, 1)]
COMBO_IDS = [f"{tollgate_id}_{direction}" for tollgate_id, direction in COMBOS]
LAG_COLS = [f"lag_{idx}_volume" for idx in range(6, 0, -1)]
WEATHER_COLS = [
    "pressure",
    "sea_pressure",
    "wind_direction",
    "wind_speed",
    "temperature",
    "rel_humidity",
    "precipitation",
]
ROUTE_COLS = [
    "route_count",
    "route_total_length_mean",
    "route_link_count_mean",
    "route_mean_width_mean",
    "route_mean_lanes_mean",
    "route_total_lanes_mean",
]
SUBMISSION_COLUMNS = ["tollgate_id", "time_window", "direction", "volume"]


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


@dataclass
class TargetScaler:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.log1p(values) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        # 训练样本较少，神经网络偶尔会给出过大的标准化输出；反变换前裁剪
        # 到交通流量的合理 log 区间，避免 exp 溢出影响提交文件。
        log_values = np.clip(values * self.std + self.mean, 0.0, np.log1p(500.0))
        return np.expm1(log_values)


@dataclass
class GCNMLPWeights:
    graph_weight: np.ndarray
    graph_bias: np.ndarray
    dense_weight: np.ndarray
    dense_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray


@dataclass
class AdamState:
    m: list[np.ndarray]
    v: list[np.ndarray]
    step: int = 0


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def build_normalized_adjacency(
    train_volume: pd.DataFrame,
    cutoff_time: pd.Timestamp | None = None,
) -> np.ndarray:
    """用历史流量相关性构造 5 节点图，并做 GCN 归一化。"""
    volume = train_volume.copy()
    if cutoff_time is not None:
        volume = volume[volume["time_window_start"] < cutoff_time].copy()
    volume["combo_id"] = (
        volume["tollgate_id"].astype(str) + "_" + volume["direction"].astype(str)
    )
    pivot = (
        volume.pivot_table(
            index="time_window_start",
            columns="combo_id",
            values="volume",
            aggfunc="mean",
        )
        .reindex(columns=COMBO_IDS)
        .fillna(0.0)
    )
    corr = pivot.corr().fillna(0.0).to_numpy(dtype=float)
    adjacency = np.clip(corr, a_min=0.0, a_max=None)
    np.fill_diagonal(adjacency, 1.0)

    degrees = adjacency.sum(axis=1)
    inv_sqrt_degree = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-6)))
    return inv_sqrt_degree @ adjacency @ inv_sqrt_degree


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_features = pd.read_csv(TRAIN_FEATURES_PATH, parse_dates=["target_time"])
    predict_features = pd.read_csv(PREDICT_FEATURES_PATH, parse_dates=["target_time"])
    train_volume = pd.read_csv(TRAIN_VOLUME_PATH, parse_dates=["time_window_start"])
    submission_template = pd.read_csv(SUBMISSION_TEMPLATE_PATH)
    return train_features, predict_features, train_volume, submission_template


def _row_to_input_features(row: pd.Series, step_idx: int, lag_value: float) -> list[float]:
    """构造单个时间步、单个节点的输入特征。"""
    base_features = [
        lag_value,
        step_idx / 5.0,
        row["session_id"],
        row["weekday"] / 6.0,
        row["is_weekend"],
        row["tollgate_id"] / 3.0,
        row["direction"],
        row["combo_code"] / 4.0,
    ]
    weather_features = [row[col] for col in WEATHER_COLS]
    route_features = [row[col] for col in ROUTE_COLS]
    return base_features + weather_features + route_features


def build_graph_samples(
    features: pd.DataFrame,
    include_target: bool,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    """将逐目标窗口特征表重组为 [样本, 6步, 5节点, 特征] 的图样本。"""
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []

    group_cols = ["date", "session"]
    for (date_value, session), group in features.groupby(group_cols, sort=True):
        group = group.copy()
        sample_x = np.zeros((6, len(COMBOS), 8 + len(WEATHER_COLS) + len(ROUTE_COLS)))
        sample_y = np.zeros((6, len(COMBOS)))
        ok = True

        for node_idx, combo_id in enumerate(COMBO_IDS):
            combo_rows = group[group["combo_id"] == combo_id].sort_values("horizon_step")
            if len(combo_rows) != 6:
                ok = False
                break

            first_row = combo_rows.iloc[0]
            for step_idx, lag_col in enumerate(LAG_COLS):
                sample_x[step_idx, node_idx, :] = _row_to_input_features(
                    first_row, step_idx, float(first_row[lag_col])
                )

            if include_target:
                sample_y[:, node_idx] = combo_rows["target_volume"].to_numpy(dtype=float)

        if not ok:
            continue

        inputs.append(sample_x)
        if include_target:
            targets.append(sample_y)
        meta_rows.append(
            {
                "date": date_value,
                "session": session,
                "target_start": group["target_time"].min(),
            }
        )

    x_array = np.asarray(inputs, dtype=float)
    y_array = np.asarray(targets, dtype=float) if include_target else None
    meta = pd.DataFrame(meta_rows)
    return x_array, y_array, meta


def fit_feature_scaler(x_train: np.ndarray) -> FeatureScaler:
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    return FeatureScaler(mean=mean, std=np.where(std < 1e-6, 1.0, std))


def fit_target_scaler(y_train: np.ndarray) -> TargetScaler:
    y_log = np.log1p(y_train)
    std = float(y_log.std())
    return TargetScaler(mean=float(y_log.mean()), std=std if std >= 1e-6 else 1.0)


def init_weights(
    feature_dim: int,
    graph_hidden_dim: int,
    dense_hidden_dim: int,
    output_dim: int,
    seed: int = 42,
) -> GCNMLPWeights:
    rng = np.random.default_rng(seed)
    flat_dim = 6 * len(COMBOS) * graph_hidden_dim
    return GCNMLPWeights(
        graph_weight=rng.normal(0, np.sqrt(2 / feature_dim), (feature_dim, graph_hidden_dim)),
        graph_bias=np.zeros(graph_hidden_dim),
        dense_weight=rng.normal(0, np.sqrt(2 / flat_dim), (flat_dim, dense_hidden_dim)),
        dense_bias=np.zeros(dense_hidden_dim),
        output_weight=rng.normal(
            0, np.sqrt(2 / dense_hidden_dim), (dense_hidden_dim, output_dim)
        ),
        output_bias=np.zeros(output_dim),
    )


def forward(
    x: np.ndarray,
    adjacency: np.ndarray,
    weights: GCNMLPWeights,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    graph_input = np.einsum("ij,btjf->btif", adjacency, x)
    graph_pre = np.einsum("btif,fh->btih", graph_input, weights.graph_weight)
    graph_pre = graph_pre + weights.graph_bias
    graph_hidden = relu(graph_pre)
    flat_hidden = graph_hidden.reshape(x.shape[0], -1)
    dense_pre = flat_hidden @ weights.dense_weight + weights.dense_bias
    dense_hidden = relu(dense_pre)
    output = dense_hidden @ weights.output_weight + weights.output_bias
    cache = {
        "graph_input": graph_input,
        "graph_pre": graph_pre,
        "graph_hidden": graph_hidden,
        "flat_hidden": flat_hidden,
        "dense_pre": dense_pre,
        "dense_hidden": dense_hidden,
    }
    return output, cache


def backward(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: GCNMLPWeights,
    cache: dict[str, np.ndarray],
) -> list[np.ndarray]:
    batch_size, output_dim = prediction.shape
    grad_output = 2.0 * (prediction - target) / (batch_size * output_dim)

    grad_output_weight = cache["dense_hidden"].T @ grad_output
    grad_output_bias = grad_output.sum(axis=0)

    grad_dense = grad_output @ weights.output_weight.T
    grad_dense_pre = grad_dense * (cache["dense_pre"] > 0)
    grad_dense_weight = cache["flat_hidden"].T @ grad_dense_pre
    grad_dense_bias = grad_dense_pre.sum(axis=0)

    grad_flat = grad_dense_pre @ weights.dense_weight.T
    grad_graph_hidden = grad_flat.reshape(cache["graph_hidden"].shape)
    grad_graph_pre = grad_graph_hidden * (cache["graph_pre"] > 0)
    grad_graph_weight = (
        cache["graph_input"].reshape(-1, cache["graph_input"].shape[-1]).T
        @ grad_graph_pre.reshape(-1, grad_graph_pre.shape[-1])
    )
    grad_graph_bias = grad_graph_pre.sum(axis=(0, 1, 2))

    return [
        grad_graph_weight,
        grad_graph_bias,
        grad_dense_weight,
        grad_dense_bias,
        grad_output_weight,
        grad_output_bias,
    ]


def get_params(weights: GCNMLPWeights) -> list[np.ndarray]:
    return [
        weights.graph_weight,
        weights.graph_bias,
        weights.dense_weight,
        weights.dense_bias,
        weights.output_weight,
        weights.output_bias,
    ]


def adam_update(
    weights: GCNMLPWeights,
    grads: list[np.ndarray],
    state: AdamState,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    state.step += 1
    for param, grad, m_value, v_value in zip(get_params(weights), grads, state.m, state.v):
        m_value *= beta1
        m_value += (1 - beta1) * grad
        v_value *= beta2
        v_value += (1 - beta2) * (grad * grad)
        m_hat = m_value / (1 - beta1**state.step)
        v_hat = v_value / (1 - beta2**state.step)
        param -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)


def clip_gradients(grads: list[np.ndarray], max_norm: float = 5.0) -> list[np.ndarray]:
    total_norm = np.sqrt(sum(float(np.sum(grad * grad)) for grad in grads))
    if total_norm <= max_norm or total_norm < 1e-12:
        return grads
    scale = max_norm / total_norm
    return [grad * scale for grad in grads]


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    adjacency: np.ndarray,
    graph_hidden_dim: int = 4,
    dense_hidden_dim: int = 12,
    epochs: int = 1200,
    learning_rate: float = 0.001,
    weight_decay: float = 3e-3,
    seed: int = 42,
) -> GCNMLPWeights:
    weights = init_weights(
        feature_dim=x_train.shape[-1],
        graph_hidden_dim=graph_hidden_dim,
        dense_hidden_dim=dense_hidden_dim,
        output_dim=y_train.shape[1],
        seed=seed,
    )
    state = AdamState(
        m=[np.zeros_like(param) for param in get_params(weights)],
        v=[np.zeros_like(param) for param in get_params(weights)],
    )

    for _ in range(epochs):
        pred, cache = forward(x_train, adjacency, weights)
        grads = backward(pred, y_train, weights, cache)
        for grad, param in zip(grads, get_params(weights)):
            grad += weight_decay * param
        grads = clip_gradients(grads)
        adam_update(weights, grads, state, learning_rate)
    return weights


def predict(
    x: np.ndarray,
    adjacency: np.ndarray,
    weights: GCNMLPWeights,
    target_scaler: TargetScaler,
) -> np.ndarray:
    pred_scaled, _ = forward(x, adjacency, weights)
    pred = target_scaler.inverse_transform(pred_scaled.reshape(x.shape[0], 6, len(COMBOS)))
    return np.clip(pred, a_min=0.0, a_max=None)


def flatten_predictions(
    meta: pd.DataFrame,
    predictions: np.ndarray,
    features: pd.DataFrame,
    include_target: bool,
) -> pd.DataFrame:
    rows = []
    lookup_cols = ["date", "session", "horizon_step", "combo_id"]
    feature_lookup = features.set_index(lookup_cols)

    for sample_idx, meta_row in meta.reset_index(drop=True).iterrows():
        date_value = meta_row["date"]
        session = meta_row["session"]
        for horizon_idx in range(6):
            horizon_step = horizon_idx + 1
            for node_idx, combo_id in enumerate(COMBO_IDS):
                source = feature_lookup.loc[(date_value, session, horizon_step, combo_id)]
                row = {
                    "target_time": source["target_time"],
                    "time_window": source["time_window"],
                    "tollgate_id": int(source["tollgate_id"]),
                    "direction": int(source["direction"]),
                    "combo_id": combo_id,
                    "session": session,
                    "horizon_step": horizon_step,
                    "gnn_pred": float(predictions[sample_idx, horizon_idx, node_idx]),
                }
                if include_target:
                    row["target_volume"] = float(source["target_volume"])
                    row["gnn_abs_pct_error"] = (
                        abs(row["target_volume"] - row["gnn_pred"])
                        / max(row["target_volume"], 1.0)
                    )
                rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["target_time", "tollgate_id", "direction"]
    ).reset_index(drop=True)


def build_metrics(validation_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "scope": "overall",
            "group": "all",
            "rows": len(validation_predictions),
            "gnn_mape": calculate_mape(
                validation_predictions["target_volume"],
                validation_predictions["gnn_pred"],
            ),
            "gnn_rmse": calculate_rmse(
                validation_predictions["target_volume"],
                validation_predictions["gnn_pred"],
            ),
        }
    ]
    for scope, group_col in [("combo_id", "combo_id"), ("session", "session")]:
        for group_value, group_df in validation_predictions.groupby(group_col):
            rows.append(
                {
                    "scope": scope,
                    "group": group_value,
                    "rows": len(group_df),
                    "gnn_mape": calculate_mape(
                        group_df["target_volume"], group_df["gnn_pred"]
                    ),
                    "gnn_rmse": calculate_rmse(
                        group_df["target_volume"], group_df["gnn_pred"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_submission(
    prediction_rows: pd.DataFrame,
    submission_template: pd.DataFrame,
) -> pd.DataFrame:
    submission = prediction_rows[
        ["tollgate_id", "time_window", "direction", "gnn_pred"]
    ].rename(columns={"gnn_pred": "volume"})
    submission = submission_template[["tollgate_id", "time_window", "direction"]].merge(
        submission,
        on=["tollgate_id", "time_window", "direction"],
        how="left",
    )
    if submission["volume"].isna().any():
        raise ValueError("GNN 提交文件存在缺失预测。")
    submission["volume"] = submission["volume"].clip(lower=0.0)
    return submission[SUBMISSION_COLUMNS]


def print_summary(metrics: pd.DataFrame, adjacency: np.ndarray) -> None:
    overall = metrics.loc[metrics["scope"] == "overall"].iloc[0]
    print("GCN temporal graph model finished.")
    print("Adjacency matrix from historical volume correlation:")
    print(pd.DataFrame(adjacency, index=COMBO_IDS, columns=COMBO_IDS).round(4))
    print("\nValidation metrics")
    print(f"  GNN MAPE: {overall['gnn_mape']:.6f}")
    print(f"  GNN RMSE: {overall['gnn_rmse']:.6f}")

    print("\nMAPE / RMSE by combo")
    combo_metrics = metrics.loc[metrics["scope"] == "combo_id"].sort_values("group")
    for _, row in combo_metrics.iterrows():
        print(
            f"  {row['group']}: "
            f"mape={row['gnn_mape']:.6f}, "
            f"rmse={row['gnn_rmse']:.6f}, "
            f"rows={int(row['rows'])}"
        )


def main() -> None:
    train_features, predict_features, train_volume, submission_template = load_tables()
    validation_adjacency = build_normalized_adjacency(
        train_volume, cutoff_time=VALIDATION_START
    )
    final_adjacency = build_normalized_adjacency(train_volume)

    x_all, y_all, meta_all = build_graph_samples(train_features, include_target=True)
    train_mask = pd.to_datetime(meta_all["target_start"]) < VALIDATION_START
    valid_mask = (
        (pd.to_datetime(meta_all["target_start"]) >= VALIDATION_START)
        & (pd.to_datetime(meta_all["target_start"]) < VALIDATION_END)
    )

    x_train_raw = x_all[train_mask.to_numpy()]
    y_train_raw = y_all[train_mask.to_numpy()]
    x_valid_raw = x_all[valid_mask.to_numpy()]
    y_valid_raw = y_all[valid_mask.to_numpy()]
    meta_valid = meta_all.loc[valid_mask].reset_index(drop=True)

    feature_scaler = fit_feature_scaler(x_train_raw)
    target_scaler = fit_target_scaler(y_train_raw)
    x_train = feature_scaler.transform(x_train_raw)
    x_valid = feature_scaler.transform(x_valid_raw)
    y_train = target_scaler.transform(y_train_raw).reshape(len(y_train_raw), -1)

    weights = train_model(x_train, y_train, validation_adjacency, seed=1)
    valid_pred = predict(x_valid, validation_adjacency, weights, target_scaler)
    validation_predictions = flatten_predictions(
        meta_valid, valid_pred, train_features, include_target=True
    )
    metrics = build_metrics(validation_predictions)

    # 最终预测阶段重新使用全部训练期样本训练，保持与现有 XGBoost 提交流程同口径。
    final_feature_scaler = fit_feature_scaler(x_all)
    final_target_scaler = fit_target_scaler(y_all)
    x_final = final_feature_scaler.transform(x_all)
    y_final = final_target_scaler.transform(y_all).reshape(len(y_all), -1)
    final_weights = train_model(x_final, y_final, final_adjacency, seed=1)

    x_predict_raw, _, meta_predict = build_graph_samples(predict_features, include_target=False)
    x_predict = final_feature_scaler.transform(x_predict_raw)
    predict_pred = predict(x_predict, final_adjacency, final_weights, final_target_scaler)
    predict_rows = flatten_predictions(
        meta_predict, predict_pred, predict_features, include_target=False
    )
    submission = build_submission(predict_rows, submission_template)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    validation_predictions.to_csv(
        PROCESSED_DIR / "gnn_validation_predictions.csv", index=False
    )
    metrics.to_csv(PROCESSED_DIR / "gnn_model_metrics.csv", index=False)
    submission.to_csv(SUBMISSION_DIR / "submission_phase1_gnn.csv", index=False)

    print_summary(metrics, validation_adjacency)
    print("\nOutput files")
    print(f"  {PROCESSED_DIR / 'gnn_validation_predictions.csv'}")
    print(f"  {PROCESSED_DIR / 'gnn_model_metrics.csv'}")
    print(f"  {SUBMISSION_DIR / 'submission_phase1_gnn.csv'}")


if __name__ == "__main__":
    main()
