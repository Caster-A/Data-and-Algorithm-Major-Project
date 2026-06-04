# 模型训练、调参与融合
# 负责人：成员四

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from utils import calculate_mape, calculate_rmse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "data" / "submission"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "train_features.csv"
PREDICT_FEATURES_PATH = PROCESSED_DIR / "predict_features_phase1.csv"
SUBMISSION_TEMPLATE_PATH = SUBMISSION_DIR / "submission_template_phase1.csv"

VALIDATION_START = pd.Timestamp("2016-10-11")
VALIDATION_END = pd.Timestamp("2016-10-18")
TARGET_COL = "target_volume"
EXCLUDE_FEATURE_COLS = {
    TARGET_COL,
    "target_time",
    "known_cutoff_time",
    "time_window",
    "date",
    "session",
    "weather_time",
    "weather_dataset",
    "combo_id",
}
ENHANCED_FEATURE_COLS = {
    "lead_2h_median",
    "lead_2h_q25",
    "lead_2h_q75",
    "lead_2h_range",
    "lead_2h_cv",
    "lead_prev_1h_sum",
    "lead_prev_1h_mean",
    "lead_last_vs_mean",
    "lead_recent_vs_prev_1h",
    "lead_recent_prev_ratio",
    "lead_diff_1",
    "lead_diff_2",
    "lead_diff_3",
    "lead_diff_4",
    "lead_diff_5",
    "lead_diff_mean",
    "lead_diff_std",
    "lead_acceleration",
    "lead_slope",
    "hist_combo_std",
    "previous_2day_same_window_volume",
    "previous_3day_same_window_volume",
    "previous_week_same_window_volume",
    "lead_mean_vs_hist_window",
    "lead_sum_vs_hist_window",
    "lead_mean_hist_window_ratio",
    "previous_day_vs_hist_window",
    "previous_day_hist_window_ratio",
    "combo_code",
    "session_horizon",
    "combo_horizon",
    "combo_session",
}
SUBMISSION_COLUMNS = ["tollgate_id", "time_window", "direction", "volume"]
PER_COMBO_PRED_COL = "per_combo_xgboost_pred"
UNIFIED_PRED_COL = "xgboost_pred"
HYBRID_PRED_COL = "hybrid_xgboost_pred"


def load_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取训练特征、预测特征和提交模板。"""
    train_features = pd.read_csv(TRAIN_FEATURES_PATH, parse_dates=["target_time"])
    predict_features = pd.read_csv(PREDICT_FEATURES_PATH, parse_dates=["target_time"])
    submission_template = pd.read_csv(SUBMISSION_TEMPLATE_PATH)
    return train_features, predict_features, submission_template


def select_feature_columns(train_features: pd.DataFrame) -> list[str]:
    """选择可直接进入模型的数值特征列。"""
    numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in EXCLUDE_FEATURE_COLS]
    if TARGET_COL in feature_cols:
        feature_cols.remove(TARGET_COL)
    return feature_cols


def select_baseline_feature_columns(train_features: pd.DataFrame) -> list[str]:
    """保留强化前已有的数值特征，作为同口径 baseline。"""
    feature_cols = select_feature_columns(train_features)
    return [col for col in feature_cols if col not in ENHANCED_FEATURE_COLS]


def split_train_validation(
    train_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间顺序切分训练集和验证集，避免随机切分造成泄露。"""
    train_mask = train_features["target_time"] < VALIDATION_START
    valid_mask = (
        (train_features["target_time"] >= VALIDATION_START)
        & (train_features["target_time"] < VALIDATION_END)
    )

    train_part = train_features.loc[train_mask].copy()
    valid_part = train_features.loc[valid_mask].copy()
    if train_part.empty or valid_part.empty:
        raise ValueError("训练集或验证集为空，请检查 target_time 和切分日期。")
    return train_part, valid_part


def build_ridge_model() -> Pipeline:
    """构建线性基线模型。"""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def build_xgboost_model(params: dict | None = None) -> XGBRegressor:
    """构建保守、可复现的 XGBoost 主模型。"""
    model_params = {
        "objective": "reg:squarederror",
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 1,
        "eval_metric": "mae",
    }
    if params:
        model_params.update(params)
    return XGBRegressor(**model_params)


def clipped_predict(model, features: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """预测并裁剪为非负流量。"""
    predictions = model.predict(features[feature_cols])
    return np.clip(predictions, a_min=0, a_max=None)


def fit_predict_per_combo(
    train_part: pd.DataFrame,
    predict_part: pd.DataFrame,
    feature_cols: list[str],
    params: dict | None = None,
) -> np.ndarray:
    """按收费站方向组合分别训练 XGBoost 并预测。"""
    predictions = pd.Series(index=predict_part.index, dtype=float)
    for combo_id, combo_train in train_part.groupby("combo_id"):
        combo_predict = predict_part[predict_part["combo_id"] == combo_id]
        if combo_predict.empty:
            continue
        model = build_xgboost_model(params)
        model.fit(combo_train[feature_cols], combo_train[TARGET_COL])
        predictions.loc[combo_predict.index] = clipped_predict(
            model, combo_predict, feature_cols
        )

    if predictions.isna().any():
        missing = predict_part.loc[predictions.isna(), "combo_id"].unique().tolist()
        raise ValueError(f"分组模型存在缺失预测，combo_id={missing}")
    return predictions.to_numpy()


def fit_predict_per_combo_session(
    train_part: pd.DataFrame,
    predict_part: pd.DataFrame,
    feature_cols: list[str],
    params: dict | None = None,
) -> np.ndarray:
    """按收费站方向组合和早晚高峰分别训练 XGBoost 并预测。"""
    predictions = pd.Series(index=predict_part.index, dtype=float)
    for (combo_id, session), group_train in train_part.groupby(["combo_id", "session"]):
        group_predict = predict_part[
            (predict_part["combo_id"] == combo_id)
            & (predict_part["session"] == session)
        ]
        if group_predict.empty:
            continue
        model = build_xgboost_model(params)
        model.fit(group_train[feature_cols], group_train[TARGET_COL])
        predictions.loc[group_predict.index] = clipped_predict(
            model, group_predict, feature_cols
        )

    if predictions.isna().any():
        missing = predict_part.loc[
            predictions.isna(), ["combo_id", "session"]
        ].drop_duplicates()
        raise ValueError(f"组合时段模型存在缺失预测，groups={missing.to_dict('records')}")
    return predictions.to_numpy()


def build_validation_predictions(
    valid_part: pd.DataFrame,
    ridge_pred: np.ndarray,
    unified_xgb_pred: np.ndarray,
    per_combo_xgb_pred: np.ndarray,
    hybrid_xgb_pred: np.ndarray,
    enhanced_unified_xgb_pred: np.ndarray,
    enhanced_per_combo_xgb_pred: np.ndarray,
    enhanced_hybrid_xgb_pred: np.ndarray,
) -> pd.DataFrame:
    """整理验证集预测结果，方便后续误差分析和报告引用。"""
    result_cols = [
        "target_time",
        "time_window",
        "tollgate_id",
        "direction",
        "combo_id",
        "session",
        "horizon_step",
        TARGET_COL,
    ]
    result = valid_part[result_cols].copy()
    result["ridge_pred"] = ridge_pred
    result["unified_xgboost_pred"] = unified_xgb_pred
    result["per_combo_xgboost_pred"] = per_combo_xgb_pred
    result["hybrid_xgboost_pred"] = hybrid_xgb_pred
    result["enhanced_unified_xgboost_pred"] = enhanced_unified_xgb_pred
    result["enhanced_per_combo_xgboost_pred"] = enhanced_per_combo_xgb_pred
    result["enhanced_hybrid_xgboost_pred"] = enhanced_hybrid_xgb_pred
    result["xgboost_pred"] = enhanced_hybrid_xgb_pred
    result["ridge_abs_pct_error"] = np.where(
        result[TARGET_COL] != 0,
        np.abs(result[TARGET_COL] - result["ridge_pred"]) / result[TARGET_COL],
        np.nan,
    )
    result["xgboost_abs_pct_error"] = np.where(
        result[TARGET_COL] != 0,
        np.abs(result[TARGET_COL] - result[UNIFIED_PRED_COL]) / result[TARGET_COL],
        np.nan,
    )
    result["per_combo_xgboost_abs_pct_error"] = np.where(
        result[TARGET_COL] != 0,
        np.abs(result[TARGET_COL] - result[PER_COMBO_PRED_COL]) / result[TARGET_COL],
        np.nan,
    )
    return result.sort_values(["target_time", "tollgate_id", "direction"]).reset_index(
        drop=True
    )


def choose_combo_model_map(validation_predictions: pd.DataFrame) -> dict[str, str]:
    """为每个组合选择统一 XGBoost 或 per-combo XGBoost 中验证 MAPE 更低者。"""
    combo_model_map = {}
    for combo_id, group_df in validation_predictions.groupby("combo_id"):
        unified_mape = calculate_mape(group_df[TARGET_COL], group_df[UNIFIED_PRED_COL])
        per_combo_mape = calculate_mape(group_df[TARGET_COL], group_df[PER_COMBO_PRED_COL])
        combo_model_map[combo_id] = (
            "per_combo_xgboost" if per_combo_mape < unified_mape else "xgboost"
        )
    return combo_model_map


def add_hybrid_predictions(
    validation_predictions: pd.DataFrame,
    combo_model_map: dict[str, str],
) -> pd.DataFrame:
    """按组合选择更优 XGBoost 版本，生成混合验证预测。"""
    result = validation_predictions.copy()
    result[HYBRID_PRED_COL] = np.where(
        result["combo_id"].map(combo_model_map) == "per_combo_xgboost",
        result[PER_COMBO_PRED_COL],
        result[UNIFIED_PRED_COL],
    )
    result["hybrid_xgboost_abs_pct_error"] = np.where(
        result[TARGET_COL] != 0,
        np.abs(result[TARGET_COL] - result[HYBRID_PRED_COL]) / result[TARGET_COL],
        np.nan,
    )
    result["combo_selected_model"] = result["combo_id"].map(combo_model_map)
    return result


def build_hybrid_submission_predictions(
    predict_features: pd.DataFrame,
    unified_predictions: np.ndarray,
    per_combo_predictions: np.ndarray,
    combo_model_map: dict[str, str],
) -> np.ndarray:
    """根据验证集组合级选择结果生成混合提交预测。"""
    use_per_combo = (
        predict_features["combo_id"].map(combo_model_map).fillna("xgboost")
        == "per_combo_xgboost"
    )
    return np.where(use_per_combo, per_combo_predictions, unified_predictions)


def build_metrics(validation_predictions: pd.DataFrame) -> pd.DataFrame:
    """生成整体、收费站方向和早晚高峰维度的 MAPE 与 RMSE 指标。"""
    model_pred_cols = {
        "ridge": "ridge_pred",
        "unified_xgboost": "unified_xgboost_pred",
        "per_combo_xgboost": "per_combo_xgboost_pred",
        "hybrid_xgboost": "hybrid_xgboost_pred",
        "enhanced_unified_xgboost": "enhanced_unified_xgboost_pred",
        "enhanced_per_combo_xgboost": "enhanced_per_combo_xgboost_pred",
        "enhanced_hybrid_xgboost": "enhanced_hybrid_xgboost_pred",
        "optimized_hybrid_xgboost": "optimized_hybrid_xgboost_pred",
        "xgboost": "xgboost_pred",
    }
    rows = [
        {
            "scope": "overall",
            "group": "all",
            "rows": len(validation_predictions),
        }
    ]
    for model_name, pred_col in model_pred_cols.items():
        rows[0][f"{model_name}_mape"] = calculate_mape(
            validation_predictions[TARGET_COL],
            validation_predictions[pred_col],
        )
        rows[0][f"{model_name}_rmse"] = calculate_rmse(
            validation_predictions[TARGET_COL],
            validation_predictions[pred_col],
        )

    for scope, group_col in [("combo_id", "combo_id"), ("session", "session")]:
        for group_value, group_df in validation_predictions.groupby(group_col):
            row = {"scope": scope, "group": group_value, "rows": len(group_df)}
            for model_name, pred_col in model_pred_cols.items():
                row[f"{model_name}_mape"] = calculate_mape(
                    group_df[TARGET_COL], group_df[pred_col]
                )
                row[f"{model_name}_rmse"] = calculate_rmse(
                    group_df[TARGET_COL], group_df[pred_col]
                )
            rows.append(row)

    return pd.DataFrame(rows)


def build_submission(
    predict_features: pd.DataFrame,
    predictions: np.ndarray,
    submission_template: pd.DataFrame,
) -> pd.DataFrame:
    """按提交模板生成 phase1 预测文件。"""
    submission = predict_features[["tollgate_id", "time_window", "direction"]].copy()
    submission["volume"] = np.clip(predictions, a_min=0, a_max=None)

    template_keys = set(
        zip(
            submission_template["tollgate_id"],
            submission_template["time_window"],
            submission_template["direction"],
        )
    )
    submission_keys = set(
        zip(submission["tollgate_id"], submission["time_window"], submission["direction"])
    )
    if submission_keys != template_keys:
        missing = template_keys - submission_keys
        extra = submission_keys - template_keys
        raise ValueError(f"提交预测窗口与模板不一致，missing={len(missing)}, extra={len(extra)}")

    submission = submission_template[["tollgate_id", "time_window", "direction"]].merge(
        submission,
        on=["tollgate_id", "time_window", "direction"],
        how="left",
    )
    submission = submission[SUBMISSION_COLUMNS]
    return submission.sort_values(["time_window", "tollgate_id", "direction"]).reset_index(
        drop=True
    )


def choose_combo_model_map(
    validation_predictions: pd.DataFrame,
    candidate_pred_cols: dict[str, str],
) -> dict[str, str]:
    """按 combo_id 选择验证集 MAPE 最低的候选模型。"""
    choices = {}
    for combo_id, group_df in validation_predictions.groupby("combo_id"):
        combo_scores = {
            model_name: calculate_mape(group_df[TARGET_COL], group_df[pred_col])
            for model_name, pred_col in candidate_pred_cols.items()
        }
        choices[combo_id] = min(combo_scores, key=combo_scores.get)
    return choices


def apply_combo_model_map(
    rows: pd.DataFrame,
    candidate_predictions: dict[str, np.ndarray],
    choices: dict[str, str],
) -> np.ndarray:
    """根据 combo_id 的模型选择拼接最终预测。"""
    first_candidate = next(iter(candidate_predictions.values()))
    selected = np.asarray(first_candidate, dtype=float).copy()
    for model_name, predictions in candidate_predictions.items():
        model_mask = rows["combo_id"].map(choices).eq(model_name).to_numpy()
        selected[model_mask] = predictions[model_mask]
    return selected


def validate_outputs(
    submission: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    submission_template: pd.DataFrame,
) -> None:
    """集中检查输出文件是否满足后续提交和报告需要。"""
    if list(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"提交字段错误: {list(submission.columns)}")
    if len(submission) != len(submission_template):
        raise ValueError("提交行数与模板不一致。")
    if submission["volume"].isna().any():
        raise ValueError("提交文件存在缺失预测值。")
    if (submission["volume"] < 0).any():
        raise ValueError("提交文件存在负流量预测。")
    prediction_cols = [
        TARGET_COL,
        "ridge_pred",
        "unified_xgboost_pred",
        "per_combo_xgboost_pred",
        "hybrid_xgboost_pred",
        "enhanced_unified_xgboost_pred",
        "enhanced_per_combo_xgboost_pred",
        "enhanced_hybrid_xgboost_pred",
        "optimized_hybrid_xgboost_pred",
        "xgboost_pred",
    ]
    if validation_predictions[prediction_cols].isna().any().any():
        raise ValueError("验证集预测结果存在缺失值。")
    required_metric_cols = {
        "ridge_mape",
        "unified_xgboost_mape",
        "per_combo_xgboost_mape",
        "hybrid_xgboost_mape",
        "enhanced_unified_xgboost_mape",
        "enhanced_per_combo_xgboost_mape",
        "enhanced_hybrid_xgboost_mape",
        "optimized_hybrid_xgboost_mape",
        "xgboost_mape",
        "ridge_rmse",
        "unified_xgboost_rmse",
        "per_combo_xgboost_rmse",
        "hybrid_xgboost_rmse",
        "enhanced_unified_xgboost_rmse",
        "enhanced_per_combo_xgboost_rmse",
        "enhanced_hybrid_xgboost_rmse",
        "optimized_hybrid_xgboost_rmse",
        "xgboost_rmse",
    }
    if not required_metric_cols.issubset(metrics.columns):
        raise ValueError("模型指标缺少必要 MAPE 或 RMSE 字段。")


def print_metric_summary(metrics: pd.DataFrame) -> None:
    """打印整体和分组 MAPE / RMSE 摘要。"""
    overall = metrics.loc[metrics["scope"] == "overall"].iloc[0]
    print("Validation MAPE")
    print(f"  Ridge:   {overall['ridge_mape']:.6f}")
    print(f"  Unified XGBoost: {overall['unified_xgboost_mape']:.6f}")
    print(f"  Per-combo XGBoost: {overall['per_combo_xgboost_mape']:.6f}")
    print(f"  Hybrid XGBoost: {overall['hybrid_xgboost_mape']:.6f}")
    print(f"  Enhanced Unified XGBoost: {overall['enhanced_unified_xgboost_mape']:.6f}")
    print(f"  Enhanced Per-combo XGBoost: {overall['enhanced_per_combo_xgboost_mape']:.6f}")
    print(f"  Enhanced Hybrid XGBoost: {overall['enhanced_hybrid_xgboost_mape']:.6f}")
    print(f"  Optimized Hybrid XGBoost: {overall['optimized_hybrid_xgboost_mape']:.6f}")
    print(
        "  Improvement: "
        f"{overall['hybrid_xgboost_mape'] - overall['optimized_hybrid_xgboost_mape']:+.6f}"
    )
    print("\nValidation RMSE")
    print(f"  Ridge:   {overall['ridge_rmse']:.6f}")
    print(f"  Hybrid XGBoost: {overall['hybrid_xgboost_rmse']:.6f}")
    print(f"  Enhanced Hybrid XGBoost: {overall['enhanced_hybrid_xgboost_rmse']:.6f}")
    print(f"  Optimized Hybrid XGBoost: {overall['optimized_hybrid_xgboost_rmse']:.6f}")
    print(
        "  Improvement: "
        f"{overall['hybrid_xgboost_rmse'] - overall['optimized_hybrid_xgboost_rmse']:+.6f}"
    )

    for scope in ["combo_id", "session"]:
        print(f"\nMAPE / RMSE by {scope}")
        scope_metrics = metrics.loc[metrics["scope"] == scope].sort_values("group")
        for _, row in scope_metrics.iterrows():
            print(
                f"  {row['group']}: "
                f"ridge={row['ridge_mape']:.6f}, "
                f"hybrid={row['hybrid_xgboost_mape']:.6f}, "
                f"enhanced_hybrid={row['enhanced_hybrid_xgboost_mape']:.6f}; "
                f"optimized_hybrid={row['optimized_hybrid_xgboost_mape']:.6f}; "
                f"hybrid_rmse={row['hybrid_xgboost_rmse']:.6f}, "
                f"optimized_hybrid_rmse={row['optimized_hybrid_xgboost_rmse']:.6f}, "
                f"rows={int(row['rows'])}"
            )


def save_predictions_and_metrics(
    validation_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    submission: pd.DataFrame,
) -> None:
    """保存验证预测、指标和提交文件。"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    validation_predictions.to_csv(
        PROCESSED_DIR / "validation_predictions.csv", index=False
    )
    metrics.to_csv(PROCESSED_DIR / "model_metrics.csv", index=False)
    submission.to_csv(SUBMISSION_DIR / "submission_phase1.csv", index=False)


def main() -> None:
    train_features, predict_features, submission_template = load_feature_tables()
    feature_cols = select_feature_columns(train_features)
    baseline_feature_cols = select_baseline_feature_columns(train_features)
    train_part, valid_part = split_train_validation(train_features)

    ridge_model = build_ridge_model()
    ridge_model.fit(train_part[baseline_feature_cols], train_part[TARGET_COL])
    ridge_pred = clipped_predict(ridge_model, valid_part, baseline_feature_cols)

    baseline_xgb_model = build_xgboost_model()
    baseline_xgb_model.fit(train_part[baseline_feature_cols], train_part[TARGET_COL])
    unified_xgb_pred = clipped_predict(
        baseline_xgb_model, valid_part, baseline_feature_cols
    )
    per_combo_xgb_pred = fit_predict_per_combo(
        train_part, valid_part, baseline_feature_cols
    )

    hybrid_seed = build_validation_predictions(
        valid_part,
        ridge_pred,
        unified_xgb_pred,
        per_combo_xgb_pred,
        unified_xgb_pred,
        unified_xgb_pred,
        per_combo_xgb_pred,
        unified_xgb_pred,
    )
    hybrid_model_map = choose_combo_model_map(
        hybrid_seed,
        {
            "unified_xgboost": "unified_xgboost_pred",
            "per_combo_xgboost": "per_combo_xgboost_pred",
        },
    )
    hybrid_xgb_pred = apply_combo_model_map(
        valid_part,
        {
            "unified_xgboost": unified_xgb_pred,
            "per_combo_xgboost": per_combo_xgb_pred,
        },
        hybrid_model_map,
    )

    xgb_model = build_xgboost_model()
    xgb_model.fit(train_part[feature_cols], train_part[TARGET_COL])
    enhanced_xgb_pred = clipped_predict(xgb_model, valid_part, feature_cols)
    enhanced_per_combo_xgb_pred = fit_predict_per_combo(train_part, valid_part, feature_cols)

    validation_predictions = build_validation_predictions(
        valid_part,
        ridge_pred,
        unified_xgb_pred,
        per_combo_xgb_pred,
        hybrid_xgb_pred,
        enhanced_xgb_pred,
        enhanced_per_combo_xgb_pred,
        hybrid_xgb_pred,
    )
    enhanced_hybrid_model_map = choose_combo_model_map(
        validation_predictions,
        {
            "unified_xgboost": "unified_xgboost_pred",
            "per_combo_xgboost": "per_combo_xgboost_pred",
            "enhanced_unified_xgboost": "enhanced_unified_xgboost_pred",
            "enhanced_per_combo_xgboost": "enhanced_per_combo_xgboost_pred",
        },
    )
    enhanced_hybrid_xgb_pred = apply_combo_model_map(
        valid_part,
        {
            "unified_xgboost": unified_xgb_pred,
            "per_combo_xgboost": per_combo_xgb_pred,
            "enhanced_unified_xgboost": enhanced_xgb_pred,
            "enhanced_per_combo_xgboost": enhanced_per_combo_xgb_pred,
        },
        enhanced_hybrid_model_map,
    )
    validation_predictions["enhanced_hybrid_xgboost_pred"] = enhanced_hybrid_xgb_pred

    depth2_params = {
        "max_depth": 2,
        "min_child_weight": 2,
        "n_estimators": 600,
        "learning_rate": 0.03,
    }
    more_trees_params = {"n_estimators": 800, "learning_rate": 0.025}
    regularized_params = {
        "max_depth": 2,
        "min_child_weight": 4,
        "reg_lambda": 4,
        "n_estimators": 600,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    unified_base_depth2_model = build_xgboost_model(depth2_params)
    unified_base_depth2_model.fit(
        train_part[baseline_feature_cols], train_part[TARGET_COL]
    )
    unified_base_depth2_pred = clipped_predict(
        unified_base_depth2_model, valid_part, baseline_feature_cols
    )
    combo_session_base_pred = fit_predict_per_combo_session(
        train_part, valid_part, baseline_feature_cols
    )
    enhanced_per_combo_depth2_pred = fit_predict_per_combo(
        train_part, valid_part, feature_cols, depth2_params
    )
    enhanced_unified_more_trees_model = build_xgboost_model(more_trees_params)
    enhanced_unified_more_trees_model.fit(train_part[feature_cols], train_part[TARGET_COL])
    enhanced_unified_more_trees_pred = clipped_predict(
        enhanced_unified_more_trees_model, valid_part, feature_cols
    )
    enhanced_unified_regularized_model = build_xgboost_model(regularized_params)
    enhanced_unified_regularized_model.fit(train_part[feature_cols], train_part[TARGET_COL])
    enhanced_unified_regularized_pred = clipped_predict(
        enhanced_unified_regularized_model, valid_part, feature_cols
    )

    optimized_candidate_predictions = {
        "unified_xgboost": unified_xgb_pred,
        "per_combo_xgboost": per_combo_xgb_pred,
        "hybrid_xgboost": hybrid_xgb_pred,
        "enhanced_unified_xgboost": enhanced_xgb_pred,
        "enhanced_per_combo_xgboost": enhanced_per_combo_xgb_pred,
        "enhanced_hybrid_xgboost": enhanced_hybrid_xgb_pred,
        "unified_base_depth2_xgboost": unified_base_depth2_pred,
        "combo_session_base_xgboost": combo_session_base_pred,
        "enhanced_per_combo_depth2_xgboost": enhanced_per_combo_depth2_pred,
        "enhanced_unified_more_trees_xgboost": enhanced_unified_more_trees_pred,
        "enhanced_unified_regularized_xgboost": enhanced_unified_regularized_pred,
    }
    for model_name, predictions in optimized_candidate_predictions.items():
        validation_predictions[f"{model_name}_pred"] = predictions

    optimized_hybrid_model_map = choose_combo_model_map(
        validation_predictions,
        {
            model_name: f"{model_name}_pred"
            for model_name in optimized_candidate_predictions
        },
    )
    optimized_hybrid_xgb_pred = apply_combo_model_map(
        valid_part, optimized_candidate_predictions, optimized_hybrid_model_map
    )
    validation_predictions["optimized_hybrid_xgboost_pred"] = optimized_hybrid_xgb_pred
    validation_predictions["xgboost_pred"] = optimized_hybrid_xgb_pred
    validation_predictions["xgboost_abs_pct_error"] = np.where(
        validation_predictions[TARGET_COL] != 0,
        np.abs(validation_predictions[TARGET_COL] - validation_predictions["xgboost_pred"])
        / validation_predictions[TARGET_COL],
        np.nan,
    )
    metrics = build_metrics(validation_predictions)
    metrics["hybrid_model_map"] = metrics.apply(
        lambda row: hybrid_model_map.get(row["group"], np.nan)
        if row["scope"] == "combo_id"
        else np.nan,
        axis=1,
    )
    metrics["optimized_model_map"] = metrics.apply(
        lambda row: optimized_hybrid_model_map.get(row["group"], np.nan)
        if row["scope"] == "combo_id"
        else np.nan,
        axis=1,
    )

    final_baseline_model = build_xgboost_model()
    final_baseline_model.fit(train_features[baseline_feature_cols], train_features[TARGET_COL])
    unified_submission_pred = clipped_predict(
        final_baseline_model, predict_features, baseline_feature_cols
    )
    per_combo_submission_pred = fit_predict_per_combo(
        train_features, predict_features, baseline_feature_cols
    )

    final_model = build_xgboost_model()
    final_model.fit(train_features[feature_cols], train_features[TARGET_COL])
    enhanced_submission_pred = clipped_predict(final_model, predict_features, feature_cols)
    enhanced_per_combo_submission_pred = fit_predict_per_combo(
        train_features, predict_features, feature_cols
    )

    final_unified_base_depth2_model = build_xgboost_model(depth2_params)
    final_unified_base_depth2_model.fit(
        train_features[baseline_feature_cols], train_features[TARGET_COL]
    )
    unified_base_depth2_submission_pred = clipped_predict(
        final_unified_base_depth2_model, predict_features, baseline_feature_cols
    )
    combo_session_base_submission_pred = fit_predict_per_combo_session(
        train_features, predict_features, baseline_feature_cols
    )
    enhanced_per_combo_depth2_submission_pred = fit_predict_per_combo(
        train_features, predict_features, feature_cols, depth2_params
    )
    final_enhanced_unified_more_trees_model = build_xgboost_model(more_trees_params)
    final_enhanced_unified_more_trees_model.fit(
        train_features[feature_cols], train_features[TARGET_COL]
    )
    enhanced_unified_more_trees_submission_pred = clipped_predict(
        final_enhanced_unified_more_trees_model, predict_features, feature_cols
    )
    final_enhanced_unified_regularized_model = build_xgboost_model(regularized_params)
    final_enhanced_unified_regularized_model.fit(
        train_features[feature_cols], train_features[TARGET_COL]
    )
    enhanced_unified_regularized_submission_pred = clipped_predict(
        final_enhanced_unified_regularized_model, predict_features, feature_cols
    )

    submission_pred = apply_combo_model_map(
        predict_features,
        {
            "unified_xgboost": unified_submission_pred,
            "per_combo_xgboost": per_combo_submission_pred,
            "enhanced_unified_xgboost": enhanced_submission_pred,
            "enhanced_per_combo_xgboost": enhanced_per_combo_submission_pred,
            "hybrid_xgboost": apply_combo_model_map(
                predict_features,
                {
                    "unified_xgboost": unified_submission_pred,
                    "per_combo_xgboost": per_combo_submission_pred,
                },
                hybrid_model_map,
            ),
            "enhanced_hybrid_xgboost": apply_combo_model_map(
                predict_features,
                {
                    "unified_xgboost": unified_submission_pred,
                    "per_combo_xgboost": per_combo_submission_pred,
                    "enhanced_unified_xgboost": enhanced_submission_pred,
                    "enhanced_per_combo_xgboost": enhanced_per_combo_submission_pred,
                },
                enhanced_hybrid_model_map,
            ),
            "unified_base_depth2_xgboost": unified_base_depth2_submission_pred,
            "combo_session_base_xgboost": combo_session_base_submission_pred,
            "enhanced_per_combo_depth2_xgboost": enhanced_per_combo_depth2_submission_pred,
            "enhanced_unified_more_trees_xgboost": enhanced_unified_more_trees_submission_pred,
            "enhanced_unified_regularized_xgboost": enhanced_unified_regularized_submission_pred,
        },
        optimized_hybrid_model_map,
    )
    submission = build_submission(predict_features, submission_pred, submission_template)

    validate_outputs(submission, validation_predictions, metrics, submission_template)
    save_predictions_and_metrics(validation_predictions, metrics, submission)
    print_metric_summary(metrics)
    print("\nOutput files")
    print(f"  {PROCESSED_DIR / 'validation_predictions.csv'}")
    print(f"  {PROCESSED_DIR / 'model_metrics.csv'}")
    print(f"  {SUBMISSION_DIR / 'submission_phase1.csv'}")


if __name__ == "__main__":
    main()
