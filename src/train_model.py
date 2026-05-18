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

from utils import calculate_mape


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
    "time_window",
    "date",
    "session",
    "weather_time",
    "weather_dataset",
    "combo_id",
}
SUBMISSION_COLUMNS = ["tollgate_id", "time_window", "direction", "volume"]


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


def build_xgboost_model() -> XGBRegressor:
    """构建保守、可复现的 XGBoost 主模型。"""
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        eval_metric="mae",
    )


def clipped_predict(model, features: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """预测并裁剪为非负流量。"""
    predictions = model.predict(features[feature_cols])
    return np.clip(predictions, a_min=0, a_max=None)


def build_validation_predictions(
    valid_part: pd.DataFrame,
    ridge_pred: np.ndarray,
    xgb_pred: np.ndarray,
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
    result["xgboost_pred"] = xgb_pred
    result["ridge_abs_pct_error"] = np.where(
        result[TARGET_COL] != 0,
        np.abs(result[TARGET_COL] - result["ridge_pred"]) / result[TARGET_COL],
        np.nan,
    )
    result["xgboost_abs_pct_error"] = np.where(
        result[TARGET_COL] != 0,
        np.abs(result[TARGET_COL] - result["xgboost_pred"]) / result[TARGET_COL],
        np.nan,
    )
    return result.sort_values(["target_time", "tollgate_id", "direction"]).reset_index(
        drop=True
    )


def build_metrics(validation_predictions: pd.DataFrame) -> pd.DataFrame:
    """生成整体、收费站方向和早晚高峰维度的 MAPE 指标。"""
    rows = [
        {
            "scope": "overall",
            "group": "all",
            "ridge_mape": calculate_mape(
                validation_predictions[TARGET_COL],
                validation_predictions["ridge_pred"],
            ),
            "xgboost_mape": calculate_mape(
                validation_predictions[TARGET_COL],
                validation_predictions["xgboost_pred"],
            ),
            "rows": len(validation_predictions),
        }
    ]

    for scope, group_col in [("combo_id", "combo_id"), ("session", "session")]:
        for group_value, group_df in validation_predictions.groupby(group_col):
            rows.append(
                {
                    "scope": scope,
                    "group": group_value,
                    "ridge_mape": calculate_mape(
                        group_df[TARGET_COL], group_df["ridge_pred"]
                    ),
                    "xgboost_mape": calculate_mape(
                        group_df[TARGET_COL], group_df["xgboost_pred"]
                    ),
                    "rows": len(group_df),
                }
            )

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
    if validation_predictions[[TARGET_COL, "ridge_pred", "xgboost_pred"]].isna().any().any():
        raise ValueError("验证集预测结果存在缺失值。")
    if not {"ridge_mape", "xgboost_mape"}.issubset(metrics.columns):
        raise ValueError("模型指标缺少 ridge_mape 或 xgboost_mape。")


def print_metric_summary(metrics: pd.DataFrame) -> None:
    """打印整体和分组 MAPE 摘要。"""
    overall = metrics.loc[metrics["scope"] == "overall"].iloc[0]
    print("Validation MAPE")
    print(f"  Ridge:   {overall['ridge_mape']:.6f}")
    print(f"  XGBoost: {overall['xgboost_mape']:.6f}")

    for scope in ["combo_id", "session"]:
        print(f"\nMAPE by {scope}")
        scope_metrics = metrics.loc[metrics["scope"] == scope].sort_values("group")
        for _, row in scope_metrics.iterrows():
            print(
                f"  {row['group']}: "
                f"ridge={row['ridge_mape']:.6f}, "
                f"xgboost={row['xgboost_mape']:.6f}, rows={int(row['rows'])}"
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
    train_part, valid_part = split_train_validation(train_features)

    ridge_model = build_ridge_model()
    ridge_model.fit(train_part[feature_cols], train_part[TARGET_COL])
    ridge_pred = clipped_predict(ridge_model, valid_part, feature_cols)

    xgb_model = build_xgboost_model()
    xgb_model.fit(train_part[feature_cols], train_part[TARGET_COL])
    xgb_pred = clipped_predict(xgb_model, valid_part, feature_cols)

    validation_predictions = build_validation_predictions(valid_part, ridge_pred, xgb_pred)
    metrics = build_metrics(validation_predictions)

    final_model = build_xgboost_model()
    final_model.fit(train_features[feature_cols], train_features[TARGET_COL])
    submission_pred = clipped_predict(final_model, predict_features, feature_cols)
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
