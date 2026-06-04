# 验证评估、结果分析与报告整合
# 负责人：成员五

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import calculate_mape, calculate_rmse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "data" / "submission"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORT_DIR = PROJECT_ROOT / "report"

VALIDATION_PREDICTIONS_PATH = PROCESSED_DIR / "validation_predictions.csv"
MODEL_METRICS_PATH = PROCESSED_DIR / "model_metrics.csv"
SUBMISSION_PATH = SUBMISSION_DIR / "submission_phase1.csv"
SUBMISSION_TEMPLATE_PATH = SUBMISSION_DIR / "submission_template_phase1.csv"

SUBMISSION_COLUMNS = ["tollgate_id", "time_window", "direction", "volume"]
MODEL_PREDICTION_COLUMNS = {
    "ridge": "ridge_pred",
    "xgboost": "xgboost_pred",
    "per_combo_xgboost": "per_combo_xgboost_pred",
    "hybrid_xgboost": "hybrid_xgboost_pred",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取训练验证输出、模型指标和提交文件。"""
    validation = pd.read_csv(VALIDATION_PREDICTIONS_PATH, parse_dates=["target_time"])
    metrics = pd.read_csv(MODEL_METRICS_PATH)
    submission = pd.read_csv(SUBMISSION_PATH)
    template = pd.read_csv(SUBMISSION_TEMPLATE_PATH)
    return validation, metrics, submission, template


def validate_submission(submission: pd.DataFrame, template: pd.DataFrame) -> list[str]:
    """检查提交文件是否符合题目格式和模板覆盖要求。"""
    checks = []
    if list(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"提交字段错误: {list(submission.columns)}")
    checks.append("提交字段正确")

    if len(submission) != len(template):
        raise ValueError(f"提交行数错误: {len(submission)} != {len(template)}")
    checks.append(f"提交行数正确: {len(submission)}")

    missing_volume = int(submission["volume"].isna().sum())
    if missing_volume:
        raise ValueError(f"提交文件存在缺失预测值: {missing_volume}")
    checks.append("提交预测值无缺失")

    negative_volume = int((submission["volume"] < 0).sum())
    if negative_volume:
        raise ValueError(f"提交文件存在负流量预测: {negative_volume}")
    checks.append("提交预测值均为非负")

    template_keys = set(zip(template["tollgate_id"], template["time_window"], template["direction"]))
    submission_keys = set(
        zip(submission["tollgate_id"], submission["time_window"], submission["direction"])
    )
    if template_keys != submission_keys:
        raise ValueError(
            f"提交窗口与模板不一致: missing={len(template_keys - submission_keys)}, "
            f"extra={len(submission_keys - template_keys)}"
        )
    checks.append("提交窗口与模板完全一致")
    return checks


def build_group_metrics(validation: pd.DataFrame) -> pd.DataFrame:
    """按多个维度统计 Ridge 和 XGBoost 的 MAPE 与 RMSE。"""
    rows = []
    available_models = {
        model_name: pred_col
        for model_name, pred_col in MODEL_PREDICTION_COLUMNS.items()
        if pred_col in validation.columns
    }
    group_specs = [
        ("overall", None),
        ("combo_id", "combo_id"),
        ("session", "session"),
        ("horizon_step", "horizon_step"),
        ("weekday", "target_time_weekday"),
        ("is_weekend", "is_weekend"),
    ]
    validation = validation.copy()
    validation["target_time_weekday"] = validation["target_time"].dt.weekday
    validation["is_weekend"] = validation["target_time_weekday"].isin([5, 6]).astype(int)

    for scope, group_col in group_specs:
        if group_col is None:
            grouped = [("all", validation)]
        else:
            grouped = validation.groupby(group_col)

        for group, group_df in grouped:
            rows.append(
                {
                    "scope": scope,
                    "group": str(group),
                    "ridge_mape": calculate_mape(
                        group_df["target_volume"], group_df["ridge_pred"]
                    ),
                    "xgboost_mape": calculate_mape(
                        group_df["target_volume"], group_df["xgboost_pred"]
                    ),
                    "ridge_rmse": calculate_rmse(
                        group_df["target_volume"], group_df["ridge_pred"]
                    ),
                    "xgboost_rmse": calculate_rmse(
                        group_df["target_volume"], group_df["xgboost_pred"]
                    ),
                    "rows": len(group_df),
                }
            )

    result = pd.DataFrame(rows)
    return result.sort_values(["scope", "group"]).reset_index(drop=True)


def save_group_metrics(group_metrics: pd.DataFrame) -> None:
    group_metrics.to_csv(PROCESSED_DIR / "evaluation_group_metrics.csv", index=False)


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", font="Arial")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180


def plot_actual_vs_pred(validation: pd.DataFrame) -> None:
    """绘制验证集真实值与当前最佳模型预测值散点图。"""
    best_model, best_pred_col = choose_best_prediction_column(validation)
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=validation,
        x="target_volume",
        y=best_pred_col,
        hue="session",
        alpha=0.7,
        s=28,
    )
    max_value = max(validation["target_volume"].max(), validation[best_pred_col].max())
    plt.plot([0, max_value], [0, max_value], color="black", linewidth=1, linestyle="--")
    plt.title(f"Validation Actual vs {best_model} Prediction")
    plt.xlabel("Actual volume")
    plt.ylabel("Predicted volume")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "validation_actual_vs_pred.png")
    plt.close()


def plot_mape_by_scope(group_metrics: pd.DataFrame, scope: str, filename: str) -> None:
    """绘制某个分组维度下当前最佳模型的 MAPE 柱状图。"""
    best_model = choose_best_model_from_group_metrics(group_metrics)
    metric_col = f"{best_model}_mape"
    plot_df = group_metrics[group_metrics["scope"] == scope].copy()
    plot_df["best_mape_pct"] = plot_df[metric_col] * 100
    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=plot_df, x="group", y="best_mape_pct", color="#4C78A8")
    plt.title(f"{best_model} MAPE by {scope}")
    plt.xlabel(scope)
    plt.ylabel("MAPE (%)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()


def choose_best_model_from_group_metrics(group_metrics: pd.DataFrame) -> str:
    """根据整体验证 MAPE 选择当前最佳模型名。"""
    overall = group_metrics[
        (group_metrics["scope"] == "overall") & (group_metrics["group"] == "all")
    ].iloc[0]
    candidate_cols = [
        col for col in group_metrics.columns if col.endswith("_mape") and col != "ridge_mape"
    ]
    if not candidate_cols:
        return "ridge"
    best_col = min(candidate_cols, key=lambda col: overall[col])
    return best_col.removesuffix("_mape")


def choose_best_prediction_column(validation: pd.DataFrame) -> tuple[str, str]:
    """根据验证集 MAPE 选择当前最佳非 Ridge 预测列。"""
    candidate_models = {
        name: col
        for name, col in MODEL_PREDICTION_COLUMNS.items()
        if name != "ridge" and col in validation.columns
    }
    if not candidate_models:
        return "ridge", "ridge_pred"
    best_model = min(
        candidate_models,
        key=lambda name: calculate_mape(
            validation["target_volume"], validation[candidate_models[name]]
        ),
    )
    return best_model, candidate_models[best_model]


def save_figures(validation: pd.DataFrame, group_metrics: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    plot_actual_vs_pred(validation)
    plot_mape_by_scope(group_metrics, "combo_id", "mape_by_combo.png")
    plot_mape_by_scope(group_metrics, "session", "mape_by_session.png")
    plot_mape_by_scope(group_metrics, "horizon_step", "mape_by_horizon.png")


def build_report(
    group_metrics: pd.DataFrame,
    submission_checks: list[str],
    metrics: pd.DataFrame,
) -> str:
    """生成评估摘要 Markdown 文本。"""
    overall = group_metrics[
        (group_metrics["scope"] == "overall") & (group_metrics["group"] == "all")
    ].iloc[0]
    best_model = choose_best_model_from_group_metrics(group_metrics)
    best_metric_col = f"{best_model}_mape"
    worst_combo = (
        group_metrics[group_metrics["scope"] == "combo_id"]
        .sort_values(best_metric_col, ascending=False)
        .iloc[0]
    )
    worst_horizon = (
        group_metrics[group_metrics["scope"] == "horizon_step"]
        .sort_values(best_metric_col, ascending=False)
        .iloc[0]
    )

    lines = [
        "# 模型评估摘要",
        "",
        "## 1. 整体验证结果",
        "",
        f"- Ridge MAPE：`{overall['ridge_mape']:.6f}`",
        f"- XGBoost MAPE：`{overall['xgboost_mape']:.6f}`",
        f"- Ridge RMSE：`{overall['ridge_rmse']:.6f}`",
        f"- XGBoost RMSE：`{overall['xgboost_rmse']:.6f}`",
        f"- 验证样本数：`{int(overall['rows'])}`",
        "",
    ]
    if {
        "hybrid_xgboost_mape",
        "enhanced_hybrid_xgboost_mape",
        "optimized_hybrid_xgboost_mape",
        "hybrid_xgboost_rmse",
        "enhanced_hybrid_xgboost_rmse",
        "optimized_hybrid_xgboost_rmse",
    }.issubset(metrics.columns):
        model_overall = metrics[metrics["scope"] == "overall"].iloc[0]
        mape_improvement = (
            model_overall["hybrid_xgboost_mape"]
            - model_overall["optimized_hybrid_xgboost_mape"]
        )
        rmse_improvement = (
            model_overall["hybrid_xgboost_rmse"]
            - model_overall["optimized_hybrid_xgboost_rmse"]
        )
        lines.extend(
            [
                "## 1.1 Hybrid 优化对比",
                "",
                f"- 原 Hybrid XGBoost MAPE：`{model_overall['hybrid_xgboost_mape']:.6f}`",
                f"- 纯强化特征 Hybrid MAPE：`{model_overall['enhanced_hybrid_xgboost_mape']:.6f}`",
                f"- 优化版 Hybrid MAPE：`{model_overall['optimized_hybrid_xgboost_mape']:.6f}`",
                f"- 原 Hybrid XGBoost RMSE：`{model_overall['hybrid_xgboost_rmse']:.6f}`",
                f"- 纯强化特征 Hybrid RMSE：`{model_overall['enhanced_hybrid_xgboost_rmse']:.6f}`",
                f"- 优化版 Hybrid RMSE：`{model_overall['optimized_hybrid_xgboost_rmse']:.6f}`",
                f"- 相比原 Hybrid MAPE 改善：`{mape_improvement:.6f}`",
                f"- 相比原 Hybrid RMSE 改善：`{rmse_improvement:.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 2. 提交文件检查",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in submission_checks])
    lines.extend(
        [
            "",
            "## 3. 主要误差来源",
            "",
            f"- 当前误差最高的收费站方向组合是 `{worst_combo['group']}`，XGBoost MAPE 为 `{worst_combo['xgboost_mape']:.6f}`。",
            f"- 当前误差最高的预测步长是 `horizon_step={worst_horizon['group']}`，XGBoost MAPE 为 `{worst_horizon['xgboost_mape']:.6f}`。",
            f"- 对应 RMSE 分别为 `{worst_combo['xgboost_rmse']:.6f}` 和 `{worst_horizon['xgboost_rmse']:.6f}`，用于观察绝对流量误差大小。",
            "",
            "## 4. 生成图表",
            "",
            "- `figures/validation_actual_vs_pred.png`",
            "- `figures/mape_by_combo.png`",
            "- `figures/mape_by_session.png`",
            "- `figures/mape_by_horizon.png`",
            "",
            "## 5. 后续建议",
            "",
            "- 优先分析高误差组合和高误差预测步长。",
            "- 对比统一 XGBoost 和 per-combo XGBoost 的分组表现。",
            "- 如果 per-combo 只改善部分组合，可尝试混合模型策略。",
            "- 在验证集上比较后，再考虑 ExtraTrees 或 RandomForest 融合。",
        ]
    )

    if not metrics.empty:
        lines.extend(
            [
                "",
                "## 6. 原始模型指标文件",
                "",
                "`data/processed/model_metrics.csv` 已保留训练脚本输出的整体、组合和时段 MAPE / RMSE 指标。",
            ]
        )
    return "\n".join(lines) + "\n"


def save_report(
    group_metrics: pd.DataFrame,
    submission_checks: list[str],
    metrics: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(group_metrics, submission_checks, metrics)
    (REPORT_DIR / "模型评估摘要.md").write_text(report, encoding="utf-8")


def main() -> None:
    validation, metrics, submission, template = load_inputs()
    submission_checks = validate_submission(submission, template)
    group_metrics = build_group_metrics(validation)

    save_group_metrics(group_metrics)
    save_figures(validation, group_metrics)
    save_report(group_metrics, submission_checks, metrics)

    overall = group_metrics[
        (group_metrics["scope"] == "overall") & (group_metrics["group"] == "all")
    ].iloc[0]
    best_model = choose_best_model_from_group_metrics(group_metrics)
    print("Evaluation completed.")
    print(f"Ridge MAPE: {overall['ridge_mape']:.6f}")
    print(f"XGBoost MAPE: {overall['xgboost_mape']:.6f}")
    print(f"Ridge RMSE: {overall['ridge_rmse']:.6f}")
    print(f"XGBoost RMSE: {overall['xgboost_rmse']:.6f}")
    print(f"Group metrics: {PROCESSED_DIR / 'evaluation_group_metrics.csv'}")
    print(f"Report: {REPORT_DIR / '模型评估摘要.md'}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
