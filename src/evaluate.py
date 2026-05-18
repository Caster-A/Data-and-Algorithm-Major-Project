# 验证评估、结果分析与报告整合
# 负责人：成员五

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import calculate_mape


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
    """按多个维度统计各模型的 MAPE。"""
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
            row = {"scope": scope, "group": str(group), "rows": len(group_df)}
            for model_name, pred_col in available_models.items():
                row[f"{model_name}_mape"] = calculate_mape(
                    group_df["target_volume"], group_df[pred_col]
                )
            rows.append(row)

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


def plot_daily_average_volume(validation: pd.DataFrame) -> None:
    """绘制按日期聚合的真实与预测平均流量折线图。"""
    best_model, best_pred_col = choose_best_prediction_column(validation)
    daily_avg = (
        validation.assign(date=validation["target_time"].dt.strftime("%m-%d"))
        .groupby("date", as_index=False)
        .agg(actual=("target_volume", "mean"), predicted=(best_pred_col, "mean"))
    )
    plot_df = daily_avg.melt(
        id_vars="date",
        value_vars=["actual", "predicted"],
        var_name="series",
        value_name="average_volume",
    )

    plt.figure(figsize=(9, 4.2))
    sns.lineplot(
        data=plot_df,
        x="date",
        y="average_volume",
        hue="series",
        marker="o",
        linewidth=2.4,
        palette={"actual": "#2563eb", "predicted": "#dc2626"},
    )
    plt.title(f"Validation Daily Average Volume: actual vs {best_model}")
    plt.xlabel("date")
    plt.ylabel("average volume")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "validation_daily_avg_volume.png")
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
    plot_daily_average_volume(validation)
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
        f"- Per-combo XGBoost MAPE：`{overall.get('per_combo_xgboost_mape', float('nan')):.6f}`",
        f"- Hybrid XGBoost MAPE：`{overall.get('hybrid_xgboost_mape', float('nan')):.6f}`",
        f"- 当前最佳模型：`{best_model}`",
        f"- 验证样本数：`{int(overall['rows'])}`",
        "",
        "## 2. 提交文件检查",
        "",
    ]
    lines.extend([f"- {item}" for item in submission_checks])
    lines.extend(
        [
            "",
            "## 3. 主要误差来源",
            "",
            f"- 当前误差最高的收费站方向组合是 `{worst_combo['group']}`，{best_model} MAPE 为 `{worst_combo[best_metric_col]:.6f}`。",
            f"- 当前误差最高的预测步长是 `horizon_step={worst_horizon['group']}`，{best_model} MAPE 为 `{worst_horizon[best_metric_col]:.6f}`。",
            "",
            "## 4. 生成图表",
            "",
            "- `figures/validation_actual_vs_pred.png`",
            "- `figures/validation_daily_avg_volume.png`",
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
                "`data/processed/model_metrics.csv` 已保留训练脚本输出的整体、组合和时段指标。",
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
    if "per_combo_xgboost_mape" in overall:
        print(f"Per-combo XGBoost MAPE: {overall['per_combo_xgboost_mape']:.6f}")
    if "hybrid_xgboost_mape" in overall:
        print(f"Hybrid XGBoost MAPE: {overall['hybrid_xgboost_mape']:.6f}")
    print(f"Best model: {best_model}")
    print(f"Group metrics: {PROCESSED_DIR / 'evaluation_group_metrics.csv'}")
    print(f"Report: {REPORT_DIR / '模型评估摘要.md'}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
