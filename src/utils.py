# 公共工具函数
# 全体成员共用

from __future__ import annotations

import numpy as np


def calculate_mape(y_true, y_pred):
    """计算 MAPE（Mean Absolute Percentage Error）"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid_mask = y_true != 0
    if not valid_mask.any():
        return np.nan

    return np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask]))


def calculate_rmse(y_true, y_pred):
    """计算 RMSE（Root Mean Squared Error）"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))
