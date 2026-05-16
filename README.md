# 数据与算法大作业

KDD Cup 2017 高速公路收费站交通流量预测

## 项目结构

```
├── data/
│   ├── raw/                 # 原始数据
│   ├── processed/           # 清洗和聚合后的数据
│   └── submission/          # 最终提交文件
├── src/
│   ├── data_cleaning.py     # 数据清洗、异常值处理与时间窗口聚合
│   ├── feature_engineering.py  # 特征工程与滑动窗口样本构造
│   ├── train_model.py       # 模型训练、调参与融合
│   ├── evaluate.py          # 验证评估、结果分析
│   └── utils.py             # 公共工具函数
├── notebooks/
│   └── analysis.ipynb       # 探索性分析
├── figures/                 # 图表输出
├── report/                  # 课程报告与答辩 PPT
└── 小组分工.md
```

## 项目环境要求

### 1. Python 版本

建议使用 Python 3.9 或以上版本，推荐版本：

```bash
Python 3.9+
```

本项目主要使用 Python 完成数据清洗、特征工程、模型训练、验证评估和结果生成。

### 2. 推荐运行环境

建议使用独立虚拟环境，避免依赖包版本冲突。

使用 Anaconda，也可以创建 Conda 环境：

```bash
conda create -n traffic-flow python=3.9
conda activate traffic-flow
```

### 3. 依赖库要求

项目建议安装以下 Python 依赖：

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn jupyter openpyxl
```

各依赖用途如下：

| 依赖库 | 主要用途 |
| --- | --- |
| `numpy` | 数值计算、数组处理 |
| `pandas` | 数据读取、清洗、时间窗口聚合、特征表构造 |
| `scikit-learn` | Linear/Ridge、RandomForest、GBDT、模型评估与数据划分 |
| `xgboost` | XGBoost 回归模型训练与预测 |
| `matplotlib` | 绘制流量趋势、误差分析、特征重要性图 |
| `seaborn` | 辅助可视化和统计图绘制 |
| `jupyter` | 运行 `notebooks/analysis.ipynb` 进行探索性分析 |
| `openpyxl` | 读写 Excel 文件，便于整理实验结果 |

### 4. 目录准备

首次运行前，需要确保以下目录存在：

```bash
mkdir -p data/raw data/processed data/submission figures report
```

原始数据应放入：

```text
data/raw/
```

中间处理结果建议输出到：

```text
data/processed/
```

最终预测提交文件建议输出到：

```text
data/submission/
```

### 5. 环境验证

安装完成后，可运行以下命令检查核心依赖是否安装成功：

```bash
python - <<'PY'
import numpy
import pandas
import sklearn
import xgboost
import matplotlib
import seaborn

print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("xgboost:", xgboost.__version__)
print("matplotlib:", matplotlib.__version__)
print("seaborn:", seaborn.__version__)
print("Environment OK")
PY
```

### 6. 推荐运行顺序

在完成数据放置和环境配置后，建议按以下顺序运行项目脚本：

```bash
python src/data_cleaning.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate.py
```

其中：

- `src/data_cleaning.py` 负责数据清洗、异常值处理和 20 分钟窗口聚合。
- `src/feature_engineering.py` 负责滑动窗口样本构造和多源特征融合。
- `src/train_model.py` 负责基线模型、XGBoost 模型训练与预测。
- `src/evaluate.py` 负责 MAPE 计算、误差分析和提交文件检查。

### 7. 注意事项

- 训练集和验证集必须按照时间顺序划分，不能随机打乱。
- 构造特征时只能使用预测窗口之前的数据，不能引入未来信息。
- 最终提交文件需要包含 `tollgate_id`、`time_window`、`direction`、`volume` 四个字段。
- `volume` 预测值应保证非负。
- 如在不同电脑上运行，建议统一 Python 版本和依赖库版本，减少结果差异。
