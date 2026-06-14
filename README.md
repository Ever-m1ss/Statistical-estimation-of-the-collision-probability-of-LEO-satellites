# LEO 卫星碰撞概率分析

本项目基于 TLE 轨道数据，对近距离掠过的 LEO 卫星对进行碰撞概率估计与统计检验。

当前代码主要由四个可运行模块组成：

- `collision_analysis.py`：轨道几何计算、蒙特卡洛估计、重要性采样、子集模拟。
- `ground_truth_analysis.py`：局部数值积分、大样本蒙特卡洛、Bootstrap、偏差-方差分析。
- `statistical_testing.py`：重复实验、ROC/AUC、功效分析与统计检验结果汇总。
- `hypothesis_test.py`：通用假设检验函数、功效曲线和样本量估计工具。

## 输入数据

- `3le.txt`：三行式 TLE 数据，格式为名称、line 1、line 2。

## 输出结果

- `results/summary.json`：`collision_analysis.py` 的主分析结果。
- `results/subsim_levels.json`：子集模拟各层级诊断信息。
- `results/ground_truth_summary.json`：真值分析、MC、Bootstrap、偏差-方差结果。
- `results/statistical_testing_summary.json`：ROC 和功效汇总结果。
- `results/**/*.png`：生成的诊断图。

## 安装

```bash
pip install -r requirements.txt
```

依赖项：

- `numpy`
- `scipy`
- `matplotlib`

## 运行

运行主分析：

```bash
python collision_analysis.py
```

运行真值与不确定性分析：

```bash
python ground_truth_analysis.py
```

运行重复统计检验实验：

```bash
python statistical_testing.py
```

运行假设检验示例：

```bash
python hypothesis_test.py
```

## 说明

- 距离计算基于 TLE 平均轨道根数和二体开普勒模型，适合统计方法实验，不适合作为实务中的会合规避判定。
- 碰撞半径在 `collision_analysis.py` 中定义为 `R_COLL = 10 km`。
- 项目已不再保留论文生成脚本，结果统一输出为 `results/` 下的 JSON 和 PNG 文件。
- 目前源码保持扁平结构，便于直接脚本运行和简单导入。
