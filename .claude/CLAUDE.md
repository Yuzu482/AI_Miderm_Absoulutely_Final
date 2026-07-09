# CM3020 AI Mid-term Coursework — Mountain Climbing GA

## 项目概述
遗传算法爬山生物进化项目。在 PyBullet 物理仿真环境中，进化出能够爬上高斯的生物。

## 文件结构

| 文件 | 用途 |
|---|---|
| `mountain_ga.py` | 主 GA 训练 + MountainSimulation 类 |
| `genome.py` | 基因组编码/解码/交叉/突变 |
| `creature.py` | 生物类（从基因构建 URDF） |
| `population.py` | 种群管理 + 选择操作 |
| `simulation.py` | PyBullet 仿真运行器 |
| `cw-envt.py` | 原始环境 GUI 展示 |
| `prepare_shapes.py` | 山体 URDF 生成 |
| `run_experiments.py` | 批量实验入口 |
| `test_ga.py` / `test_ga_no_threads.py` | GA 测试 |

## GA 关键参数
- `pop_size`: 种群大小（默认 10）
- `gene_count`: 基因数量（默认 3）
- `mutation_rate`: 突变率（默认 0.1）
- `generations`: 进化代数（默认 100）
- `elitism`: 精英策略（默认开启）

## 实验结果
CSV 文件格式: `elite_mountain_{实验名}_gen{代数}.csv`
实验结果图: `experiment_results.png`
实验数据: `experiment_results.json`

## 虚拟环境
```powershell
.venv\Scripts\Activate.ps1
```
