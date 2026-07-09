# CM3020 AI Mid-term Coursework — Mountain Climbing GA

## 项目概述
遗传算法爬山生物进化项目。在 PyBullet 物理仿真环境中，进化出能够爬上高斯的生物。

## 文件结构

| 文件 | 用途 |
|---|---|
| `train.py` | **[v2.0]** 统一训练入口，300 代默认，CLI 参数化 |
| `auto_experiment.py` | **[v2.0]** 全自动三阶段参数搜索系统 |
| `genome.py` | **[v2.0]** 基因组编码/解码/交叉/突变（Bug 已修复） |
| `creature.py` | **[v2.0]** 生物类 + Motor（Bug 已修复） |
| `population.py` | 种群管理 + 选择操作 |
| `simulation.py` | PyBullet 仿真运行器 |
| `mountain_ga.py` | [v1.0] 原始实验脚本（保留对照） |
| `run_experiments.py` | [v1.0] 批量实验入口 |
| `prepare_shapes.py` | 山体 URDF 生成 |
| `cw-envt.py` | 原始环境 GUI 展示 |
| `code_integration.md` | **[v2.0]** 全部核心代码 + 版本标注 |

## 训练入口

```bash
python train.py                              # 默认 300 代
python train.py --pop 20 --genes 5 --label exp1
python auto_experiment.py --phase1-only      # Phase1 验证 (~3h)
python auto_experiment.py                    # 全三阶段 (~11h)
```

## v2.0 关键改动
- Motor 振幅现在实际生效
- point_mutate amount 参数不再硬编码
- 新增 disconnect 防止 PyBullet 连接泄漏
- 三阶段自适应实验系统

## 虚拟环境
```powershell
.venv\Scripts\Activate.ps1
```