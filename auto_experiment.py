"""
auto_experiment.py — 全自动迭代实验系统 (v2.0)
==============================================
三阶段自适应参数搜索：粗网格 → 聚焦优化 → 消融验证。
输出全套对比图表 + 汇总 JSON，直接用于 Part B 报告。

用法:
  python auto_experiment.py                      # 全三阶段 (~11h)
  python auto_experiment.py --phase1-only        # 仅 Phase 1 验证 (~3h)
  python auto_experiment.py --phases phase1 phase3  # 指定阶段
  python auto_experiment.py --seeds 42 99 777    # 自定义种子
  python auto_experiment.py --top-n 5            # Phase 2 聚焦 Top 5
"""

import argparse
import copy
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train import run_ga

# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ParamConfig:
    """单个实验的参数配置."""
    label: str
    pop_size: int = 10
    gene_count: int = 3
    generations: int = 300
    mutation_rate: float = 0.1
    mut_amount: float = 0.25
    shrink_rate: float = 0.25
    grow_rate: float = 0.1
    elitism: bool = True
    sim_iterations: int = 2400
    phase: str = ""
    parent_config: str = ""
    rationale: str = ""


@dataclass
class SeedResult:
    """单个种子 run 的结果."""
    seed: int
    final_best: float
    total_time_s: float
    history: list = field(default_factory=list)


@dataclass
class ConfigResult:
    """单个配置跨所有种子的聚合结果."""
    config: ParamConfig
    seeds: List[SeedResult] = field(default_factory=list)

    @property
    def mean_best(self) -> float:
        return float(np.mean([s.final_best for s in self.seeds]))

    @property
    def std_best(self) -> float:
        if len(self.seeds) <= 1:
            return 0.0
        return float(np.std([s.final_best for s in self.seeds]))

    @property
    def best_seed(self) -> float:
        return float(np.min([s.final_best for s in self.seeds]))

    @property
    def mean_time_s(self) -> float:
        return float(np.mean([s.total_time_s for s in self.seeds]))


# ═══════════════════════════════════════════════════════════════════════════
# Strategy Classes
# ═══════════════════════════════════════════════════════════════════════════


class BroadGridStrategy:
    """
    Phase 1: 粗粒度极值网格。
    对 6 个参数各取 2 个极值，共 64 组合，200 代快速扫描。
    """

    def __init__(self):
        self.grid = {
            "pop_size": [5, 20],
            "gene_count": [2, 8],
            "mutation_rate": [0.02, 0.2],
            "mut_amount": [0.05, 0.4],
            "shrink_rate": [0.05, 0.4],
            "grow_rate": [0.05, 0.35],
        }

    @staticmethod
    def _make_label(params: dict) -> str:
        return (f"P1_p{params['pop_size']}_g{params['gene_count']}"
                f"_m{params['mutation_rate']}_a{params['mut_amount']}"
                f"_s{params['shrink_rate']}_gr{params['grow_rate']}")

    def generate_configs(self) -> List[ParamConfig]:
        keys = list(self.grid.keys())
        values = list(self.grid.values())
        configs = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            configs.append(ParamConfig(
                label=self._make_label(params),
                pop_size=params["pop_size"],
                gene_count=params["gene_count"],
                generations=200,
                mutation_rate=params["mutation_rate"],
                mut_amount=params["mut_amount"],
                shrink_rate=params["shrink_rate"],
                grow_rate=params["grow_rate"],
                phase="phase1",
                rationale="Broad grid search — testing extremes",
            ))
        return configs


class AdaptiveFocusStrategy:
    """
    Phase 2: 在 Phase 1 的 Top N 配置周围做局部精细扫描。
    """

    def __init__(self, top_n: int = 3):
        self.top_n = top_n
        self.top_configs: List[ParamConfig] = []

    def update_from_results(self, results: List[ConfigResult]) -> None:
        sorted_results = sorted(results, key=lambda r: r.mean_best)
        self.top_configs = [r.config for r in sorted_results[:self.top_n]]

    def generate_configs(self) -> List[ParamConfig]:
        configs = []
        for base in self.top_configs:
            base_dict = asdict(base)

            # 精调 mutation_rate
            for m in [base.mutation_rate * 0.5, base.mutation_rate * 1.5]:
                if 0.005 <= m <= 0.5:
                    cfg = self._derived(base_dict, mutation_rate=round(m, 4),
                                        rationale=f"Refine mut_rate near {base.label}")
                    configs.append(cfg)

            # 精调 mut_amount
            for a in [base.mut_amount * 0.5, base.mut_amount * 1.5]:
                if 0.01 <= a <= 0.5:
                    cfg = self._derived(base_dict, mut_amount=round(a, 4),
                                        rationale=f"Refine mut_amt near {base.label}")
                    configs.append(cfg)

            # 测试 shrink/grow 配比
            for s_mult, g_mult in [(0.5, 1.5), (1.5, 0.5)]:
                cfg = self._derived(base_dict,
                                    shrink_rate=round(base.shrink_rate * s_mult, 4),
                                    grow_rate=round(base.grow_rate * g_mult, 4),
                                    rationale=f"Shrink/grow balance near {base.label}")
                configs.append(cfg)

            # 精调 pop_size
            for ps in [max(5, base.pop_size - 5), base.pop_size + 5]:
                if ps != base.pop_size:
                    cfg = self._derived(base_dict, pop_size=ps,
                                        rationale=f"Refine pop_size near {base.label}")
                    configs.append(cfg)

            # 精调 gene_count
            for gc in [max(1, base.gene_count - 1), base.gene_count + 1]:
                if gc != base.gene_count and 1 <= gc <= 10:
                    cfg = self._derived(base_dict, gene_count=gc,
                                        rationale=f"Refine gene_count near {base.label}")
                    configs.append(cfg)

        return self._deduplicate(configs)

    def _derived(self, base_dict: dict, **overrides) -> ParamConfig:
        real_overrides = {k: v for k, v in overrides.items() if k != "rationale"}
        params = {**base_dict, **real_overrides, "phase": "phase2",
                  "generations": 300, "parent_config": base_dict["label"]}
        rationale = overrides.get("rationale", "")
        return ParamConfig(**{**params, "rationale": rationale})

    def _deduplicate(self, configs: List[ParamConfig]) -> List[ParamConfig]:
        seen = set()
        unique = []
        for c in configs:
            key = (c.pop_size, c.gene_count, c.mutation_rate, c.mut_amount,
                   c.shrink_rate, c.grow_rate)
            if key not in seen:
                seen.add(key)
                # deduplicated label
                c.label = (f"P2_p{c.pop_size}_g{c.gene_count}"
                           f"_m{c.mutation_rate}_a{c.mut_amount}"
                           f"_s{c.shrink_rate}_gr{c.grow_rate}")
                unique.append(c)
        return unique


class AblationStrategy:
    """
    Phase 3: 基于全局最优配置做消融实验 + 代数缩放。
    """

    def __init__(self):
        self.best_config: Optional[ParamConfig] = None

    def update_from_results(self, results: List[ConfigResult]) -> None:
        sorted_results = sorted(results, key=lambda r: r.mean_best)
        self.best_config = sorted_results[0].config

    def generate_configs(self) -> List[ConfigResult]:
        if self.best_config is None:
            return []

        base = asdict(self.best_config)
        configs = []

        # 消融：依次关闭每个突变算子
        ablations = [
            ("no_point_mut", {"mutation_rate": 0.0}, "Disable point mutation"),
            ("no_shrink", {"shrink_rate": 0.0}, "Disable shrink mutation"),
            ("no_grow", {"grow_rate": 0.0}, "Disable grow mutation"),
            ("no_elitism", {"elitism": False}, "Disable elitism"),
        ]
        for suffix, overrides, rationale in ablations:
            params = {**base, **overrides, "phase": "phase3",
                      "parent_config": self.best_config.label,
                      "label": f"P3_{self.best_config.label}_{suffix}",
                      "rationale": rationale}
            configs.append(ParamConfig(**params))

        # 代数缩放
        for gens in [150, 500]:
            params = {**base, "generations": gens, "phase": "phase3",
                      "parent_config": self.best_config.label,
                      "label": f"P3_{self.best_config.label}_gens{gens}",
                      "rationale": f"Convergence at {gens} gens"}
            configs.append(ParamConfig(**params))

        return configs


# ═══════════════════════════════════════════════════════════════════════════
# Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════


class ExperimentRunner:
    """遍历配置 × 种子，调用 train.run_ga() 执行实验."""

    def __init__(self, out_dir: str = "output/auto_exp",
                 seeds: Optional[List[int]] = None):
        self.out_dir = out_dir
        self.seeds = seeds or [42, 123, 456]
        self.results: List[ConfigResult] = []
        self.start_time: Optional[float] = None
        self.completed = 0
        self.total = 0
        os.makedirs(out_dir, exist_ok=True)

    def run_phase(self, configs: List[ParamConfig],
                  phase_name: str) -> List[ConfigResult]:
        """执行一个阶段的所有实验."""
        self.total = len(configs) * len(self.seeds)
        self.completed = 0
        self.start_time = time.time()
        phase_results: List[ConfigResult] = []

        print(f"\n{'#' * 60}")
        print(f"# {phase_name.upper()}: {len(configs)} configs x {len(self.seeds)} seeds")
        print(f"# Total runs: {self.total}")
        print(f"{'#' * 60}\n")

        for ci, config in enumerate(configs):
            seed_results: List[SeedResult] = []
            for seed in self.seeds:
                self.completed += 1
                elapsed = time.time() - self.start_time
                eta = (elapsed / self.completed) * (self.total - self.completed) if self.completed > 0 else 0

                print(f"[{self.completed}/{self.total}] {config.label} "
                      f"seed={seed} | elapsed: {elapsed / 60:.1f}m | "
                      f"ETA: {eta / 60:.1f}m")

                np.random.seed(seed)

                run_label = f"{config.label}_s{seed}"
                run_ga(
                    pop_size=config.pop_size,
                    gene_count=config.gene_count,
                    generations=config.generations,
                    mutation_rate=config.mutation_rate,
                    mut_amount=config.mut_amount,
                    shrink_rate=config.shrink_rate,
                    grow_rate=config.grow_rate,
                    elitism=config.elitism,
                    sim_iterations=config.sim_iterations,
                    label=run_label,
                    out_dir=self.out_dir,
                    keep_elites=False,
                )

                # 读取 run_ga 输出的 JSON
                results_path = os.path.join(
                    self.out_dir, f"results_{run_label}.json")
                try:
                    with open(results_path) as f:
                        run_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"  WARNING: Could not read {results_path}: {e}")
                    run_data = {"final_best": float("inf"),
                                "total_time_s": 0, "history": []}

                seed_results.append(SeedResult(
                    seed=seed,
                    final_best=run_data.get("final_best", float("inf")),
                    total_time_s=run_data.get("total_time_s", 0),
                    history=run_data.get("history", []),
                ))

            phase_results.append(ConfigResult(config=config, seeds=seed_results))

        self.results.extend(phase_results)
        return phase_results


# ═══════════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════════

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0",
          "#00BCD4", "#795548", "#607D8B", "#E91E63", "#CDDC39"]


def _align_histories(histories, max_len):
    """对齐不同长度的 history 数组."""
    curves = []
    for h in histories:
        if not h:
            continue
        vals = [row[1] for row in h]  # best fitness
        if len(vals) < max_len:
            vals = vals + [vals[-1]] * (max_len - len(vals))
        curves.append(vals)
    return curves


def plot_phase_comparison(results, phase_name, out_dir, top_n=8):
    """Phase 结果对比：收敛曲线 + 柱状图 + link 曲线 + 耗时."""
    top = sorted(results, key=lambda r: r.mean_best)[:top_n]
    if not top:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{phase_name}: Top {len(top)} Configurations",
                 fontsize=14, fontweight="bold")

    # 左上：收敛曲线（含标准差阴影）
    ax = axes[0, 0]
    for i, (r, col) in enumerate(zip(top, COLORS)):
        histories = [s.history for s in r.seeds if s.history]
        if not histories:
            continue
        max_len = max(len(h) for h in histories)
        curves = _align_histories(histories, max_len)
        mean_c = np.mean(curves, axis=0)
        std_c = np.std(curves, axis=0)
        gens = list(range(len(mean_c)))
        lbl = f"{r.config.label[:30]} ({r.mean_best:.4f})"
        ax.plot(gens, mean_c, color=col, linewidth=1.3, label=lbl)
        ax.fill_between(gens, mean_c - std_c, mean_c + std_c,
                        color=col, alpha=0.12)
    ax.set_title("Best Fitness (mean ± std across seeds)")
    ax.set_xlabel("Generation"); ax.set_ylabel("Min-Distance to Peak")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # 右上：最终 best 柱状图
    ax = axes[0, 1]
    names = [r.config.label[:25] for r in top]
    means = [r.mean_best for r in top]
    stds = [r.std_best for r in top]
    ax.bar(names, means, yerr=stds, color=COLORS[:len(top)],
           capsize=4, edgecolor="white")
    ax.set_title("Final Best (with std dev)")
    ax.set_ylabel("Min-Distance"); ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    # 左下：mean links 曲线
    ax = axes[1, 0]
    for i, (r, col) in enumerate(zip(top, COLORS)):
        histories = [s.history for s in r.seeds if s.history]
        if not histories:
            continue
        max_len = max(len(h) for h in histories)
        curves = [[row[3] for row in h] + ([h[-1][3]] * (max_len - len(h)))
                  for h in histories]
        mean_l = np.mean(curves, axis=0)
        ax.plot(range(len(mean_l)), mean_l, color=col, linewidth=1.1,
                label=r.config.label[:25])
    ax.set_title("Mean Link Count Over Generations")
    ax.set_xlabel("Generation"); ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # 右下：运行时间
    ax = axes[1, 1]
    times = [r.mean_time_s for r in top]
    ax.barh(names, times, color=COLORS[:len(top)])
    ax.set_title("Mean Runtime per Config (s)")
    ax.set_xlabel("Seconds")

    plt.tight_layout()
    path = os.path.join(out_dir, f"{phase_name}_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {path}")


def plot_sensitivity(all_results, out_dir):
    """参数敏感度散点图."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Parameter Sensitivity Analysis", fontsize=14, fontweight="bold")

    params = [
        ("pop_size", "Population Size"),
        ("gene_count", "Gene Count"),
        ("mutation_rate", "Mutation Rate"),
        ("mut_amount", "Mutation Amount"),
        ("shrink_rate", "Shrink Rate"),
        ("grow_rate", "Grow Rate"),
    ]

    for ax, (param, title) in zip(axes.flat, params):
        xs = [getattr(r.config, param) for r in all_results]
        ys = [r.mean_best for r in all_results]
        ax.scatter(xs, ys, alpha=0.5, c="#2196F3", edgecolors="white")
        ax.set_xlabel(param); ax.set_ylabel("Mean Best Fitness")
        ax.set_title(title); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "parameter_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {path}")


def plot_overall_convergence(all_results, out_dir):
    """跨阶段最优配置收敛对比."""
    fig, ax = plt.subplots(figsize=(12, 6))
    phases = {}
    for r in all_results:
        phases.setdefault(r.config.phase, []).append(r)

    for i, (phase, results) in enumerate(sorted(phases.items())):
        if not results:
            continue
        best = min(results, key=lambda r: r.mean_best)
        histories = [s.history for s in best.seeds if s.history]
        if not histories:
            continue
        max_len = max(len(h) for h in histories)
        curves = _align_histories(histories, max_len)
        mean_c = np.mean(curves, axis=0)
        std_c = np.std(curves, axis=0)
        gens = list(range(len(mean_c)))
        ax.plot(gens, mean_c, color=COLORS[i], linewidth=1.8,
                label=f"{phase} best: {best.config.label[:40]} ({best.mean_best:.5f})")
        ax.fill_between(gens, mean_c - std_c, mean_c + std_c,
                        color=COLORS[i], alpha=0.1)

    ax.set_title("Cross-Phase Convergence Comparison")
    ax.set_xlabel("Generation"); ax.set_ylabel("Min-Distance to Peak")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, "overall_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Master JSON & Summary
# ═══════════════════════════════════════════════════════════════════════════


def build_master(phase_results: Dict[str, List[ConfigResult]],
                 total_time_s: float, seeds: List[int]) -> dict:
    """构建 master.json."""
    master = {
        "experiment": "GA Mountain Climbing — Automated Parameter Search (v2.0)",
        "timestamp": datetime.now().isoformat(),
        "total_time_h": round(total_time_s / 3600, 1),
        "total_runs": sum(len(cfgs) * len(seeds) for cfgs in phase_results.values()),
        "seeds": seeds,
        "phases": {},
    }

    all_configs = []
    for pname, results in phase_results.items():
        sorted_r = sorted(results, key=lambda r: r.mean_best)
        master["phases"][pname] = {
            "num_configs": len(results),
            "top_5": [
                {"rank": i + 1, "label": r.config.label,
                 "mean_best": r.mean_best, "std_best": r.std_best,
                 "config": asdict(r.config)}
                for i, r in enumerate(sorted_r[:5])
            ],
        }
        all_configs.extend(sorted_r)

    if all_configs:
        best = min(all_configs, key=lambda r: r.mean_best)
        master["overall_best"] = {
            "label": best.config.label,
            "phase": best.config.phase,
            "mean_best": best.mean_best,
            "std_best": best.std_best,
            "best_seed": best.best_seed,
            "config": asdict(best.config),
        }

    return master


def print_summary(master: dict):
    """控制台打印汇总."""
    print("\n" + "=" * 70)
    print("AUTO-EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Total runtime: {master['total_time_h']} hours")
    print(f"  Total runs:    {master['total_runs']}")

    for pname, pdata in master["phases"].items():
        top = pdata["top_5"][0] if pdata["top_5"] else None
        print(f"\n  [{pname}]  {pdata['num_configs']} configs tested")
        if top:
            print(f"    Best: {top['label']}  "
                  f"(mean={top['mean_best']:.5f}, std={top['std_best']:.5f})")

    if master.get("overall_best"):
        ob = master["overall_best"]
        print(f"\n  ★ OVERALL BEST ★")
        print(f"    Config: {ob['label']}  ({ob['phase']})")
        print(f"    Mean:   {ob['mean_best']:.5f} ± {ob['std_best']:.5f}")
        print(f"    Config params: pop={ob['config']['pop_size']}, "
              f"genes={ob['config']['gene_count']}, "
              f"mut={ob['config']['mutation_rate']}, "
              f"mut_amt={ob['config']['mut_amount']}")

    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


def run_auto_experiments(out_dir: str = "output/auto_exp",
                         seeds: Optional[List[int]] = None,
                         phases_to_run: Optional[List[str]] = None,
                         phase2_top_n: int = 3):
    """全自动实验主流程."""
    seeds = seeds or [42, 123, 456]
    phases_to_run = phases_to_run or ["phase1", "phase2", "phase3"]
    os.makedirs(out_dir, exist_ok=True)

    t_total_start = time.time()
    runner = ExperimentRunner(out_dir=out_dir, seeds=seeds)
    all_phase_results: Dict[str, List[ConfigResult]] = {}

    # ── Phase 1: Broad Grid ──
    if "phase1" in phases_to_run:
        s1 = BroadGridStrategy()
        cfgs = s1.generate_configs()
        print(f"Phase 1: {len(cfgs)} configs x {len(seeds)} seeds = "
              f"{len(cfgs) * len(seeds)} runs (~{len(cfgs) * len(seeds) * 2.8 / 60:.0f}m)")
        all_phase_results["phase1"] = runner.run_phase(cfgs, "phase1")
        plot_phase_comparison(all_phase_results["phase1"], "phase1", out_dir)

    # ── Phase 2: Adaptive Focus ──
    if "phase2" in phases_to_run:
        s2 = AdaptiveFocusStrategy(top_n=phase2_top_n)
        if "phase1" in all_phase_results:
            s2.update_from_results(all_phase_results["phase1"])
        else:
            s2.top_configs = [
                ParamConfig(label="fallback", pop_size=20, gene_count=3,
                            generations=300, mutation_rate=0.2, mut_amount=0.05,
                            shrink_rate=0.05, grow_rate=0.35, phase="phase2",
                            rationale="Fallback — no Phase 1 data"),
            ]
        cfgs = s2.generate_configs()
        print(f"\nPhase 2: {len(cfgs)} configs x {len(seeds)} seeds = "
              f"{len(cfgs) * len(seeds)} runs "
              f"(from {len(s2.top_configs)} parent configs)")
        all_phase_results["phase2"] = runner.run_phase(cfgs, "phase2")
        plot_phase_comparison(all_phase_results["phase2"], "phase2", out_dir)

        # Feed Phase 2 results back to update Phase 3's best config
        # Combine all prior results
        all_prior = []
        for results in all_phase_results.values():
            all_prior.extend(results)
    else:
        all_prior = []
        for results in all_phase_results.values():
            all_prior.extend(results)

    # ── Phase 3: Ablation ──
    if "phase3" in phases_to_run:
        s3 = AblationStrategy()
        if all_prior:
            s3.update_from_results(all_prior)
        else:
            s3.best_config = ParamConfig(
                label="fallback_best", pop_size=20, gene_count=3,
                generations=300, mutation_rate=0.2, mut_amount=0.05,
                shrink_rate=0.05, grow_rate=0.35, phase="phase3",
                rationale="Fallback — no prior data",
            )
        cfgs = s3.generate_configs()
        print(f"\nPhase 3: {len(cfgs)} configs x {len(seeds)} seeds = "
              f"{len(cfgs) * len(seeds)} runs "
              f"(baseline: {s3.best_config.label if s3.best_config else 'N/A'})")
        all_phase_results["phase3"] = runner.run_phase(cfgs, "phase3")
        plot_phase_comparison(all_phase_results["phase3"], "phase3", out_dir)

    # ── Final ──
    t_total = time.time() - t_total_start
    master = build_master(all_phase_results, t_total, seeds)

    master_path = os.path.join(out_dir, "master.json")
    with open(master_path, "w") as f:
        json.dump(master, f, indent=2)

    # 综合图表
    all_results = []
    for results in all_phase_results.values():
        all_results.extend(results)
    if all_results:
        plot_sensitivity(all_results, out_dir)
        plot_overall_convergence(all_results, out_dir)

    # 汇总报告
    summary_path = os.path.join(out_dir, "summary_report.json")
    report = {
        "title": master["experiment"],
        "timestamp": master["timestamp"],
        "total_runtime_h": master["total_time_h"],
        "overall_best": master.get("overall_best"),
        "phase_summaries": {
            pname: {
                "top_config": pdata["top_5"][0] if pdata["top_5"] else None,
                "num_tested": pdata["num_configs"],
            }
            for pname, pdata in master["phases"].items()
        },
    }
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)

    print_summary(master)
    print(f"\nAll results in: {out_dir}/")
    return master


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated GA Parameter Search (v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_experiment.py                         # Full 3-phase run (~11h)
  python auto_experiment.py --phase1-only            # Quick validation (~3h)
  python auto_experiment.py --phases phase1 phase3   # Skip Phase 2
  python auto_experiment.py --top-n 5                # Focus on Top 5 in Phase 2
  python auto_experiment.py --seeds 42 99            # Custom seeds
        """)
    parser.add_argument("--out", default="output/auto_exp",
                        help="Output directory (default: output/auto_exp)")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456],
                        help="Random seeds (default: 42 123 456)")
    parser.add_argument("--phases", nargs="+",
                        default=["phase1", "phase2", "phase3"],
                        help="Phases to run (default: phase1 phase2 phase3)")
    parser.add_argument("--phase1-only", action="store_true",
                        help="Run only Phase 1 for validation")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Top N configs to focus on in Phase 2 (default: 3)")

    args = parser.parse_args()
    phases = ["phase1"] if args.phase1_only else args.phases
    run_auto_experiments(
        out_dir=args.out,
        seeds=args.seeds,
        phases_to_run=phases,
        phase2_top_n=args.top_n,
    )
