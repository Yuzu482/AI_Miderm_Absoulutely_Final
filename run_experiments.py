"""
Complete GA mountain-climbing experiment runner.
Runs multiple parameter configurations and produces comparison charts.
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mountain_ga as mga


EXPERIMENTS = [
    {"name": "Baseline",        "pop_size": 10, "gene_count": 3, "generations": 50,
     "mutation_rate": 0.1, "shrink_rate": 0.25, "grow_rate": 0.1},
    {"name": "Large Pop (20)",  "pop_size": 20, "gene_count": 3, "generations": 50,
     "mutation_rate": 0.1, "shrink_rate": 0.25, "grow_rate": 0.1},
    {"name": "More Genes (5)",  "pop_size": 10, "gene_count": 5, "generations": 50,
     "mutation_rate": 0.1, "shrink_rate": 0.25, "grow_rate": 0.1},
    {"name": "High Mutation",   "pop_size": 10, "gene_count": 3, "generations": 50,
     "mutation_rate": 0.3, "shrink_rate": 0.25, "grow_rate": 0.1},
]

RESULTS_FILE = "experiment_results.json"


def run_all():
    all_results = {}
    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Running: {exp['name']}")
        print(f"{'='*60}")
        history = mga.run_experiment(
            pop_size=exp["pop_size"],
            gene_count=exp["gene_count"],
            generations=exp["generations"],
            mutation_rate=exp["mutation_rate"],
            shrink_rate=exp["shrink_rate"],
            grow_rate=exp["grow_rate"],
            label=exp["name"].replace(" ", "_"))
        all_results[exp["name"]] = {
            "config": exp,
            "history": [(int(g), float(b), float(m) if m != float('inf') else None,
                         float(ml), int(mxl)) for g, b, m, ml, mxl in history]
        }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    return all_results


def plot_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    # 1. Best fitness (min distance to peak) over generations
    ax = axes[0, 0]
    for (name, data), color in zip(results.items(), colors):
        history = data["history"]
        gens = [h[0] for h in history]
        bests = [h[1] for h in history]
        ax.plot(gens, bests, color=color, linewidth=1.5, alpha=0.9, label=name)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Min-Distance to Peak")
    ax.set_title("Best Fitness (lower = closer to peak)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Mean fitness (non-inf only)
    ax = axes[0, 1]
    for (name, data), color in zip(results.items(), colors):
        history = data["history"]
        gens = [h[0] for h in history]
        means = [h[2] if h[2] is not None else float('nan') for h in history]
        # filter out NaN for plotting
        valid = [(g, m) for g, m in zip(gens, means) if not np.isnan(m)]
        if valid:
            gs, ms = zip(*valid)
            ax.plot(gs, ms, color=color, linewidth=1.5, alpha=0.9, label=name)
        else:
            ax.plot(gens, [0]*len(gens), color=color, linewidth=1.5, alpha=0.3,
                    label=f"{name} (no valid)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Min-Distance to Peak")
    ax.set_title("Mean Fitness (non-flying creatures)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Mean link count
    ax = axes[1, 0]
    for (name, data), color in zip(results.items(), colors):
        history = data["history"]
        gens = [h[0] for h in history]
        mean_links = [h[3] for h in history]
        ax.plot(gens, mean_links, color=color, linewidth=1.5, alpha=0.9, label=name)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Link Count")
    ax.set_title("Body Complexity (mean links)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Final best fitness comparison (bar chart)
    ax = axes[1, 1]
    names = list(results.keys())
    final_bests = [results[n]["history"][-1][1] for n in names]
    bars = ax.bar(names, final_bests, color=colors, edgecolor='white')
    ax.set_ylabel("Final Best Min-Distance to Peak")
    ax.set_title("Final Best Fitness Comparison")
    for bar, val in zip(bars, final_bests):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("experiment_results.png", dpi=150, bbox_inches='tight')
    print("Charts saved to experiment_results.png")


def print_summary_table(results):
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    header = f"{'Experiment':<25} {'Final Best':>12} {'Final Mean':>12} {'Mean Links':>12}"
    print(header)
    print("-"*80)
    for name, data in results.items():
        h = data["history"]
        final_best = h[-1][1]
        final_mean = h[-1][2]
        final_links = h[-1][3]
        mean_str = f"{final_mean:.3f}" if final_mean is not None else "inf"
        print(f"{name:<25} {final_best:>12.4f} {mean_str:>12} {final_links:>12.1f}")
    print("="*80)


if __name__ == "__main__":
    results = run_all()
    print_summary_table(results)
    plot_results(results)