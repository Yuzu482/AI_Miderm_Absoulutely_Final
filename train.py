"""
train.py — 爬山 GA 训练入口脚本（v2.0）
=========================================
基于 mountain_ga.py 重构，整合 Bug 修复后的代码模块。
作为后续所有实验的统一训练起点。

修复内容（相对于 v1.0）:
  - Motor.get_output() 现在正确应用 control-amp 参数
  - point_mutate() 的 amount 参数不再被硬编码覆盖

使用方式:
  python train.py                          # 默认配置运行
  python train.py --pop 20 --genes 5       # 自定义参数
  python train.py --help                   # 查看所有参数
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pybullet as p
import pybullet_data

import genome
import creature
import population


# ═══════════════════════════════════════════════════════════════════════════
# Mountain Environment
# ═══════════════════════════════════════════════════════════════════════════

PEAK_POS = (0, 0, 4)
MOUNTAIN_H = 5
MOUNTAIN_SIGMA = 5
MOUNTAIN_BASE_Z = -1


def gaussian_height(x, y, height=MOUNTAIN_H, sigma=MOUNTAIN_SIGMA):
    return height * np.exp(-((x**2 + y**2) / (2 * sigma**2)))


def make_arena(arena_size=20, wall_height=3):
    wall_thickness = 0.5
    floor_collision = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[arena_size / 2, arena_size / 2, wall_thickness])
    floor_visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[arena_size / 2, arena_size / 2, wall_thickness],
        rgbaColor=[1, 1, 0, 1])
    p.createMultiBody(0, floor_collision, floor_visual,
                      basePosition=[0, 0, -wall_thickness])

    wall_collision = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2])
    wall_visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2],
        rgbaColor=[0.7, 0.7, 0.7, 1])
    p.createMultiBody(0, wall_collision, wall_visual,
                      basePosition=[0, arena_size / 2, wall_height / 2])
    p.createMultiBody(0, wall_collision, wall_visual,
                      basePosition=[0, -arena_size / 2, wall_height / 2])

    wall_collision2 = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2])
    wall_visual2 = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2],
        rgbaColor=[0.7, 0.7, 0.7, 1])
    p.createMultiBody(0, wall_collision2, wall_visual2,
                      basePosition=[arena_size / 2, 0, wall_height / 2])
    p.createMultiBody(0, wall_collision2, wall_visual2,
                      basePosition=[-arena_size / 2, 0, wall_height / 2])


TERRAIN_FILES = {
    "gaussian": "gaussian_pyramid.urdf",
    "pyramid": "pyramid.urdf",
    "rocky": "rocky_mountain.urdf",
}


def ensure_terrains():
    """确保所有地形 OBJ/URDF 文件存在，缺失则自动生成."""
    import prepare_shapes as ps
    os.makedirs("shapes", exist_ok=True)
    if not os.path.exists("shapes/pyramid.obj"):
        ps.make_pyramid("shapes/pyramid.obj")
    if not os.path.exists("shapes/pyramid.urdf"):
        _write_terrain_urdf("shapes/pyramid.urdf", "pyramid.obj")
    if not os.path.exists("shapes/rocky_mountain.obj"):
        ps.make_rocky_moutain("shapes/rocky_mountain.obj")
    if not os.path.exists("shapes/rocky_mountain.urdf"):
        _write_terrain_urdf("shapes/rocky_mountain.urdf", "rocky_mountain.obj")


def _write_terrain_urdf(urdf_path, obj_name):
    urdf = f'''<?xml version="1.0"?>
<robot name="terrain">
  <link name="baseLink">
    <visual>
      <geometry><mesh filename="{obj_name}" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="{obj_name}" scale="1 1 1"/></geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>'''
    with open(urdf_path, 'w') as f:
        f.write(urdf)


def load_mountain(terrain="gaussian"):
    mountain_position = (0, 0, MOUNTAIN_BASE_Z)
    mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
    p.setAdditionalSearchPath('shapes/')
    filename = TERRAIN_FILES.get(terrain, "gaussian_pyramid.urdf")
    return p.loadURDF(filename, mountain_position,
                       mountain_orientation, useFixedBase=1)


def surface_z_at(x, y):
    return MOUNTAIN_BASE_Z + gaussian_height(x, y)


def is_flying(pos, threshold=1.0):
    x, y, z = pos
    return z > surface_z_at(x, y) + threshold


# ═══════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    封装单次训练的完整流程：仿真 + GA 循环 + 结果记录。
    每个 Trainer 实例持有一个独立的 PyBullet DIRECT 连接。
    """

    def __init__(self, sim_id=0, terrain="gaussian"):
        self.pid = p.connect(p.DIRECT)
        self.sim_id = sim_id
        self.terrain = terrain

    def simulate(self, cr, iterations=2400):
        """对单个生物运行仿真，返回适应度（min_dist_to_target）。"""
        pid = self.pid
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pid)

        p.setGravity(0, 0, -10, physicsClientId=pid)
        make_arena(arena_size=20)
        load_mountain(self.terrain)

        cr.set_target_position(PEAK_POS)

        xml_file = 'temp' + str(self.sim_id) + '.urdf'
        with open(xml_file, 'w') as f:
            f.write(cr.to_xml())

        cid = p.loadURDF(xml_file, physicsClientId=pid)
        p.resetBasePositionAndOrientation(cid, [-8, -8, 3],
                                          [0, 0, 0, 1], physicsClientId=pid)

        SETTLE_STEPS = 200
        GRACE_STEPS = 200
        flying_count = 0
        total_checked = 0
        out_of_bounds = False

        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)
            if step >= SETTLE_STEPS and step % 24 == 0:
                self._update_motors(cid, cr)

            pos, _ = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
            cr.update_position(pos)

            # 越界检测：防止出生爆炸弹飞 → 卡墙刷分
            x, y, z = pos
            if abs(x) > 9.0 or abs(y) > 9.0:
                out_of_bounds = True
                break

            if step >= GRACE_STEPS:
                total_checked += 1
                if is_flying(pos):
                    flying_count += 1

        if out_of_bounds:
            cr.min_dist_to_target = float('inf')
        elif total_checked > 0 and flying_count / total_checked > 0.5:
            cr.min_dist_to_target = float('inf')

    def _update_motors(self, cid, cr):
        for jid in range(p.getNumJoints(cid, physicsClientId=self.pid)):
            m = cr.get_motors()[jid]
            p.setJointMotorControl2(cid, jid,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=m.get_output(),
                                    force=5,
                                    physicsClientId=self.pid)


# ═══════════════════════════════════════════════════════════════════════════
# Parallel Worker (module-level for pickle)
# ═══════════════════════════════════════════════════════════════════════════════

def _sim_worker(args):
    """在子进程中仿真单个生物，返回 fitness."""
    dna, sim_id, iterations, terrain = args
    pid = None
    xml_file = None
    try:
        pid = p.connect(p.DIRECT)
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pid)
        p.setGravity(0, 0, -10, physicsClientId=pid)

        make_arena(arena_size=20)
        load_mountain(terrain)

        cr = creature.Creature(1)
        cr.update_dna(dna)
        cr.set_target_position(PEAK_POS)

        xml_file = f'temp_w{os.getpid()}_{sim_id}.urdf'
        with open(xml_file, 'w') as f:
            f.write(cr.to_xml())

        cid = p.loadURDF(xml_file, physicsClientId=pid)
        p.resetBasePositionAndOrientation(cid, [-8, -8, 3], [0, 0, 0, 1], physicsClientId=pid)

        SETTLE_STEPS = 200
        GRACE_STEPS = 200
        flying_count = 0
        total_checked = 0
        out_of_bounds = False

        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)
            if step >= SETTLE_STEPS and step % 24 == 0:
                for jid in range(p.getNumJoints(cid, physicsClientId=pid)):
                    m = cr.get_motors()[jid]
                    p.setJointMotorControl2(cid, jid,
                                            controlMode=p.VELOCITY_CONTROL,
                                            targetVelocity=m.get_output(),
                                            force=5, physicsClientId=pid)

            pos, _ = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
            cr.update_position(pos)

            x, y, z = pos
            if abs(x) > 9.0 or abs(y) > 9.0:
                out_of_bounds = True
                break

            if step >= GRACE_STEPS:
                total_checked += 1
                if is_flying(pos):
                    flying_count += 1

        if out_of_bounds:
            cr.min_dist_to_target = float('inf')
        elif total_checked > 0 and flying_count / total_checked > 0.5:
            cr.min_dist_to_target = float('inf')

        fitness = cr.min_dist_to_target
    except Exception:
        fitness = float('inf')
    finally:
        if pid is not None:
            p.disconnect(pid)
        if xml_file is not None:
            try:
                os.remove(xml_file)
            except OSError:
                pass
    return fitness


# ═══════════════════════════════════════════════════════════════════════════
# GA Loop
# ═══════════════════════════════════════════════════════════════════════════

def run_ga(pop_size=10, gene_count=3, generations=300,
           mutation_rate=0.1, mut_amount=0.25,
           shrink_rate=0.25, grow_rate=0.1,
           elitism=True, sim_iterations=2400,
           label="train", out_dir="output",
           keep_elites=False, terrain="gaussian",
           freeze_indices=None, force_motor=None,
           n_workers=1):
    """
    运行完整 GA 训练。

    Parameters
    ----------
    pop_size : int       种群大小
    gene_count : int     初始基因数量
    generations : int    进化代数
    mutation_rate : float 点突变触发概率
    mut_amount : float   点突变变异幅度
    shrink_rate : float  收缩突变概率（删除基因）
    grow_rate : float    增长突变概率（新增基因）
    elitism : bool       是否启用精英策略
    sim_iterations : int 每次仿真步数（240Hz）
    label : str          实验标签
    out_dir : str        输出目录

    Returns
    -------
    history : list[(gen, best, mean, mean_links, max_links)]
    best_cr  : Creature  最终代最优个体
    """
    os.makedirs(out_dir, exist_ok=True)

    ensure_terrains()
    pop = population.Population(pop_size=pop_size, gene_count=gene_count)
    trainer = Trainer(sim_id=0, terrain=terrain) if n_workers <= 1 else None

    # 多进程：在循环外创建一次 Pool，跨代复用
    pool = Pool(processes=n_workers) if n_workers > 1 else None

    history = []
    best_overall_fit = float('inf')
    best_overall_dna = None

    t_start = time.time()

    for gen in range(generations):
        # ── 仿真评估 ──
        if pool is not None:
            args = [(cr.dna, i, sim_iterations, terrain)
                    for i, cr in enumerate(pop.creatures)]
            fitnesses = pool.map(_sim_worker, args)
            for cr, fit in zip(pop.creatures, fitnesses):
                cr.min_dist_to_target = fit
        else:
            for cr in pop.creatures:
                trainer.simulate(cr, iterations=sim_iterations)

        fits = [cr.get_min_dist_to_target() for cr in pop.creatures]
        links = [len(cr.get_expanded_links()) for cr in pop.creatures]

        best_fit = np.min(fits)
        mean_fit = np.mean([f for f in fits if f != float('inf')]) if any(f != float('inf') for f in fits) else float('inf')

        # 记录全局最优
        if best_fit < best_overall_fit:
            best_overall_fit = best_fit
            best_idx = np.argmin(fits)
            best_overall_dna = [g.copy() for g in pop.creatures[best_idx].dna]

        elapsed = time.time() - t_start
        eta = (elapsed / (gen + 1)) * (generations - gen - 1) if gen > 0 else 0

        print(f"[{label}] gen {gen:4d}/{generations} | "
              f"best: {best_fit:.5f} | mean: {mean_fit:.4f} | "
              f"links(mean): {np.mean(links):.1f} | "
              f"elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

        history.append((gen, best_fit, mean_fit, np.mean(links), np.max(links)))

        # ── 选择 + 繁殖 ──
        inv_fits = [1.0 / (f + 0.01) for f in fits]
        fit_map = population.Population.get_fitness_map(inv_fits)
        new_creatures = []

        for _ in range(len(pop.creatures)):
            p1_ind = population.Population.select_parent(fit_map)
            p2_ind = population.Population.select_parent(fit_map)
            dna = genome.Genome.crossover(pop.creatures[p1_ind].dna,
                                          pop.creatures[p2_ind].dna)
            if freeze_indices:
                dna = genome.Genome.selective_point_mutate(
                    dna, rate=mutation_rate, amount=mut_amount,
                    frozen_indices=freeze_indices)
            else:
                dna = genome.Genome.point_mutate(dna, rate=mutation_rate,
                                                 amount=mut_amount)
            dna = genome.Genome.shrink_mutate(dna, rate=shrink_rate)
            dna = genome.Genome.grow_mutate(dna, rate=grow_rate)
            if force_motor:
                dna = genome.Genome.override_motor_type(dna, force_motor)
            cr = creature.Creature(1)
            cr.update_dna(dna)
            new_creatures.append(cr)

        # ── 精英保留 ──
        if elitism:
            best_idx = np.argmin(fits)
            elite = creature.Creature(1)
            elite.update_dna(pop.creatures[best_idx].dna)
            new_creatures[0] = elite
            if keep_elites:
                genome.Genome.to_csv(elite.dna,
                                     os.path.join(out_dir, f"elite_{label}_gen{gen}.csv"))

        pop.creatures = new_creatures

    # ── 保存结果 ──
    results = {
        "label": label,
        "config": {
            "pop_size": pop_size,
            "gene_count": gene_count,
            "generations": generations,
            "mutation_rate": mutation_rate,
            "mut_amount": mut_amount,
            "shrink_rate": shrink_rate,
            "grow_rate": grow_rate,
            "elitism": elitism,
            "sim_iterations": sim_iterations,
        },
        "final_best": float(best_overall_fit),
        "total_time_s": round(time.time() - t_start, 1),
        "timestamp": datetime.now().isoformat(),
        "history": [(int(g), float(b), float(m) if m != float('inf') else None,
                      float(ml), int(mxl)) for g, b, m, ml, mxl in history]
    }

    results_path = os.path.join(out_dir, f"results_{label}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 保存最终最优个体
    if best_overall_dna is not None:
        genome.Genome.to_csv(best_overall_dna,
                             os.path.join(out_dir, f"best_{label}.csv"))

    if pool is not None:
        pool.close()
        pool.join()
    else:
        p.disconnect(trainer.pid)

    print(f"\n{'='*60}")
    print(f"Training complete: {label}")
    print(f"  Generations:    {generations}")
    print(f"  Best fitness:   {best_overall_fit:.5f}")
    print(f"  Total time:     {time.time() - t_start:.0f}s")
    print(f"  Results saved:  {results_path}")
    print(f"{'='*60}")

    return history


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "pop_size": 10,
    "gene_count": 3,
    "generations": 300,
    "mutation_rate": 0.1,
    "mut_amount": 0.25,
    "shrink_rate": 0.25,
    "grow_rate": 0.1,
    "elitism": True,
    "sim_iterations": 2400,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="GA Mountain Climbing — Training Script (v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py
  python train.py --pop 20 --genes 5 --label large_pop_5genes
  python train.py --mut 0.3 --mut-amount 0.15 --label high_mut
  python train.py --gens 500 --pop 15 --label deep_search
        """)
    p.add_argument("--pop", type=int, default=DEFAULT_CONFIG["pop_size"],
                   help=f"Population size (default: {DEFAULT_CONFIG['pop_size']})")
    p.add_argument("--genes", type=int, default=DEFAULT_CONFIG["gene_count"],
                   help=f"Initial gene count (default: {DEFAULT_CONFIG['gene_count']})")
    p.add_argument("--gens", type=int, default=DEFAULT_CONFIG["generations"],
                   help=f"Generations (default: {DEFAULT_CONFIG['generations']})")
    p.add_argument("--mut", type=float, default=DEFAULT_CONFIG["mutation_rate"],
                   help=f"Point mutation rate (default: {DEFAULT_CONFIG['mutation_rate']})")
    p.add_argument("--mut-amount", type=float, default=DEFAULT_CONFIG["mut_amount"],
                   help=f"Point mutation delta (default: {DEFAULT_CONFIG['mut_amount']})")
    p.add_argument("--shrink", type=float, default=DEFAULT_CONFIG["shrink_rate"],
                   help=f"Shrink mutation rate (default: {DEFAULT_CONFIG['shrink_rate']})")
    p.add_argument("--grow", type=float, default=DEFAULT_CONFIG["grow_rate"],
                   help=f"Grow mutation rate (default: {DEFAULT_CONFIG['grow_rate']})")
    p.add_argument("--label", type=str, default="train",
                   help="Experiment label (default: train)")
    p.add_argument("--out", type=str, default="output",
                   help="Output directory (default: output)")
    p.add_argument("--no-elitism", action="store_true",
                   help="Disable elitism")
    p.add_argument("--iterations", type=int, default=DEFAULT_CONFIG["sim_iterations"],
                   help=f"Simulation steps per creature (default: {DEFAULT_CONFIG['sim_iterations']})")
    p.add_argument("--keep-elites", action="store_true",
                   help="Save elite CSV per generation (default: False, saves disk)")
    p.add_argument("--terrain", type=str, default="gaussian",
                   choices=["gaussian", "pyramid", "rocky"],
                   help="Terrain type (default: gaussian)")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel workers (default: 1, sequential)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ga(
        pop_size=args.pop,
        gene_count=args.genes,
        generations=args.gens,
        mutation_rate=args.mut,
        mut_amount=args.mut_amount,
        shrink_rate=args.shrink,
        grow_rate=args.grow,
        elitism=not args.no_elitism,
        sim_iterations=args.iterations,
        label=args.label,
        out_dir=args.out,
        keep_elites=args.keep_elites,
        terrain=args.terrain,
        n_workers=args.workers,
    )