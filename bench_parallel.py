"""
bench_parallel.py — 多进程提速基准测试
========================================
快速验证多进程并行仿真的实际加速比。
"""

import time
import os
import numpy as np
from multiprocessing import Pool
import pybullet as p
import pybullet_data

import genome
import creature

# ═══════════════════════════════════════════════════════════════
# 复用 train.py 的环境设置
# ═══════════════════════════════════════════════════════════════

PEAK_POS = (0, 0, 4)
MOUNTAIN_H = 5
MOUNTAIN_SIGMA = 5
MOUNTAIN_BASE_Z = -1


def gaussian_height(x, y):
    return MOUNTAIN_H * np.exp(-((x**2 + y**2) / (2 * MOUNTAIN_SIGMA**2)))


def surface_z_at(x, y):
    return MOUNTAIN_BASE_Z + gaussian_height(x, y)


def is_flying(pos, threshold=1.0):
    x, y, z = pos
    return z > surface_z_at(x, y) + threshold


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


def load_mountain():
    p.setAdditionalSearchPath('shapes/')
    return p.loadURDF("gaussian_pyramid.urdf", (0, 0, MOUNTAIN_BASE_Z),
                       p.getQuaternionFromEuler((0, 0, 0)), useFixedBase=1)


# ═══════════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════════

def _sim_worker(args):
    """模块级 worker：独立 PyBullet 连接，返回 fitness."""
    dna, sim_id, iterations, seed = args
    np.random.seed(seed)

    pid = p.connect(p.DIRECT)
    p.resetSimulation(physicsClientId=pid)
    p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pid)
    p.setGravity(0, 0, -10, physicsClientId=pid)

    make_arena(arena_size=20)
    load_mountain()

    cr = creature.Creature(1)
    cr.update_dna(dna)
    cr.set_target_position(PEAK_POS)

    xml_file = f'temp_bench_{os.getpid()}_{sim_id}.urdf'
    with open(xml_file, 'w') as f:
        f.write(cr.to_xml())

    cid = p.loadURDF(xml_file, physicsClientId=pid)
    p.resetBasePositionAndOrientation(cid, [0, 0, 5], [0, 0, 0, 1], physicsClientId=pid)

    GRACE_STEPS = 200
    flying_count = 0
    total_checked = 0
    out_of_bounds = False

    for step in range(iterations):
        p.stepSimulation(physicsClientId=pid)
        if step % 24 == 0:
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

    p.disconnect(pid)
    return cr.min_dist_to_target


# ═══════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════

def run_benchmark(pop_size=20, gene_count=3, generations=5,
                  sim_iterations=2400, n_workers=1):
    """单代仿真计时（多进程版本）."""
    pop = [creature.Creature(gene_count=gene_count) for _ in range(pop_size)]
    dnas = [cr.dna for cr in pop]

    t0 = time.time()
    if n_workers == 1:
        # 单进程：顺序仿真（模拟当前 train.py 行为）
        for i, dna in enumerate(dnas):
            _sim_worker((dna, i, sim_iterations, i))
    else:
        args = [(dnas[i], i, sim_iterations, i) for i in range(pop_size)]
        with Pool(processes=n_workers) as pool:
            pool.map(_sim_worker, args)

    elapsed = time.time() - t0
    return elapsed


if __name__ == "__main__":
    import sys

    GEN = 5  # 每轮跑 5 代取平均
    RUNS = 3  # 跑 3 轮取平均

    print("=" * 60)
    print("多进程 PyBullet 仿真基准测试")
    print(f"pop=20, gene_count=3, {GEN} 代 × {RUNS} 轮取平均")
    print("=" * 60)

    baseline = None
    for n in [1, 2, 4, 6, 8]:
        times = []
        for r in range(RUNS):
            sys.stdout.write(f"  Workers={n}, run {r+1}/{RUNS}...")
            sys.stdout.flush()
            t = run_benchmark(n_workers=n, generations=GEN)
            times.append(t)
            print(f" {t:.1f}s")

        avg = np.mean(times)
        if n == 1:
            baseline = avg
            print(f"  → avg: {avg:.1f}s (baseline)")
        else:
            speedup = baseline / avg
            print(f"  → avg: {avg:.1f}s, speedup: {speedup:.1f}x")

    print(f"\n预估加速效果 (300 代, 2 种子, Phase 4):")
    if baseline:
        single_300 = baseline * (300 / GEN)  # 单进程 300 代
        for n in [4, 6, 8]:
            avg_t = np.mean([run_benchmark(n_workers=n, generations=GEN) for _ in range(2)])
            parallel_300 = avg_t * (300 / GEN)
            total = parallel_300 * 6 * 2  # 6 configs × 2 seeds
            hours = total / 3600
            print(f"  {n} workers: {total:.0f}s = {hours:.1f}h for Phase 4")