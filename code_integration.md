# CM3020 AI Mid-term — 代码整合文档 (v2.0)

> **版本说明**: v2.0 修复了两个关键 Bug：Motor 振幅应用 和 point_mutate 参数传递。
> 训练代数提升至 300 代，新增独立训练脚本 `train.py` 作为后续实验统一入口。

---

## 版本演进

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 初始 | Baseline 实现，100 代 GA，4 组实验完成 |
| v2.0 | 2026-07-10 | Bug 修复 + 300 代训练 + train.py 统一入口 |

---

## 文件结构

```
AI-Mid/
├── train.py              ← [v2.0 新增] 统一训练入口
├── genome.py             ← [v2.0 修改] Bug 修复
├── creature.py           ← [v2.0 修改] Bug 修复
├── population.py         ← [v1.0 不变]
├── simulation.py         ← [v1.0 不变]
├── mountain_ga.py        ← [v1.0] 原始实验脚本（保留对照）
├── run_experiments.py    ← [v1.0] 批量实验入口
├── prepare_shapes.py     ← [v1.0] 地形生成
├── cw-envt.py            ← [v1.0] GUI 环境演示
├── test_*.py             ← [v1.0] 单元测试
└── code_integration.md   ← [v2.0 新增] 本文档
```

---

## 1. genome.py — 基因组编码与遗传操作

### v2.0 修改点
- **`point_mutate()` 第 129 行**: 硬编码 `0.1` → 使用 `amount` 参数

```python
# genome.py — v2.0
import numpy as np
import copy
import random

class Genome():
    @staticmethod
    def get_random_gene(length):
        gene = np.array([np.random.random() for i in range(length)])
        return gene

    @staticmethod
    def get_random_genome(gene_length, gene_count):
        genome = [Genome.get_random_gene(gene_length) for i in range(gene_count)]
        return genome

    @staticmethod
    def get_gene_spec():
        gene_spec = {
            "link-shape":          {"scale": 1},
            "link-length":         {"scale": 2},
            "link-radius":         {"scale": 1},
            "link-recurrence":     {"scale": 3},
            "link-mass":           {"scale": 1},
            "joint-type":          {"scale": 1},
            "joint-parent":        {"scale": 1},
            "joint-axis-xyz":      {"scale": 1},
            "joint-origin-rpy-1":  {"scale": np.pi * 2},
            "joint-origin-rpy-2":  {"scale": np.pi * 2},
            "joint-origin-rpy-3":  {"scale": np.pi * 2},
            "joint-origin-xyz-1":  {"scale": 1},
            "joint-origin-xyz-2":  {"scale": 1},
            "joint-origin-xyz-3":  {"scale": 1},
            "control-waveform":    {"scale": 1},
            "control-amp":         {"scale": 0.25},
            "control-freq":        {"scale": 1}
        }
        ind = 0
        for key in gene_spec.keys():
            gene_spec[key]["ind"] = ind
            ind = ind + 1
        return gene_spec

    @staticmethod
    def get_gene_dict(gene, spec):
        gdict = {}
        for key in spec:
            ind = spec[key]["ind"]
            scale = spec[key]["scale"]
            gdict[key] = gene[ind] * scale
        return gdict

    @staticmethod
    def get_genome_dicts(genome, spec):
        gdicts = []
        for gene in genome:
            gdicts.append(Genome.get_gene_dict(gene, spec))
        return gdicts

    @staticmethod
    def expandLinks(parent_link, uniq_parent_name, flat_links, exp_links):
        children = [l for l in flat_links if l.parent_name == parent_link.name]
        sibling_ind = 1
        for c in children:
            for r in range(int(c.recur)):
                sibling_ind = sibling_ind + 1
                c_copy = copy.copy(c)
                c_copy.parent_name = uniq_parent_name
                uniq_name = c_copy.name + str(len(exp_links))
                c_copy.name = uniq_name
                c_copy.sibling_ind = sibling_ind
                exp_links.append(c_copy)
                assert c.parent_name != c.name, \
                    "Genome::expandLinks: link joined to itself: " + c.name
                Genome.expandLinks(c, uniq_name, flat_links, exp_links)

    @staticmethod
    def genome_to_links(gdicts):
        links = []
        link_ind = 0
        parent_names = [str(link_ind)]
        for gdict in gdicts:
            link_name = str(link_ind)
            parent_ind = gdict["joint-parent"] * len(parent_names)
            parent_name = parent_names[int(parent_ind)]
            recur = gdict["link-recurrence"]
            link = URDFLink(
                name=link_name,
                parent_name=parent_name,
                recur=recur + 1,
                link_length=gdict["link-length"],
                link_radius=gdict["link-radius"],
                link_mass=gdict["link-mass"],
                joint_type=gdict["joint-type"],
                joint_parent=gdict["joint-parent"],
                joint_axis_xyz=gdict["joint-axis-xyz"],
                joint_origin_rpy_1=gdict["joint-origin-rpy-1"],
                joint_origin_rpy_2=gdict["joint-origin-rpy-2"],
                joint_origin_rpy_3=gdict["joint-origin-rpy-3"],
                joint_origin_xyz_1=gdict["joint-origin-xyz-1"],
                joint_origin_xyz_2=gdict["joint-origin-xyz-2"],
                joint_origin_xyz_3=gdict["joint-origin-xyz-3"],
                control_waveform=gdict["control-waveform"],
                control_amp=gdict["control-amp"],
                control_freq=gdict["control-freq"]
            )
            links.append(link)
            if link_ind != 0:
                parent_names.append(link_name)
            link_ind = link_ind + 1
        links[0].parent_name = "None"
        return links

    @staticmethod
    def crossover(g1, g2):
        x1 = random.randint(0, len(g1) - 1)
        x2 = random.randint(0, len(g2) - 1)
        g3 = np.concatenate((g1[x1:], g2[x2:]))
        if len(g3) > len(g1):
            g3 = g3[0:len(g1)]
        return g3

    # ═══ v2.0 FIX: amount 参数现已生效 ═══
    @staticmethod
    def point_mutate(genome, rate, amount):
        new_genome = copy.copy(genome)
        for gene in new_genome:
            for i in range(len(gene)):
                if random.random() < rate:
                    gene[i] += amount       # v1.0: 硬编码 0.1
                if gene[i] >= 1.0:
                    gene[i] = 0.9999
                if gene[i] < 0.0:
                    gene[i] = 0.0
        return new_genome

    @staticmethod
    def shrink_mutate(genome, rate):
        if len(genome) == 1:
            return copy.copy(genome)
        if random.random() < rate:
            ind = random.randint(0, len(genome) - 1)
            new_genome = np.delete(genome, ind, 0)
            return new_genome
        else:
            return copy.copy(genome)

    @staticmethod
    def grow_mutate(genome, rate):
        if random.random() < rate:
            gene = Genome.get_random_gene(len(genome[0]))
            new_genome = copy.copy(genome)
            new_genome = np.append(new_genome, [gene], axis=0)
            return new_genome
        else:
            return copy.copy(genome)

    @staticmethod
    def to_csv(dna, csv_file):
        csv_str = ""
        for gene in dna:
            for val in gene:
                csv_str = csv_str + str(val) + ","
            csv_str = csv_str + '\n'
        with open(csv_file, 'w') as f:
            f.write(csv_str)

    @staticmethod
    def from_csv(filename):
        csv_str = ''
        with open(filename) as f:
            csv_str = f.read()
        dna = []
        lines = csv_str.split('\n')
        for line in lines:
            vals = line.split(',')
            gene = [float(v) for v in vals if v != '']
            if len(gene) > 0:
                dna.append(gene)
        return dna


class URDFLink:
    def __init__(self, name, parent_name, recur,
                 link_length=0.1, link_radius=0.1, link_mass=0.1,
                 joint_type=0.1, joint_parent=0.1,
                 joint_axis_xyz=0.1,
                 joint_origin_rpy_1=0.1, joint_origin_rpy_2=0.1,
                 joint_origin_rpy_3=0.1,
                 joint_origin_xyz_1=0.1, joint_origin_xyz_2=0.1,
                 joint_origin_xyz_3=0.1,
                 control_waveform=0.1, control_amp=0.1, control_freq=0.1):
        self.name = name
        self.parent_name = parent_name
        self.recur = recur
        self.link_length = link_length
        self.link_radius = link_radius
        self.link_mass = link_mass
        self.joint_type = joint_type
        self.joint_parent = joint_parent
        self.joint_axis_xyz = joint_axis_xyz
        self.joint_origin_rpy_1 = joint_origin_rpy_1
        self.joint_origin_rpy_2 = joint_origin_rpy_2
        self.joint_origin_rpy_3 = joint_origin_rpy_3
        self.joint_origin_xyz_1 = joint_origin_xyz_1
        self.joint_origin_xyz_2 = joint_origin_xyz_2
        self.joint_origin_xyz_3 = joint_origin_xyz_3
        self.control_waveform = control_waveform
        self.control_amp = control_amp
        self.control_freq = control_freq
        self.sibling_ind = 1

    def to_link_element(self, adom):
        link_tag = adom.createElement("link")
        link_tag.setAttribute("name", self.name)
        vis_tag = adom.createElement("visual")
        geom_tag = adom.createElement("geometry")
        cyl_tag = adom.createElement("cylinder")
        cyl_tag.setAttribute("length", str(self.link_length))
        cyl_tag.setAttribute("radius", str(self.link_radius))
        geom_tag.appendChild(cyl_tag)
        vis_tag.appendChild(geom_tag)
        coll_tag = adom.createElement("collision")
        c_geom_tag = adom.createElement("geometry")
        c_cyl_tag = adom.createElement("cylinder")
        c_cyl_tag.setAttribute("length", str(self.link_length))
        c_cyl_tag.setAttribute("radius", str(self.link_radius))
        c_geom_tag.appendChild(c_cyl_tag)
        coll_tag.appendChild(c_geom_tag)
        inertial_tag = adom.createElement("inertial")
        mass_tag = adom.createElement("mass")
        mass = np.pi * (self.link_radius * self.link_radius) * self.link_length
        mass_tag.setAttribute("value", str(mass))
        inertia_tag = adom.createElement("inertia")
        inertia_tag.setAttribute("ixx", "0.03")
        inertia_tag.setAttribute("iyy", "0.03")
        inertia_tag.setAttribute("izz", "0.03")
        inertia_tag.setAttribute("ixy", "0")
        inertia_tag.setAttribute("ixz", "0")
        inertia_tag.setAttribute("iyx", "0")
        inertial_tag.appendChild(mass_tag)
        inertial_tag.appendChild(inertia_tag)
        link_tag.appendChild(vis_tag)
        link_tag.appendChild(coll_tag)
        link_tag.appendChild(inertial_tag)
        return link_tag

    def to_joint_element(self, adom):
        joint_tag = adom.createElement("joint")
        joint_tag.setAttribute("name", self.name + "_to_" + self.parent_name)
        if self.joint_type >= 0.5:
            joint_tag.setAttribute("type", "revolute")
        else:
            joint_tag.setAttribute("type", "revolute")
        parent_tag = adom.createElement("parent")
        parent_tag.setAttribute("link", self.parent_name)
        child_tag = adom.createElement("child")
        child_tag.setAttribute("link", self.name)
        axis_tag = adom.createElement("axis")
        if self.joint_axis_xyz <= 0.33:
            axis_tag.setAttribute("xyz", "1 0 0")
        if self.joint_axis_xyz > 0.33 and self.joint_axis_xyz <= 0.66:
            axis_tag.setAttribute("xyz", "0 1 0")
        if self.joint_axis_xyz > 0.66:
            axis_tag.setAttribute("xyz", "0 0 1")
        limit_tag = adom.createElement("limit")
        limit_tag.setAttribute("effort", "1")
        limit_tag.setAttribute("upper", "-3.1415")
        limit_tag.setAttribute("lower", "3.1415")
        limit_tag.setAttribute("velocity", "1")
        orig_tag = adom.createElement("origin")
        rpy1 = self.joint_origin_rpy_1 * self.sibling_ind
        rpy = str(rpy1) + " " + str(self.joint_origin_rpy_2) + " " + str(self.joint_origin_rpy_3)
        orig_tag.setAttribute("rpy", rpy)
        xyz = str(self.joint_origin_xyz_1) + " " + str(self.joint_origin_xyz_2) + " " + str(self.joint_origin_xyz_3)
        orig_tag.setAttribute("xyz", xyz)
        joint_tag.appendChild(parent_tag)
        joint_tag.appendChild(child_tag)
        joint_tag.appendChild(axis_tag)
        joint_tag.appendChild(limit_tag)
        joint_tag.appendChild(orig_tag)
        return joint_tag
```

---

## 2. creature.py — 生物体与马达控制

### v2.0 修改点
- **`Motor.get_output()` 第 25、27、30 行**: 输出值现在乘以 `self.amp`

```python
# creature.py — v2.0
import genome
from xml.dom.minidom import getDOMImplementation
from enum import Enum
import numpy as np


class MotorType(Enum):
    PULSE = 1
    SINE = 2


class Motor:
    def __init__(self, control_waveform, control_amp, control_freq):
        if control_waveform <= 0.5:
            self.motor_type = MotorType.PULSE
        else:
            self.motor_type = MotorType.SINE
        self.amp = control_amp
        self.freq = control_freq
        self.phase = 0

    # ═══ v2.0 FIX: amp 现在应用到输出 ═══
    def get_output(self):
        self.phase = (self.phase + self.freq) % (np.pi * 2)
        if self.motor_type == MotorType.PULSE:
            if self.phase < np.pi:
                output = self.amp          # v1.0: 硬编码 1
            else:
                output = -self.amp         # v1.0: 硬编码 -1
        if self.motor_type == MotorType.SINE:
            output = self.amp * np.sin(self.phase)  # v1.0: 未乘 amp
        return output


class Creature:
    def __init__(self, gene_count):
        self.spec = genome.Genome.get_gene_spec()
        self.dna = genome.Genome.get_random_genome(len(self.spec), gene_count)
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None
        self.positions = []
        self.min_dist_to_target = float('inf')
        self.target_position = None

    def get_flat_links(self):
        if self.flat_links is None:
            gdicts = genome.Genome.get_genome_dicts(self.dna, self.spec)
            self.flat_links = genome.Genome.genome_to_links(gdicts)
        return self.flat_links

    def get_expanded_links(self):
        self.get_flat_links()
        if self.exp_links is not None:
            return self.exp_links
        exp_links = [self.flat_links[0]]
        genome.Genome.expandLinks(
            self.flat_links[0], self.flat_links[0].name,
            self.flat_links, exp_links)
        self.exp_links = exp_links
        return self.exp_links

    def to_xml(self):
        self.get_expanded_links()
        domimpl = getDOMImplementation()
        adom = domimpl.createDocument(None, "start", None)
        robot_tag = adom.createElement("robot")
        for link in self.exp_links:
            robot_tag.appendChild(link.to_link_element(adom))
        first = True
        for link in self.exp_links:
            if first:
                first = False
                continue
            robot_tag.appendChild(link.to_joint_element(adom))
        robot_tag.setAttribute("name", "pepe")
        return '<?xml version="1.0"?>' + robot_tag.toprettyxml()

    def get_motors(self):
        self.get_expanded_links()
        if self.motors is None:
            motors = []
            for i in range(1, len(self.exp_links)):
                l = self.exp_links[i]
                m = Motor(l.control_waveform, l.control_amp, l.control_freq)
                motors.append(m)
            self.motors = motors
        return self.motors

    def update_position(self, pos):
        if self.start_position is None:
            self.start_position = pos
        else:
            self.last_position = pos
        self.positions.append(pos)
        if self.target_position is not None:
            p1 = np.asarray(pos)
            p2 = np.asarray(self.target_position)
            dist = np.linalg.norm(p1 - p2)
            if dist < self.min_dist_to_target:
                self.min_dist_to_target = dist

    def get_distance_travelled(self):
        if self.start_position is None or self.last_position is None:
            return 0
        p1 = np.asarray(self.start_position)
        p2 = np.asarray(self.last_position)
        dist = np.linalg.norm(p1 - p2)
        return dist

    def get_min_dist_to_target(self):
        return self.min_dist_to_target

    def set_target_position(self, target_pos):
        self.target_position = target_pos
        self.min_dist_to_target = float('inf')

    def update_dna(self, dna):
        self.dna = dna
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None
        self.positions = []
        self.min_dist_to_target = float('inf')
        self.target_position = None
```

---

## 3. population.py — 种群管理与选择

### v1.0（无变更）

```python
# population.py — v1.0
import creature
import numpy as np


class Population:
    def __init__(self, pop_size, gene_count):
        self.creatures = [creature.Creature(gene_count=gene_count)
                          for i in range(pop_size)]

    @staticmethod
    def get_fitness_map(fits):
        fitmap = []
        total = 0
        for f in fits:
            total = total + f
            fitmap.append(total)
        return fitmap

    @staticmethod
    def select_parent(fitmap):
        r = np.random.rand()
        r = r * fitmap[-1]
        for i in range(len(fitmap)):
            if r <= fitmap[i]:
                return i
```

---

## 4. simulation.py — 仿真引擎

### v1.0（无变更）

```python
# simulation.py — v1.0
import pybullet as p
from multiprocessing import Pool


class Simulation:
    def __init__(self, sim_id=0):
        self.physicsClientId = p.connect(p.DIRECT)
        self.sim_id = sim_id

    def run_creature(self, cr, iterations=2400):
        pid = self.physicsClientId
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)
        p.setGravity(0, 0, -10, physicsClientId=pid)
        plane_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=pid)
        floor = p.createMultiBody(plane_shape, plane_shape, physicsClientId=pid)
        xml_file = 'temp' + str(self.sim_id) + '.urdf'
        xml_str = cr.to_xml()
        with open(xml_file, 'w') as f:
            f.write(xml_str)
        cid = p.loadURDF(xml_file, physicsClientId=pid)
        p.resetBasePositionAndOrientation(cid, [0, 0, 2.5], [0, 0, 0, 1],
                                           physicsClientId=pid)
        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)
            if step % 24 == 0:
                self.update_motors(cid=cid, cr=cr)
            pos, orn = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
            cr.update_position(pos)

    def update_motors(self, cid, cr):
        for jid in range(p.getNumJoints(cid,
                                        physicsClientId=self.physicsClientId)):
            m = cr.get_motors()[jid]
            p.setJointMotorControl2(cid, jid,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=m.get_output(),
                                    force=5,
                                    physicsClientId=self.physicsClientId)

    def eval_population(self, pop, iterations):
        for cr in pop.creatures:
            self.run_creature(cr, 2400)


class ThreadedSim():
    def __init__(self, pool_size):
        self.sims = [Simulation(i) for i in range(pool_size)]

    @staticmethod
    def static_run_creature(sim, cr, iterations):
        sim.run_creature(cr, iterations)
        return cr

    def eval_population(self, pop, iterations):
        pool_args = []
        start_ind = 0
        pool_size = len(self.sims)
        while start_ind < len(pop.creatures):
            this_pool_args = []
            for i in range(start_ind, start_ind + pool_size):
                if i == len(pop.creatures):
                    break
                sim_ind = i % len(self.sims)
                this_pool_args.append([
                    self.sims[sim_ind],
                    pop.creatures[i],
                    iterations])
            pool_args.append(this_pool_args)
            start_ind = start_ind + pool_size
        new_creatures = []
        for pool_argset in pool_args:
            with Pool(pool_size) as p:
                creatures = p.starmap(ThreadedSim.static_run_creature, pool_argset)
                new_creatures.extend(creatures)
        pop.creatures = new_creatures
```

---

## 5. train.py — 统一训练入口 (v2.0 新增)

```python
# train.py — v2.0
"""
统一 GA 训练脚本。
整合 Bug 修复后的所有模块，作为后续实验的唯一起点。

用法:
  python train.py                           # 默认 300 代
  python train.py --pop 20 --genes 5        # 自定义参数
  python train.py --label my_experiment     # 指定实验标签
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pybullet as p
import pybullet_data

import genome
import creature
import population

# ── Mountain Environment ──
PEAK_POS = (0, 0, 4)
MOUNTAIN_H = 5
MOUNTAIN_SIGMA = 5
MOUNTAIN_BASE_Z = -1


def gaussian_height(x, y, height=MOUNTAIN_H, sigma=MOUNTAIN_SIGMA):
    return height * np.exp(-((x**2 + y**2) / (2 * sigma**2)))


def make_arena(arena_size=20, wall_height=1):
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
    mountain_position = (0, 0, MOUNTAIN_BASE_Z)
    mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
    p.setAdditionalSearchPath('shapes/')
    return p.loadURDF("gaussian_pyramid.urdf", mountain_position,
                       mountain_orientation, useFixedBase=1)


def surface_z_at(x, y):
    return MOUNTAIN_BASE_Z + gaussian_height(x, y)


def is_flying(pos, threshold=1.0):
    x, y, z = pos
    return z > surface_z_at(x, y) + threshold


# ── Trainer ──
class Trainer:
    def __init__(self, sim_id=0):
        self.pid = p.connect(p.DIRECT)
        self.sim_id = sim_id

    def simulate(self, cr, iterations=2400):
        pid = self.pid
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pid)
        p.setGravity(0, 0, -10, physicsClientId=pid)
        make_arena(arena_size=20)
        load_mountain()
        cr.set_target_position(PEAK_POS)
        xml_file = 'temp' + str(self.sim_id) + '.urdf'
        with open(xml_file, 'w') as f:
            f.write(cr.to_xml())
        cid = p.loadURDF(xml_file, physicsClientId=pid)
        p.resetBasePositionAndOrientation(cid, [0, 0, 5],
                                          [0, 0, 0, 1], physicsClientId=pid)
        GRACE_STEPS = 200
        flying_count = 0
        total_checked = 0
        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)
            if step % 24 == 0:
                self._update_motors(cid, cr)
            pos, _ = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
            cr.update_position(pos)
            if step >= GRACE_STEPS:
                total_checked += 1
                if is_flying(pos):
                    flying_count += 1
        if total_checked > 0 and flying_count / total_checked > 0.5:
            cr.min_dist_to_target = float('inf')

    def _update_motors(self, cid, cr):
        for jid in range(p.getNumJoints(cid, physicsClientId=self.pid)):
            m = cr.get_motors()[jid]
            p.setJointMotorControl2(cid, jid,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=m.get_output(),
                                    force=5, physicsClientId=self.pid)


# ── GA Loop ──
def run_ga(pop_size=10, gene_count=3, generations=300,
           mutation_rate=0.1, mut_amount=0.25,
           shrink_rate=0.25, grow_rate=0.1,
           elitism=True, sim_iterations=2400,
           label="train", out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    pop = population.Population(pop_size=pop_size, gene_count=gene_count)
    trainer = Trainer(sim_id=0)
    history = []
    best_overall_fit = float('inf')
    best_overall_dna = None
    t_start = time.time()

    for gen in range(generations):
        for cr in pop.creatures:
            trainer.simulate(cr, iterations=sim_iterations)

        fits = [cr.get_min_dist_to_target() for cr in pop.creatures]
        links = [len(cr.get_expanded_links()) for cr in pop.creatures]
        best_fit = np.min(fits)
        valid_fits = [f for f in fits if f != float('inf')]
        mean_fit = np.mean(valid_fits) if valid_fits else float('inf')

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

        inv_fits = [1.0 / (f + 0.01) for f in fits]
        fit_map = population.Population.get_fitness_map(inv_fits)
        new_creatures = []
        for _ in range(len(pop.creatures)):
            p1_ind = population.Population.select_parent(fit_map)
            p2_ind = population.Population.select_parent(fit_map)
            dna = genome.Genome.crossover(pop.creatures[p1_ind].dna,
                                          pop.creatures[p2_ind].dna)
            dna = genome.Genome.point_mutate(dna, rate=mutation_rate,
                                             amount=mut_amount)
            dna = genome.Genome.shrink_mutate(dna, rate=shrink_rate)
            dna = genome.Genome.grow_mutate(dna, rate=grow_rate)
            cr = creature.Creature(1)
            cr.update_dna(dna)
            new_creatures.append(cr)

        if elitism:
            best_idx = np.argmin(fits)
            elite = creature.Creature(1)
            elite.update_dna(pop.creatures[best_idx].dna)
            new_creatures[0] = elite
            genome.Genome.to_csv(
                elite.dna,
                os.path.join(out_dir, f"elite_{label}_gen{gen}.csv"))

        pop.creatures = new_creatures

    results = {
        "label": label,
        "config": {
            "pop_size": pop_size, "gene_count": gene_count,
            "generations": generations, "mutation_rate": mutation_rate,
            "mut_amount": mut_amount, "shrink_rate": shrink_rate,
            "grow_rate": grow_rate, "elitism": elitism,
            "sim_iterations": sim_iterations,
        },
        "final_best": float(best_overall_fit),
        "total_time_s": round(time.time() - t_start, 1),
        "timestamp": datetime.now().isoformat(),
        "history": [(int(g), float(b),
                      float(m) if m != float('inf') else None,
                      float(ml), int(mxl))
                     for g, b, m, ml, mxl in history]
    }
    results_path = os.path.join(out_dir, f"results_{label}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    if best_overall_dna is not None:
        genome.Genome.to_csv(best_overall_dna,
                             os.path.join(out_dir, f"best_{label}.csv"))
    print(f"\n{'='*60}")
    print(f"Training complete: {label}")
    print(f"  Best fitness:   {best_overall_fit:.5f}")
    print(f"  Total time:     {time.time() - t_start:.0f}s")
    print(f"  Results saved:  {results_path}")
    print(f"{'='*60}")
    return history


# ── CLI ──
DEFAULT_CONFIG = {
    "pop_size": 10, "gene_count": 3, "generations": 300,
    "mutation_rate": 0.1, "mut_amount": 0.25,
    "shrink_rate": 0.25, "grow_rate": 0.1,
    "elitism": True, "sim_iterations": 2400,
}

def parse_args():
    p = argparse.ArgumentParser(description="GA Mountain Climbing — v2.0")
    p.add_argument("--pop", type=int, default=10)
    p.add_argument("--genes", type=int, default=3)
    p.add_argument("--gens", type=int, default=300)
    p.add_argument("--mut", type=float, default=0.1)
    p.add_argument("--mut-amount", type=float, default=0.25)
    p.add_argument("--shrink", type=float, default=0.25)
    p.add_argument("--grow", type=float, default=0.1)
    p.add_argument("--label", type=str, default="train")
    p.add_argument("--out", type=str, default="output")
    p.add_argument("--no-elitism", action="store_true")
    p.add_argument("--iterations", type=int, default=2400)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_ga(pop_size=args.pop, gene_count=args.genes, generations=args.gens,
           mutation_rate=args.mut, mut_amount=args.mut_amount,
           shrink_rate=args.shrink, grow_rate=args.grow,
           elitism=not args.no_elitism, sim_iterations=args.iterations,
           label=args.label, out_dir=args.out)
```

---

## GA 参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pop_size` | 10 | 种群大小 |
| `gene_count` | 3 | 初始基因数（每个基因 17 个参数） |
| `generations` | 300 | 进化代数 |
| `mutation_rate` | 0.1 | 点突变触发概率 |
| `mut_amount` | 0.25 | 点突变变异幅度 |
| `shrink_rate` | 0.25 | 收缩突变概率 |
| `grow_rate` | 0.1 | 增长突变概率 |
| `sim_iterations` | 2400 | 仿真步数（240Hz = 10秒） |
| `elitism` | True | 精英策略开关 |

## v2.0 Bug 修复清单

| Bug | 位置 | v1.0 | v2.0 |
|-----|------|------|------|
| Motor 振幅未应用 | `creature.py:25-30` | `output = ±1 / sin(phase)` | `output = self.amp * (±1 / sin(phase))` |
| point_mutate amount 硬编码 | `genome.py:129` | `gene[i] += 0.1` | `gene[i] += amount` |