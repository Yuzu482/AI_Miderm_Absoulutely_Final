import pybullet as p
import pybullet_data
import math
import random
import numpy as np
import genome
import creature
import population
import simulation


# ── Mountain environment helpers ──────────────────────────────────────────

def gaussian_height(x, y, height=5, sigma=5):
    return height * math.exp(-((x**2 + y**2) / (2 * sigma**2)))


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
    mountain_position = (0, 0, -1)
    mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
    p.setAdditionalSearchPath('shapes/')
    return p.loadURDF("gaussian_pyramid.urdf", mountain_position,
                      mountain_orientation, useFixedBase=1)


PEAK_POS = (0, 0, 4)       # mountain peak in world coords
MOUNTAIN_H = 5
MOUNTAIN_SIGMA = 5
MOUNTAIN_BASE_Z = -1


def surface_z_at(x, y):
    return MOUNTAIN_BASE_Z + gaussian_height(x, y, MOUNTAIN_H, MOUNTAIN_SIGMA)


def is_flying(pos, threshold=1.0):
    """True if the creature is significantly above the mountain surface."""
    x, y, z = pos
    return z > surface_z_at(x, y) + threshold


# ── Mountain simulation ──────────────────────────────────────────────────

class MountainSimulation:
    def __init__(self, sim_id=0):
        self.physicsClientId = p.connect(p.DIRECT)
        self.sim_id = sim_id

    def run_creature(self, cr, iterations=2400):
        pid = self.physicsClientId
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pid)

        p.setGravity(0, 0, -10, physicsClientId=pid)
        make_arena(arena_size=20)
        load_mountain()

        cr.set_target_position(PEAK_POS)

        xml_file = 'temp' + str(self.sim_id) + '.urdf'
        xml_str = cr.to_xml()
        with open(xml_file, 'w') as f:
            f.write(xml_str)

        cid = p.loadURDF(xml_file, physicsClientId=pid)
        p.resetBasePositionAndOrientation(cid, [0, 0, 5],
                                          [0, 0, 0, 1], physicsClientId=pid)

        GRACE_STEPS = 200
        flying_count = 0
        total_checked = 0
        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)
            if step % 24 == 0:
                self._update_motors(cid, cr, pid)

            pos, _ = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
            cr.update_position(pos)
            if step >= GRACE_STEPS:
                total_checked += 1
                if is_flying(pos):
                    flying_count += 1

        if total_checked > 0 and flying_count / total_checked > 0.5:
            cr.min_dist_to_target = float('inf')

    def _update_motors(self, cid, cr, pid):
        for jid in range(p.getNumJoints(cid, physicsClientId=pid)):
            m = cr.get_motors()[jid]
            p.setJointMotorControl2(cid, jid,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=m.get_output(),
                                    force=5,
                                    physicsClientId=pid)


# ── GA loop ──────────────────────────────────────────────────────────────

def run_experiment(pop_size=10, gene_count=3, generations=100,
                   mutation_rate=0.1, shrink_rate=0.25, grow_rate=0.1,
                   elitism=True, label=""):
    pop = population.Population(pop_size=pop_size, gene_count=gene_count)
    sim = MountainSimulation()

    history = []
    for gen in range(generations):
        for cr in pop.creatures:
            sim.run_creature(cr, iterations=2400)

        fits = [cr.get_min_dist_to_target() for cr in pop.creatures]
        links = [len(cr.get_expanded_links()) for cr in pop.creatures]

        best_fit = np.min(fits)
        mean_fit = np.mean(fits)

        print(f"[{label}] gen {gen:4d} | best(min_dist): {best_fit:.3f} | "
              f"mean: {mean_fit:.3f} | mean_links: {np.mean(links):.1f} | "
              f"max_links: {np.max(links)}")

        history.append((gen, best_fit, mean_fit, np.mean(links), np.max(links)))

        # selection uses inverse of distance (closer = better)
        inv_fits = [1.0 / (f + 0.01) for f in fits]
        fit_map = population.Population.get_fitness_map(inv_fits)
        new_creatures = []
        for _ in range(len(pop.creatures)):
            p1_ind = population.Population.select_parent(fit_map)
            p2_ind = population.Population.select_parent(fit_map)
            dna = genome.Genome.crossover(pop.creatures[p1_ind].dna,
                                          pop.creatures[p2_ind].dna)
            dna = genome.Genome.point_mutate(dna, rate=mutation_rate, amount=0.25)
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
            genome.Genome.to_csv(elite.dna, f"elite_mountain_{label}_gen{gen}.csv")

        pop.creatures = new_creatures

    return history


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Experiment 1: baseline (pop=10, gene_count=3, gen=100) ===")
    h1 = run_experiment(pop_size=10, gene_count=3, generations=100,
                        label="baseline")

    print("\n=== Experiment 2: larger pop (pop=20, gene_count=3, gen=100) ===")
    h2 = run_experiment(pop_size=20, gene_count=3, generations=100,
                        label="pop20")

    print("\n=== Experiment 3: more genes (pop=10, gene_count=5, gen=100) ===")
    h3 = run_experiment(pop_size=10, gene_count=5, generations=100,
                        label="gene5")

    print("\n=== Experiment 4: high mutation (pop=10, gene_count=3, gen=100, mut=0.3) ===")
    h4 = run_experiment(pop_size=10, gene_count=3, generations=100,
                        mutation_rate=0.3, label="high_mut")

    print("\nDone. Histories saved in variables h1-h4.")