"""GUI 可视化最优精英个体"""
import pybullet as p
import pybullet_data
import numpy as np
import genome
import creature
import time

# 加载最优 DNA
dna_path = "output/auto_exp/best_P2_p20_g8_m0.2_a0.4_s0.6_gr0.175_s42.csv"
dna = genome.Genome.from_csv(dna_path)
print(f"Loaded DNA: {len(dna)} genes")

cr = creature.Creature(1)
cr.update_dna(dna)
cr.set_target_position((0, 0, 4))

# GUI 模式
pid = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetSimulation(physicsClientId=pid)
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pid)
p.setGravity(0, 0, -10, physicsClientId=pid)
p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)

# 调整视角
p.resetDebugVisualizerCamera(
    cameraDistance=15, cameraYaw=45, cameraPitch=-30,
    cameraTargetPosition=[0, 0, 2], physicsClientId=pid)

# 场地 + 山
PEAK_POS = (0, 0, 4)
MOUNTAIN_H = 5
MOUNTAIN_SIGMA = 5
MOUNTAIN_BASE_Z = -1

def gaussian_height(x, y):
    return MOUNTAIN_H * np.exp(-((x**2 + y**2) / (2 * MOUNTAIN_SIGMA**2)))

def make_arena(arena_size=20, wall_height=3):
    wall_thickness = 0.5
    floor_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness])
    floor_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness], rgbaColor=[1,1,0,1])
    p.createMultiBody(0, floor_collision, floor_visual, basePosition=[0, 0, -wall_thickness])
    wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2])
    wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2], rgbaColor=[0.7,0.7,0.7,1])
    p.createMultiBody(0, wall_collision, wall_visual, basePosition=[0, arena_size/2, wall_height/2])
    p.createMultiBody(0, wall_collision, wall_visual, basePosition=[0, -arena_size/2, wall_height/2])
    wall_collision2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2])
    wall_visual2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2], rgbaColor=[0.7,0.7,0.7,1])
    p.createMultiBody(0, wall_collision2, wall_visual2, basePosition=[arena_size/2, 0, wall_height/2])
    p.createMultiBody(0, wall_collision2, wall_visual2, basePosition=[-arena_size/2, 0, wall_height/2])

make_arena()
p.setAdditionalSearchPath('shapes/')
mountain = p.loadURDF("gaussian_pyramid.urdf", (0, 0, MOUNTAIN_BASE_Z), p.getQuaternionFromEuler((0,0,0)), useFixedBase=1)

# 加载生物 URDF
xml_file = 'temp_view_best.urdf'
with open(xml_file, 'w') as f:
    f.write(cr.to_xml())
cid = p.loadURDF(xml_file, physicsClientId=pid)
p.resetBasePositionAndOrientation(cid, [-8, -8, 1], [0, 0, 0, 1], physicsClientId=pid)

print(f"Creature has {p.getNumJoints(cid, physicsClientId=pid)} joints")
print("Starting simulation — best fitness was 0.000054 (0.05mm from peak!)")
print("Press Ctrl+C to stop, or close window.")

# 仿真循环
step = 0
while p.isConnected(pid):
    p.stepSimulation(physicsClientId=pid)
    if step % 24 == 0:
        for jid in range(p.getNumJoints(cid, physicsClientId=pid)):
            m = cr.get_motors()[jid]
            p.setJointMotorControl2(cid, jid, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=m.get_output(), force=5, physicsClientId=pid)
    pos, _ = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
    cr.update_position(pos)
    if step % 240 == 0:
        dist = cr.get_min_dist_to_target()
        x, y, z = pos
        print(f"  step={step:5d} | pos=({x:.2f},{y:.2f},{z:.2f}) | dist_to_peak={dist:.5f}")
    step += 1
    time.sleep(1/480.)
