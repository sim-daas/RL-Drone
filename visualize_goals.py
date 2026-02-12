"""Spawn goal cylinders in the warehouse to visually pick good positions."""

import numpy as np
from base import Env
import time

# ============================
# EDIT THESE GOAL POSITIONS
# ============================
GOALS = [
    [8.0, -0.5, 1.0],     # ~10m forward
    # [5.0, 5.0, 1.0],     # ~9m forward-right
    # [0.0, 8.0, 1.0],     # ~10m right
    # [-5.0, 5.0, 1.0],    # ~9m back-right
    [-8.0, -0.5, 1.0],    # ~10m behind
    [-5.0, -5.0, 1.0],   # ~8m back-left
    [5.0, -5.0, 1.0],    # ~8m forward-left
    [0.0, -8.0, 1.0],    # ~10m left
]

env = Env(rl=False, track=False)

# Spawn cylinders at each goal
for i, goal in enumerate(GOALS):
    cyl_id = env.loadURDF("models/cylinder.urdf", basePosition=goal, globalScaling=1, useFixedBase=True)
    
    # Add 3D text label above each cylinder
    dist = np.linalg.norm(np.array(goal) - np.array([0, -2, 1]))
    env.addUserDebugText(
        f"G{i}: [{goal[0]}, {goal[1]}] d={dist:.1f}m",
        [goal[0], goal[1], goal[2] + 1.5],
        textColorRGB=[1, 1, 0],
        textSize=1.2
    )

env.register_all_new_bodies()

print(f"\nSpawned {len(GOALS)} goal cylinders. Drone starts at [0, -2, 1]")
print("Use mouse to navigate the warehouse and check positions.")
print("Press Ctrl+C to exit.\n")

for i, g in enumerate(GOALS):
    d = np.linalg.norm(np.array(g) - np.array([0, -2, 1]))
    print(f"  Goal {i}: [{g[0]:6.1f}, {g[1]:6.1f}, {g[2]:.1f}]  dist={d:.1f}m")

# Keep alive for visual inspection
try:
    while True:
        env.step()
        time.sleep(1/60)
except KeyboardInterrupt:
    print("\nDone!")
    env.disconnect()
