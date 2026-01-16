"""Implements a simple time invariant, stateless wind model."""

import numpy as np

from PyFlyt.core import Aviary


drone_options = []
drone_options.append(dict(drone_model="primitive_drone", control_hz=240))

# the starting position and orientations
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# environment setup, attach the windfield
env = Aviary(start_pos=start_pos, start_orn=start_orn, render=True, drone_type="quadx", drone_options=drone_options)

# set the flight mode
env.set_mode(6)

drone = env.drones[0]

# Find visual rotor joints
visual_joints = []
for i in range(drone.p.getNumJoints(drone.Id)):
    joint_info = drone.p. getJointInfo(drone.Id, i)
    if b"visual_rotor" in joint_info[1]: 
        visual_joints.append(i)

rotor_angles = np.zeros(len(visual_joints))
VISUAL_RPM = 600  # Adjust for desired visual speed

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(20000):
    env.step()
    
    throttle = drone.motors.get_states()  # (4,) array
    
    # Manually rotate each visual rotor
    for j, joint_idx in enumerate(visual_joints):
        # RPM based on throttle
        rpm = throttle[j] * VISUAL_RPM
        angular_vel = (rpm / 60.0) * 2.0 * np.pi
        rotor_angles[j] += angular_vel * env.step_period
        
        # Update joint position
        drone.p. resetJointState(drone.Id, joint_idx, rotor_angles[j])


env.disconnect()