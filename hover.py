"""Implements a simple time invariant, stateless wind model."""

import numpy as np

from PyFlyt.core import Aviary


def find_visual_rotor_joints(drone):
    visual_joints = []
    for i in range(drone.p.getNumJoints(drone.Id)):
        joint_info = drone.p.getJointInfo(drone.Id, i)
        if b"visual_rotor" in joint_info[1]: 
            visual_joints.append(i)
    return visual_joints

def update_rotor_angles(rotor_angles, throttle, visual_joints, drone, env, VISUAL_RPM):
    for j, joint_idx in enumerate(visual_joints):
        rpm = throttle[j] * VISUAL_RPM
        angular_vel = (rpm / 60.0) * 2.0 * np.pi
        rotor_angles[j] += angular_vel * env.step_period
        drone.p.resetJointState(drone.Id, joint_idx, rotor_angles[j])

drone_options = []
drone_options.append(dict(drone_model="primitive_drone", control_hz=240))

# the starting position and orientations
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# environment setup, attach the windfield
env = Aviary(start_pos=start_pos, start_orn=start_orn, render=True, drone_type="quadx", drone_options=drone_options)
env.configureDebugVisualizer(env.COV_ENABLE_GUI, 0)  # Hide GUI panels

# set the flight mode
env.set_mode(7)
drone = env.drones[0]
visual_joints = find_visual_rotor_joints(drone)
rotor_angles = np.zeros(len(visual_joints))
VISUAL_RPM = 600  # Adjust for desired visual speed





obstacle_id = env.loadURDF(
    "models/jetty.urdf",  # Built-in PyBullet URDF
    basePosition=[2.0, 0.0, 1.0],
#    useFixedBase=True,  # MUST be False for dynamics!
)
env.changeDynamics(
    obstacle_id,
    -1,
    mass=0,  # Zero mass = kinematic
    linearDamping=0,
    angularDamping=0
)


env.register_all_new_bodies()
setpoint = np.array([2.0, 0.0, 0.0, 1.0])
env.set_setpoint(0, setpoint)


for i in range(20000):
    t = i * 0.01
    
    # Define trajectory (circular path)
    radius = 2.0
    height = 1.0
    angular_speed = 0
    x = radius * np.cos(angular_speed * t)
    y = radius * np.sin(angular_speed * t)
    z= height
    # Set position directly
    env.resetBasePositionAndOrientation(
        obstacle_id,
        [x, y, z],
        env.getQuaternionFromEuler([0, 0, angular_speed * t])  # Also rotate
    )

    env.step()
    throttle = drone.motors.get_states()  # (4,) array
    update_rotor_angles(rotor_angles, throttle, visual_joints, drone, env, VISUAL_RPM)

env.close()