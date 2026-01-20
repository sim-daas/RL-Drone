import os
import numpy as np
from typing import Any

from PyFlyt.core import Aviary
from PyFlyt.core.abstractions import WindFieldClass



class Env(Aviary):
    def __init__(
        self,
        wind_type: None | str | type[WindFieldClass] = None,
        wind_options: dict[str, Any] = {},
        rpm: int = 600,
    ):
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 2.0]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            render=True,
            drone_type="quadx",
            drone_options=[dict(drone_model="primitive_drone")],
        )
        
        self.configureDebugVisualizer(self.COV_ENABLE_GUI, 0)  # Hide GUI panels
        self.configureDebugVisualizer(self.COV_ENABLE_SHADOWS, 0)

        # set the flight mode
        self.set_mode(6)
        self.drone = self.drones[0]
        self.visual_joints = self.find_visual_rotor_joints()
        self.rotor_angles = np.zeros(len(self.visual_joints))
        self.VISUAL_RPM = rpm
        
        self.loadURDF("converted_assets/planer.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/wall.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/column.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/cieling.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/rafter.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/metalstairs.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/black.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/blackpure.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/blank.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/bumper.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/container.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/containerfloor.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/door.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/doorbase.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/glass.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/ramp.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/slidingdoor.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        self.loadURDF("converted_assets/shelves.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.1])
        
        self.register_all_new_bodies()
        
    def start(self, steps:int=20000):
        for i in range(steps):
            self.step()
            throttle = self.drone.motors.get_states()  # (4,) array
            self.update_rotor_angles(throttle, self.visual_joints, self.drone, self, self.VISUAL_RPM)
            
        self.disconnect()
    
    def stop(self):
        self.disconnect()
     
    def find_visual_rotor_joints(self):
        visual_joints = []
        for i in range(self.drone.p.getNumJoints(self.drone.Id)):
            joint_info = self.drone.p.getJointInfo(self.drone.Id, i)
            if b"visual_rotor" in joint_info[1]: 
                visual_joints.append(i)
        return visual_joints

    def update_rotor_angles(self, throttle, visual_joints, drone, env, VISUAL_RPM):
        for j, joint_idx in enumerate(self.visual_joints):
            rpm = throttle[j] * self.VISUAL_RPM
            angular_vel = (rpm / 60.0) * 2.0 * np.pi
            self.rotor_angles[j] += angular_vel * self.step_period
            self.drone.p.resetJointState(self.drone.Id, joint_idx, self.rotor_angles[j])


if __name__ == "__main__":
    env = Env(rpm=600)
    env.start(steps=20000)