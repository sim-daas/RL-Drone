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
        rl: bool = False,
        track: bool = False,
    ):
        super().__init__(
            start_pos=np.array([[0, 0.0, 2.0]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            render=not rl,  # Disable rendering during RL training for speed
            drone_type="quadx",
            drone_options=[dict(drone_model="primitive_drone")],
            wind_type=wind_type,
            wind_options=wind_options,
        )
        
        self.configureDebugVisualizer(self.COV_ENABLE_GUI, 0)  # Hide GUI panels
        self.configureDebugVisualizer(self.COV_ENABLE_SHADOWS, 1 if not rl else 0)

        # set the flight mode - default to 0 for rate control task
        self.set_mode(0)
        self.drone = self.drones[0]
        self.visual_joints = self.find_visual_rotor_joints()
        self.rotor_angles = np.zeros(len(self.visual_joints))
        self.VISUAL_RPM = rpm
        self.rl = rl
        self.track = track
        
        # Camera tracking parameters
        self.camera_step_counter = 0
        self.camera_update_interval = 1  # Update every 6 steps for 30fps at 240Hz
        self.camera_pos = np.array([0.0, 0.0, 0.0])  # Smoothed camera position
        self.camera_target = np.array([0.0, 0.0, 0.0])  # Smoothed look-at target
        self.camera_smoothing = 0.1  # Lower = smoother, higher = more responsive
        
        # # Load the warehouse environment
        # self.warehouse_id = self.loadURDF("converted_assets/planer.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        # self.darken_object(self.warehouse_id, brightness_factor=0.4)
        # shelves_id = self.loadURDF("converted_assets/shelves.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
        # self.darken_object(shelves_id, brightness_factor=0.4)
        # wall_id = self.loadURDF("converted_assets/wall.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        # self.darken_object(wall_id, brightness_factor=0.4)
        
        # if not rl:    
        #     ceiling_id = self.loadURDF("converted_assets/cieling.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        #     # Make ceiling semi-transparent and darker
        #     for i in range(-1, self.getNumJoints(ceiling_id)):
        #         self.changeVisualShape(ceiling_id, i, rgbaColor=[0.4, 0.4, 0.4, 0.95])
            
        #     column_id = self.loadURDF("converted_assets/column.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        #     self.darken_object(column_id, brightness_factor=0.4)
        #     rafter_id = self.loadURDF("converted_assets/rafter.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        #     for i in range(-1, self.getNumJoints(rafter_id)):
        #         self.changeVisualShape(rafter_id, i, rgbaColor=[0.4, 0.4, 0.4, 0.95])
            
        #     stairs_id = self.loadURDF("converted_assets/metalstairs.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        #     self.darken_object(stairs_id, brightness_factor=0.4)
            
        #     black_id = self.loadURDF("converted_assets/black.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        #     blackpure_id = self.loadURDF("converted_assets/blackpure.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        #     blank_id = self.loadURDF("converted_assets/blank.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            
        #     bumper_id = self.loadURDF("converted_assets/bumper.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        #     self.darken_object(bumper_id, brightness_factor=0.4)
            
        #     container_id = self.loadURDF("converted_assets/container.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        #     self.darken_object(container_id, brightness_factor=0.4)
            
        #     containerfloor_id = self.loadURDF("converted_assets/containerfloor.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        #     self.darken_object(containerfloor_id, brightness_factor=0.4)
            
        #     door_id = self.loadURDF("converted_assets/door.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        #     self.darken_object(door_id, brightness_factor=0.4)
            
        #     doorbase_id = self.loadURDF("converted_assets/doorbase.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        #     self.darken_object(doorbase_id, brightness_factor=0.4)
            
        #     glass_id = self.loadURDF("converted_assets/glass.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            
        #     ramp_id = self.loadURDF("converted_assets/ramp.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        #     self.darken_object(ramp_id, brightness_factor=0.4)
            
        #     slidingdoor_id = self.loadURDF("converted_assets/slidingdoor.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        #     self.darken_object(slidingdoor_id, brightness_factor=0.4)
        
        # self.register_all_new_bodies()
        
        # Initialize camera position if tracking is enabled
        if self.track:
            drone_pos, drone_orn = self.drone.p.getBasePositionAndOrientation(self.drone.Id)
            self.camera_pos = np.array(drone_pos) + np.array([-1.0, 0.0, 1.0])
            self.camera_target = np.array(drone_pos)
    
    def darken_object(self, object_id, brightness_factor=0.6):
        """
        Darken an object by reducing its color brightness and reflectivity
        brightness_factor: 0.0 (black) to 1.0 (original brightness)
        """
        num_joints = self.getNumJoints(object_id)
        for link_idx in range(-1, num_joints):
            try:
                self.changeVisualShape(
                    object_id, 
                    link_idx, 
                    specularColor=[0.1, 0.1, 0.1],
                    rgbaColor=[brightness_factor, brightness_factor, brightness_factor, 1.0]
                )
            except:
                pass
        
    def start(self, steps:int=20000):
        for i in range(steps):
            self.step()
        self.disconnect()
        
    def step(self):
        super().step()
        if not self.rl:
            self.update_rotor_angles()
            
            # Update camera tracking
            # if self.track:
            #     self.camera_step_counter += 1
            #     if self.camera_step_counter >= self.camera_update_interval:
            #         self.update_camera()
            #         self.camera_step_counter = 0
    
    def stop(self):
        self.disconnect()
      
    def update_rotor_angles(self):
        self.throttle = self.drone.motors.get_states()
        for j, joint_idx in enumerate(self.visual_joints):
            rpm = self.throttle[j] * self.VISUAL_RPM
            angular_vel = (rpm / 60.0) * 2.0 * np.pi
            self.rotor_angles[j] += angular_vel * self.step_period
            self.drone.p.resetJointState(self.drone.Id, joint_idx, self.rotor_angles[j])

    def find_visual_rotor_joints(self):
        visual_joints = []
        for i in range(self.drone.p.getNumJoints(self.drone.Id)):
            joint_info = self.drone.p.getJointInfo(self.drone.Id, i)
            if b"visual_rotor" in joint_info[1]: 
                visual_joints.append(i)
        return visual_joints
    
    def update_camera(self):
        """Update camera position to track the drone with smooth interpolation"""
        drone_pos, drone_orn = self.drone.p.getBasePositionAndOrientation(self.drone.Id)
        drone_pos = np.array(drone_pos)
        drone_euler = self.drone.p.getEulerFromQuaternion(drone_orn)
        yaw = drone_euler[2]
        
        local_offset = np.array([0.0, 4, -0.5])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        world_offset = np.array([
            local_offset[0] * cos_yaw - local_offset[1] * sin_yaw,
            local_offset[0] * sin_yaw + local_offset[1] * cos_yaw,
            local_offset[2]
        ])
        
        target_cam_pos = drone_pos + world_offset
        self.camera_pos = self.camera_pos + self.camera_smoothing * (target_cam_pos - self.camera_pos)
        
        look_ahead_distance = 3.0
        target_look_at = drone_pos + np.array([
            look_ahead_distance * cos_yaw,
            look_ahead_distance * sin_yaw,
            drone_pos[2]
        ])
        self.camera_target = self.camera_target + self.camera_smoothing * (target_look_at - self.camera_target)
        
        cam_to_target = self.camera_target - self.camera_pos
        distance = np.linalg.norm(cam_to_target)
        
        if distance > 0.001:
            cam_yaw = np.degrees(np.arctan2(cam_to_target[1], cam_to_target[0]))
            horizontal_dist = np.sqrt(cam_to_target[0]**2 + cam_to_target[1]**2)
            cam_pitch = -np.degrees(np.arctan2(cam_to_target[2], horizontal_dist))
            
            self.drone.p.resetDebugVisualizerCamera(
                cameraDistance=distance,
                cameraYaw=cam_yaw,
                cameraPitch=cam_pitch,
                cameraTargetPosition=self.camera_target.tolist()
            )

if __name__ == "__main__":
    env = Env(rpm=600)
    env.start(steps=20000)
