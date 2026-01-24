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
    ):
        super().__init__(
            start_pos=np.array([[-10, 10.0, 1.0]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            render=not rl,  # Disable rendering during RL training for speed
            drone_type="quadx",
            drone_options=[dict(drone_model="primitive_drone")],
            wind_type=wind_type,
            wind_options=wind_options,
        )
        
        self.configureDebugVisualizer(self.COV_ENABLE_GUI, 0)  # Hide GUI panels
        self.configureDebugVisualizer(self.COV_ENABLE_SHADOWS, 0)

        # set the flight mode
        self.set_mode(6)
        self.drone = self.drones[0]
        self.visual_joints = self.find_visual_rotor_joints()
        self.rotor_angles = np.zeros(len(self.visual_joints))
        self.VISUAL_RPM = rpm
        self.rl = rl
        
        self.warehouse_id = self.loadURDF("converted_assets/planer.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
        self.loadURDF("converted_assets/shelves.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
        if not rl:    
            self.loadURDF("converted_assets/wall.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/cieling.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/column.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/rafter.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/metalstairs.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/black.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/blackpure.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/blank.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/bumper.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/container.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/containerfloor.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/door.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/doorbase.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/glass.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/ramp.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
            self.loadURDF("converted_assets/slidingdoor.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
        
        self.register_all_new_bodies()
        
        # Initialize LiDAR parameters
        self.laser_range = 10.0  # Maximum range in meters
        self.laser_link_idx = self.find_laser_link()
        self.num_rays = 24  # 24 rays = 15° spacing for 360° coverage
        self.lidar_line_ids = [None] * self.num_rays  # Store line IDs for each ray
        
    def start(self, steps:int=20000):
        for i in range(steps):
            self.step()
        self.disconnect()
        
    def step(self):
        super().step()
        if not self.rl:
            self.update_rotor_angles()
            if not self.rl:
                lidar_data = self.get_lidar_reading(visualize=False)
    
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

    def find_laser_link(self):
        """Find the laser sensor link index"""
        for i in range(self.drone.p.getNumJoints(self.drone.Id)):
            joint_info = self.drone.p.getJointInfo(self.drone.Id, i)
            if b"laser_sensor" in joint_info[12]:  # Link name
                return i
        return -1
    
    def get_lidar_reading(self, visualize=True):
        """Perform 360° raycast from laser sensor with multiple rays"""
        if self.laser_link_idx < 0:
            return None
            
        # Get laser link world position and orientation
        link_state = self.drone.p.getLinkState(self.drone.Id, self.laser_link_idx)
        laser_pos = link_state[0]
        laser_orn = link_state[1]
        
        # Convert quaternion to rotation matrix
        rot_matrix = self.drone.p.getMatrixFromQuaternion(laser_orn)
        
        distances = []
        
        # Create rays in 360° around the sensor
        # Starting from forward (0°) and going clockwise
        angle_increment = 2 * np.pi / self.num_rays
        
        for i in range(self.num_rays):
            # Clockwise: negate angle
            angle = -i * angle_increment
            
            # Calculate direction vector (rotating around Z-axis in sensor's local frame)
            # Forward direction in local frame (X-axis forward)
            local_forward = np.array([np.cos(angle), np.sin(angle), 0])
            
            # Transform to world frame
            forward = [
                rot_matrix[0] * local_forward[0] + rot_matrix[1] * local_forward[1] + rot_matrix[2] * local_forward[2],
                rot_matrix[3] * local_forward[0] + rot_matrix[4] * local_forward[1] + rot_matrix[5] * local_forward[2],
                rot_matrix[6] * local_forward[0] + rot_matrix[7] * local_forward[1] + rot_matrix[8] * local_forward[2]
            ]
            
            # Calculate ray end point
            ray_end = [
                laser_pos[0] + forward[0] * self.laser_range,
                laser_pos[1] + forward[1] * self.laser_range,
                laser_pos[2] + forward[2] * self.laser_range
            ]
            
            # Perform raycast
            ray_result = self.drone.p.rayTest(laser_pos, ray_end)[0]
            
            distance = ray_result[2] * self.laser_range
            distances.append(distance)
            
            # Visualize the laser ray
            if visualize:
                hit_pos = ray_result[3] if ray_result[0] >= 0 else ray_end
                line_color = [1, 0, 0] if ray_result[0] >= 0 else [0, 1, 0]  # Red if hit, green if no hit
                
                if self.lidar_line_ids[i] is not None:
                    self.lidar_line_ids[i] = self.drone.p.addUserDebugLine(
                        laser_pos, hit_pos, line_color, lineWidth=1, 
                        replaceItemUniqueId=self.lidar_line_ids[i]
                    )
                else:
                    self.lidar_line_ids[i] = self.drone.p.addUserDebugLine(
                        laser_pos, hit_pos, line_color, lineWidth=1
                    )
        
        return {
            'num_rays': self.num_rays,
            'distances': distances  # Starting from forward (0°), clockwise
        }

if __name__ == "__main__":
    env = Env(rpm=600)
    env.start(steps=20000)
