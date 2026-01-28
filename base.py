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
            start_pos=np.array([[0, -2.0, 1.0]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            render=not rl,  # Disable rendering during RL training for speed
            drone_type="quadx",
            drone_options=[dict(drone_model="primitive_drone")],
            wind_type=wind_type,
            wind_options=wind_options,
        )
        
        self.configureDebugVisualizer(self.COV_ENABLE_GUI, 0)  # Hide GUI panels
        self.configureDebugVisualizer(self.COV_ENABLE_SHADOWS, 1 if not rl else 0)

        # set the flight mode
        self.set_mode(6)
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
        self.camera_smoothing = 0.01  # Lower = smoother, higher = more responsive
        
        self.warehouse_id = self.loadURDF("converted_assets/planer.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
        # Darken the warehouse floor and environment
        self.darken_object(self.warehouse_id, brightness_factor=0.4)
        
        shelves_id = self.loadURDF("converted_assets/shelves.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
        self.darken_object(shelves_id, brightness_factor=0.4)
        
#         shelf_id = self.loadURDF("converted_assets/shelf.urdf", useFixedBase=True, globalScaling=1, basePosition=[0,0,0.01])
#         self.darken_object(shelf_id, brightness_factor=0.4)
        
#         pillar1_id = self.loadURDF("models/pillar.urdf", useFixedBase=True, globalScaling=1, basePosition=[6,-3,0.01])
#         self.darken_object(pillar1_id, brightness_factor=0.4)
        
#         pillar2_id = self.loadURDF("models/pillar.urdf", useFixedBase=True, globalScaling=1, basePosition=[3,-6,0.01])
#         self.darken_object(pillar2_id, brightness_factor=0.4)
# #        self.loadURDF("models/sqpillar.urdf", useFixedBase=True, globalScaling=1, basePosition=[3,-6,0.01])
        if not rl:    
            wall_id = self.loadURDF("converted_assets/wall.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            self.darken_object(wall_id, brightness_factor=0.4)
            
            ceiling_id = self.loadURDF("converted_assets/cieling.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            # Make ceiling semi-transparent and darker
            for i in range(-1, self.getNumJoints(ceiling_id)):
                self.changeVisualShape(ceiling_id, i, rgbaColor=[0.4, 0.4, 0.4, 0.95])
            
            column_id = self.loadURDF("converted_assets/column.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            self.darken_object(column_id, brightness_factor=0.4)
            # self.loadURDF("converted_assets/rafter.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            
            # stairs_id = self.loadURDF("converted_assets/metalstairs.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            # self.darken_object(stairs_id, brightness_factor=0.4)
            
            # black_id = self.loadURDF("converted_assets/black.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            # # self.darken_object(black_id, brightness_factor=0.4)
            
            # blackpure_id = self.loadURDF("converted_assets/blackpure.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            # # self.darken_object(blackpure_id, brightness_factor=0.4)
            
            # blank_id = self.loadURDF("converted_assets/blank.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            # # self.darken_object(blank_id, brightness_factor=0.4)
            
            # bumper_id = self.loadURDF("converted_assets/bumper.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            # self.darken_object(bumper_id, brightness_factor=0.4)
            
            # container_id = self.loadURDF("converted_assets/container.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            # self.darken_object(container_id, brightness_factor=0.4)
            
            # containerfloor_id = self.loadURDF("converted_assets/containerfloor.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            # self.darken_object(containerfloor_id, brightness_factor=0.4)
            
            
            # door_id = self.loadURDF("converted_assets/door.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            # self.darken_object(door_id, brightness_factor=0.4)
            
            # doorbase_id = self.loadURDF("converted_assets/doorbase.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            # self.darken_object(doorbase_id, brightness_factor=0.4)
            
            # glass_id = self.loadURDF("converted_assets/glass.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            # # self.darken_object(glass_id, brightness_factor=0.4)
            
            # ramp_id = self.loadURDF("converted_assets/ramp.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            # self.darken_object(ramp_id, brightness_factor=0.4)
            
            # slidingdoor_id = self.loadURDF("converted_assets/slidingdoor.urdf", useFixedBase=True, globalScaling=0.8, basePosition=[0,0,0.01])
            # self.darken_object(slidingdoor_id, brightness_factor=0.4)
        
        self.register_all_new_bodies()
        
        # Initialize LiDAR parameters
        self.laser_range = 10.0  # Maximum range in meters
        self.laser_link_idx = self.find_laser_link()
        self.num_rays = 24  # 24 rays = 15° spacing for 360° coverage
        self.lidar_line_ids = [None] * self.num_rays  # Store line IDs for each ray
        
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
        # Iterate through all links including the base (-1)
        num_joints = self.getNumJoints(object_id)
        for link_idx in range(-1, num_joints):
            try:
                # Change visual properties to make it darker
                # Reduce specular color to remove reflections (makes it less bright)
                # Set RGBA to darker values
                self.changeVisualShape(
                    object_id, 
                    link_idx, 
                    specularColor=[0.1, 0.1, 0.1],  # Very low specular = less reflection
                    rgbaColor=[brightness_factor, brightness_factor, brightness_factor, 1.0]  # Darker base color
                )
            except:
                # Some links might not have visual shapes, skip them
                pass
        
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
            
            # Update camera tracking
            if self.track:
                self.camera_step_counter += 1
                if self.camera_step_counter >= self.camera_update_interval:
                    self.update_camera()
                    self.camera_step_counter = 0
    
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
        # Get drone position and orientation
        drone_pos, drone_orn = self.drone.p.getBasePositionAndOrientation(self.drone.Id)
        drone_pos = np.array(drone_pos)
        
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        drone_euler = self.drone.p.getEulerFromQuaternion(drone_orn)
        yaw = drone_euler[2]  # Yaw angle (rotation around Z-axis)
        
        # Calculate camera position offset in drone's local frame
        # Offset: -1 in X (behind drone), 0 in Y (centered), +1 in Z (above drone)
        local_offset = np.array([3.0, 4, -0.5])
        
        # Rotate the X,Y offset by drone's yaw to get world frame offset
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        world_offset = np.array([
            local_offset[0] * cos_yaw - local_offset[1] * sin_yaw,
            local_offset[0] * sin_yaw + local_offset[1] * cos_yaw,
            local_offset[2]  # Z offset stays the same (world frame up)
        ])
        
        # Calculate target camera position (behind and above the drone)
        target_cam_pos = drone_pos + world_offset
        
        # Smooth camera position using exponential smoothing
        self.camera_pos = self.camera_pos + self.camera_smoothing * (target_cam_pos - self.camera_pos)
        
        # Calculate look-at target (point ahead of drone in its facing direction)
        look_ahead_distance = 3.0
        target_look_at = drone_pos + np.array([
            look_ahead_distance * cos_yaw,
            look_ahead_distance * sin_yaw,
            drone_pos[2]  # Keep same height as drone
        ])
        
        # Smooth look-at target
        self.camera_target = self.camera_target + self.camera_smoothing * (target_look_at - self.camera_target)
        
        # PyBullet's resetDebugVisualizerCamera works by positioning the camera
        # at spherical coordinates (distance, yaw, pitch) relative to a target point.
        # We need to set the target to where we want to LOOK AT, and calculate
        # the spherical coordinates from our desired camera position.
        
        # The target is what the camera looks at (point ahead of drone)
        # The camera position is behind and above the drone
        # We need to express camera position in spherical coords relative to target
        
        # Vector from target to camera (reverse of view direction)
        cam_to_target = self.camera_target - self.camera_pos
        distance = np.linalg.norm(cam_to_target)
        
        if distance > 0.001:
            # Calculate yaw: angle in XY plane from target to camera
            cam_yaw = np.degrees(np.arctan2(cam_to_target[1], cam_to_target[0]))
            
            # Calculate pitch: angle from horizontal plane
            horizontal_dist = np.sqrt(cam_to_target[0]**2 + cam_to_target[1]**2)
            cam_pitch = -np.degrees(np.arctan2(cam_to_target[2], horizontal_dist))
            
            # Set the debug camera
            # cameraTargetPosition: where the camera looks at
            # cameraDistance: distance from target to camera
            # cameraYaw/Pitch: spherical angles from target to camera position
            self.drone.p.resetDebugVisualizerCamera(
                cameraDistance=distance,
                cameraYaw=cam_yaw,
                cameraPitch=cam_pitch,
                cameraTargetPosition=self.camera_target.tolist()
            )

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
