from __future__ import annotations

from typing import Any, Literal

import gymnasium
import numpy as np
import pybullet as p
from gymnasium import spaces
from .base import Env
from collections import deque

from PyFlyt.core.aviary import Aviary
from PyFlyt.core.utils.compile_helpers import check_numpy


class GymEnv(gymnasium.Env):
    '''Gymnasium environment class for basic drone hover using rate control (Flight Mode 0).
    
    Flight Mode 0: [vp, vq, vr, T]
    - vp, vq: Roll and pitch rates
    - vr: Yaw rate
    - T: Thrust
    '''
    def __init__(
        self,
        goal_tolerance: float = 0.2,
        flight_mode: int = 0,
        flight_dome_size: float = 10.0,
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        rl: bool = True,
        track: bool = False,
    ):
 
        """ ENVIRONMENT CONSTANTS """
        self.goal_position = np.array([0.0, 0.0, 2.0], dtype=np.float64)
        self.goal_tolerance = goal_tolerance
        self.render_mode = render_mode
        self.render_resolution = (480, 480)
        self.max_duration_seconds = 30.0
        self.flight_mode = flight_mode
        self.flight_dome_size = flight_dome_size
        self.max_steps = int(agent_hz * self.max_duration_seconds)
        self.env_step_ratio = int(120 / agent_hz)
        self.agent_hz = agent_hz
        self.rl = rl
        self.track = track

        # Curriculum and Reward parameters
        self.alpha = 1.0
        self.alpha_final = 10.0  # Steeper gradient at the end
        self.curriculum_steps = 500_000  # Reach max sharpness after this many steps
        self.total_env_steps = 0
        
        self.w_upright = 1.0
        self.w_smoothness = 0.1
        self.w_vel = 1.0  # Base velocity penalty weight
        
        # Integral error buffer
        self.error_window = 50
        self.error_buffer = deque(maxlen=self.error_window)

        """GYMNASIUM STUFF"""
        # Action space remains 4D: [roll_rate, pitch_rate, yaw_rate, thrust]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64)

        # Physical limits for scaling
        self.max_roll_pitch_rate = 50.0 * np.pi / 180.0
        self.max_yaw_rate = 15.0 * np.pi / 180.0
        self.max_thrust = 1.0

        # Observation space: 16 dimensions
        # [pos_error(3), quat(4), lin_vel(3), ang_vel(3), integral_error(3)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float64,
        )

        self.state = np.zeros(16)
        self.prev_action = np.zeros(4)

    def close(self) -> None:
        if hasattr(self, "env"):
            self.env.disconnect()

    def reset(
        self, *, seed: None | int = None, options: dict[str, Any] | None = dict()
    ) -> tuple[Any, dict[str, Any]]:
        self.begin_reset(seed=seed, options=options)
        self.end_reset(seed=seed, options=options)
        
        return self.state, self.info

    def begin_reset(
        self,
        seed: None | int = None,
        options: None | dict[str, Any] = dict(),
        drone_options: None | dict[str, Any] = dict(),
    ) -> None:
        super().reset(seed=seed)

        if hasattr(self, "env"):
            self.env.disconnect()

        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.reward = 0.0
        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["collision"] = False
        self.info["env_complete"] = False
        self.prev_action = np.zeros(4)
        self.error_buffer.clear()

        self.env = Env(rl=self.rl, track=self.track)
        # self.env.loadURDF("models/cylinder.urdf", basePosition=self.goal_position, globalScaling=1, useFixedBase=True)

        if self.render_mode == "human":
            self.camera_parameters = self.env.getDebugVisualizerCamera()

    def end_reset(
        self, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> None:
        self.env.register_all_new_bodies()
        self.env.set_mode(self.flight_mode)

        for _ in range(3):
            self.env.step()

        self.compute_state()

    def compute_state(self) -> None:
        """Computes the state (16 dimensions)."""
        raw_state = self.env.state(0)
        
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]
        
        # Position error
        pos_error = self.goal_position - lin_pos
        
        # Update integral error
        self.error_buffer.append(pos_error)
        integral_error = np.sum(self.error_buffer, axis=0)
        
        # Quaternion
        quaternion = np.array(p.getQuaternionFromEuler(ang_pos))
        
        self.state = np.concatenate([
            pos_error,      # 3
            quaternion,     # 4
            lin_vel,        # 3
            ang_vel,        # 3
            integral_error  # 3
        ], axis=-1)
        
        return self.state

    def compute_reward(self, action: np.ndarray) -> None:
        # Curriculum: Adjust alpha based on total steps
        # Each environment instance will scale its own alpha as it gains experience
        # progress = min(1.0, self.total_env_steps / self.curriculum_steps)
        # alpha = self.alpha_init + progress * (self.alpha_final - self.alpha_init)
        
        # 1. Position Tracking Reward
        pos_error_norm = np.linalg.norm(self.state[0:3])
        # Gaussian reward centered at 0, ranging from 0 (far) to 1 (at goal)
        # Then subtract 1 to get the range [-1, 0]
        reward_pos = np.exp(-self.alpha * (pos_error_norm**2)) - 1.0
        
        # 2. Velocity Reward
        # reward = 0 if velocity is 0, negative if non-zero, range [-1, 0]
        lin_vel_norm = np.linalg.norm(self.state[7:10])
        ang_vel_norm = np.linalg.norm(self.state[10:13])
        
        # Combine linear and angular velocity for the penalty
        # w_vel controls the sensitivity/sharpness of the penalty
        total_vel_sq = lin_vel_norm**2 + 0.1 * ang_vel_norm**2
        reward_vel = np.exp(-self.w_vel * total_vel_sq) - 1.0
        
        # 3. Upright Penalty
        rot_mat = np.array(p.getMatrixFromQuaternion(self.state[3:7])).reshape(3, 3)
        body_z_axis = rot_mat[:, 2]
        world_z_axis = np.array([0, 0, 1])
        
        tilt_cos = np.clip(np.dot(body_z_axis, world_z_axis), -1.0, 1.0)
        tilt_angle = np.arccos(tilt_cos)
        reward_upright = -self.w_upright * (tilt_angle**2)
        
        # 4. Action Rate Penalty (Smoothness)
        reward_smoothness = -self.w_smoothness * np.linalg.norm(action - self.prev_action)**2
        
        self.reward = reward_pos + reward_vel + reward_upright + reward_smoothness
        
        # Store individual components for debugging/logging
        self.info["reward_pos"] = reward_pos
        self.info["reward_vel"] = reward_vel
        self.info["reward_upright"] = reward_upright
        self.info["reward_smoothness"] = reward_smoothness
        
        # Base penalties (collisions, bounds)
        if np.any(self.env.contact_array):
            self.reward -= 100.0
            self.termination = True
            self.info["collision"] = True
            
        if np.linalg.norm(self.env.state(0)[-1]) > self.flight_dome_size:
            self.reward -= 100.0
            self.termination = True
            self.info["out_of_bounds"] = True

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # Scale action from [-1, 1] to physical limits
        # Action: [roll_rate, pitch_rate, yaw_rate, thrust]
        scaled_action = action.copy()
        scaled_action[0] *= self.max_roll_pitch_rate
        scaled_action[1] *= self.max_roll_pitch_rate
        scaled_action[2] *= self.max_yaw_rate
        # Thrust: Scale [-1, 1] to [0, 1]
        scaled_action[3] = (scaled_action[3] + 1.0) / 2.0 * self.max_thrust
        
        # Set setpoint
        self.env.set_setpoint(0, scaled_action)

        for _ in range(self.env_step_ratio):
            if self.termination or self.truncation:
                break
            self.env.step()

        self.compute_state()
        self.compute_reward(action)
        
        self.prev_action = action.copy()
        self.step_count += 1
        self.total_env_steps += 1
        if self.step_count >= self.max_steps:
            self.truncation = True

        return self.state, self.reward, self.termination, self.truncation, self.info

    def render(self) -> np.ndarray:
        check_numpy()
        if self.render_mode is None:
            raise ValueError("Please set `render_mode` in init.")

        if self.render_mode == "human":
            _, _, rgbaImg, _, _ = self.env.getCameraImage(
                width=self.render_resolution[1],
                height=self.render_resolution[0],
                viewMatrix=self.camera_parameters[2],
                projectionMatrix=self.camera_parameters[3],
            )
        else:
            _, _, rgbaImg, _, _ = self.env.getCameraImage(
                width=self.render_resolution[1],
                height=self.render_resolution[0],
                viewMatrix=self.env.drones[0].camera.view_mat,
                projectionMatrix=self.env.drones[0].camera.proj_mat,
            )

        rgbaImg = np.asarray(rgbaImg, dtype=np.uint8).reshape(
            self.render_resolution[0], self.render_resolution[1], -1
        )
        return rgbaImg
