from __future__ import annotations

from typing import Any, Literal

import gymnasium
import numpy as np
import pybullet as p
from gymnasium import spaces
from base import Env

from PyFlyt.core.aviary import Aviary
from PyFlyt.core.utils.compile_helpers import check_numpy


class GymEnv(gymnasium.Env):
    '''Gymnasium environment class for custom drone with obstacle avoidance.
    
    Args:
        goal_position (np.ndarray | list): Target position [x, y, z] in meters (required)
        goal_tolerance (float): Radius of success sphere around goal (default: 0.2m)
        flight_mode (int): Flight mode (default: 6 for velocity control)
        flight_dome_size (float): Maximum distance from origin before termination
        agent_hz (int): Agent control frequency in Hz
        render_mode (None | Literal["human", "rgb_array"]): Rendering mode
    '''
    def __init__(
        self,
        goal_position: np.ndarray | list,
        goal_positions: list | None = None,
        goal_tolerance: float = 0.2,
        flight_mode: int = 6,
        flight_dome_size: float = 10.0,
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        minvelocity: float = 0.2,
        rl: bool = True,
        track: bool = False,
    ):
 
        """ ENVIRONMENT CONSTANTS """
        self.goal_position = np.array(goal_position, dtype=np.float64)
        self.goal_positions = goal_positions  # List of goals for per-episode randomization
        self.goal_tolerance = goal_tolerance
        self.render_mode = render_mode
        self.render_resolution = (480, 480)
        self.max_duration_seconds = 30.0  # 30 seconds for navigation task
        self.attitude_space = 13
        self.flight_mode = flight_mode
        self.flight_dome_size = flight_dome_size
        self.max_steps = int(agent_hz * self.max_duration_seconds)
        self.env_step_ratio = int(120 / agent_hz)
        self.angle_representation = 1
        self.previous_distance = None  # For tracking progress reward
        self.min_velocity = minvelocity
        self.agent_hz = agent_hz
        self.rl = rl
        self.max_altitude = 2.5  # Maximum altitude constraint (meters)
        self.track = track

        """GYMNASIUM STUFF"""
        self.attitude_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.attitude_space,), dtype=np.float64
        )
        self.auxiliary_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
        )

        # define the action space for mode 6 (full ground velocity + yaw rate + vertical velocity)
        # Action: [vx, vy, vr, vz]
        velocity_limit = 5.0
        yaw_rate_limit = 2.0  # rad/s
        high = np.array([velocity_limit, velocity_limit, yaw_rate_limit, velocity_limit])
        low = np.array([-velocity_limit, -velocity_limit, -yaw_rate_limit, -velocity_limit])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # the whole implicit state space = attitude + previous action + auxiliary information
        self.combined_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.attitude_space.shape[0]
                + self.action_space.shape[0]
                + self.auxiliary_space.shape[0],
            ),
            dtype=np.float64,
        )
        
        # Update observation space to include lidar (24) + goal info + yaw
        # Total: 13 (attitude) + 4 (action) + 4 (aux) + 24 (lidar) + 1 (distance) + 3 (direction) + 1 (yaw) = 50
        total_obs_dim = (
            self.attitude_space.shape[0] +  # 13
            self.action_space.shape[0] +     # 4 (changed from 3)
            self.auxiliary_space.shape[0] +  # 4
            24 +  # lidar readings
            1 +   # goal distance
            3 +   # normalized goal direction
            1     # current yaw angle
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float64,
        )

    def close(self) -> None:
        """Disconnects the internal Aviary."""
        if hasattr(self, "env"):
            self.env.disconnect()

    def reset(
        self, *, seed: None | int = None, options: dict[str, Any] | None = dict()
    ) -> tuple[Any, dict[str, Any]]:
        """reset.

        Args:
            seed (None | int): seed
            options (dict[str, Any]): options

        Returns:
            tuple[Any, dict[str, Any]]:

        """
        # Randomize goal each episode if goal_positions provided
        if self.goal_positions is not None and len(self.goal_positions) > 0:
            idx = np.random.randint(0, len(self.goal_positions))
            self.goal_position = np.array(self.goal_positions[idx], dtype=np.float64)
        
        self.begin_reset(seed=seed, options=options)
        self.end_reset(seed=seed, options=options)
        
        return self.state, self.info

    def begin_reset(
        self,
        seed: None | int = None,
        options: None | dict[str, Any] = dict(),
        drone_options: None | dict[str, Any] = dict(),
    ) -> None:
        """The first half of the reset function."""
        super().reset(seed=seed)

        # if we already have an env, disconnect from it
        if hasattr(self, "env"):
            self.env.disconnect()

        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.state = None
        self.action = np.zeros((4,))
        self.reward = 0.0
        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["collision"] = False
        self.info["env_complete"] = False

        # need to handle Nones
        if options is None:
            options = dict()
        if drone_options is None:
            drone_options = dict()

        # camera handling
        drone_options["use_camera"] = drone_options.get("use_camera", False) or bool(
            self.render_mode
        )
        drone_options["camera_fps"] = int(120 / self.env_step_ratio)

        # init env
        self.env = Env(rl=self.rl, track=self.track)
        self.env.loadURDF("models/cylinder.urdf", basePosition=self.goal_position, globalScaling=1, useFixedBase=True)

        if self.render_mode == "human":
            self.camera_parameters = self.env.getDebugVisualizerCamera()

    def end_reset(
        self, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> None:
        """The tailing half of the reset function."""
        # register all new collision bodies
        self.env.register_all_new_bodies()

        # set flight mode
        self.env.set_mode(self.flight_mode)

        # wait for env to stabilize - REDUCED from 10 to 3 steps
        # Physics usually stabilizes faster than we think
        for _ in range(3):
            self.env.step()

        self.compute_state()
        
        # Initialize distance tracking for progress reward
        current_position = self.env.state(0)[-1]
        self.previous_distance = np.linalg.norm(self.goal_position - current_position)
        self.initial_distance = self.previous_distance
        self.max_steps = int(self.initial_distance * 1.8) * self.agent_hz

    def compute_state(self) -> None:
        """Computes the state of the QuadX.
        
        Combines:
        - Default attitude state (ang_vel, quaternion, lin_vel, lin_pos)
        - Previous action
        - Auxiliary state
        - Lidar readings (24 values)
        - Goal distance (1 value: scalar distance to goal)
        - Normalized goal direction (3 values: unit vector pointing to goal)
        - Current yaw angle (1 value: heading in radians)
        
        Total observation: 50 dimensions
        """
        # Get default state components
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = self.compute_attitude()
        aux_state = self.compute_auxiliary()
        
        # Get lidar readings
        lidar_data = self.env.get_lidar_reading(visualize=False)
        
        # Handle case where lidar isn't initialized yet
        if lidar_data is None or 'distances' not in lidar_data:
            # Use max distance (10.0) for all 24 readings as default
            lidar_distances = np.full(24, 10.0, dtype=np.float64)
        else:
            lidar_distances = np.array(lidar_data['distances'], dtype=np.float64)
        
        # Compute goal information: distance + normalized direction
        goal_vector = self.goal_position - lin_pos
        goal_distance = np.linalg.norm(goal_vector)
        
        # Normalize direction vector (avoid division by zero)
        if goal_distance > 1e-8:
            goal_direction = goal_vector / goal_distance
        else:
            goal_direction = np.zeros(3, dtype=np.float64)
        
        # Get current yaw angle from euler angles
        current_yaw = ang_pos[2]  # Yaw is the z-axis rotation in euler angles
        
        # Combine all components using quaternion representation
        self.state = np.concatenate([
            ang_vel,                    # 3
            quaternion,                 # 4
            lin_vel,                    # 3
            lin_pos,                    # 3
            self.action,                # 4 (changed from 3)
            aux_state,                  # 4
            lidar_distances,            # 24
            np.array([goal_distance]),  # 1 (scalar distance)
            goal_direction,             # 3 (normalized direction)
            np.array([current_yaw]),    # 1 (current heading)
        ], axis=-1)  # Total: 50
        return self.state

    def compute_auxiliary(self) -> np.ndarray:
        """This returns the auxiliary state form the drone."""
        return self.env.aux_state(0)

    def compute_attitude(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """state.

        This returns the base attitude for the drone.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - quaternion (vector of 4 values)
        """
        raw_state = self.env.state(0)

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # quaternion angles
        quaternion = p.getQuaternionFromEuler(ang_pos)

        return ang_vel, ang_pos, lin_vel, lin_pos, quaternion

    def compute_term_trunc_reward(self) -> None:
        # Check base termination conditions first (collision, out of bounds, max steps)
        self.compute_base_term_trunc_reward()
        
        # Calculate progress toward goal
        current_position = self.env.state(0)[-1]
        current_distance = np.linalg.norm(self.goal_position - current_position)
        
        # Check for goal success (overrides progress reward)
        if not self.termination and not self.truncation:
            if current_distance <= self.goal_tolerance:
                self.reward = 25 * self.initial_distance  # Big positive reward for success
                self.info["env_complete"] = True
                self.termination = True
                return
        
        # Progress-based reward (only if not terminated)
        if not self.termination and not self.truncation:
            delta = self.previous_distance - current_distance
            THRESHOLD = 0.006  # 6mm threshold for meaningful movement
            
            if delta > THRESHOLD:
                # Moving toward goal - GOOD!
                self.reward = 20.0 * delta
            elif delta <= -THRESHOLD:
                # Moving away from goal - BAD!
                self.reward = -0.1
            else:
                # Hovering or minimal movement - BAD!
                self.reward = -0.1
            
            # Update tracking for next step
            self.previous_distance = current_distance

    def compute_base_term_trunc_reward(self) -> None:
        """Check base termination conditions with fixed penalties."""
        # Exceed step count
        if self.step_count > self.max_steps:
            self.truncation |= True

        # Collision with any object
        if np.any(self.env.contact_array):
            self.reward = -10 * self.initial_distance  # Fixed penalty
            self.info["collision"] = True
            self.termination |= True

        # Exceed flight dome
        if np.linalg.norm(self.env.state(0)[-1]) > self.flight_dome_size:
            self.reward = -10 * self.initial_distance  # Fixed penalty
            self.info["out_of_bounds"] = True
            self.termination |= True
        
        # Exceed altitude ceiling (Z-ceiling constraint)
        current_position = self.env.state(0)[-1]
        if current_position[2] > self.max_altitude:
            self.reward = -10 * self.initial_distance  # Fixed penalty
            self.info["altitude_violation"] = True
            self.termination |= True

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Steps the environment.

        Args:
            action (np.ndarray): action

        Returns:
            state, reward, termination, truncation, info

        """
        # unsqueeze the action to be usable in aviary
        self.action = action.copy()

        # reset the reward and set the action
        self.reward = -0.1
        self.env.set_setpoint(0, action)

        # step through env, the internal env updates a few steps before the outer env
        for _ in range(self.env_step_ratio):
            # if we've already ended, don't continue
            if self.termination or self.truncation:
                break

            self.env.step()

            # compute state and done
            self.compute_state()
            self.compute_term_trunc_reward()

        # increment step count
        self.step_count += 1

        return self.state, self.reward, self.termination, self.truncation, self.info

    def render(self) -> np.ndarray:
        """Render."""
        check_numpy()
        if self.render_mode is None:
            raise ValueError(
                "Please set `render_mode='human'` or `render_mode='rgb_array'` in init to use this function."
            )

        if self.render_mode == "human":
            _, _, rgbaImg, _, _ = self.env.getCameraImage(
                width=self.render_resolution[1],
                height=self.render_resolution[0],
                viewMatrix=self.camera_parameters[2],
                projectionMatrix=self.camera_parameters[3],
            )
        elif self.render_mode == "rgb_array":
            _, _, rgbaImg, _, _ = self.env.getCameraImage(
                width=self.render_resolution[1],
                height=self.render_resolution[0],
                viewMatrix=self.env.drones[0].camera.view_mat,
                projectionMatrix=self.env.drones[0].camera.proj_mat,
            )
        else:
            raise ValueError(
                f"Unknown render mode {self.render_mode}, should not have ended up here"
            )

        rgbaImg = np.asarray(rgbaImg, dtype=np.uint8).reshape(
            self.render_resolution[0], self.render_resolution[1], -1
        )

        return rgbaImg
