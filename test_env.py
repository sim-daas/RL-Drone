"""Test script for GymEnv with improved goal representation"""

import numpy as np
from gymenv import GymEnv

def test_environment():
    """Test the custom drone obstacle avoidance environment."""
    
    # Create environment with a goal position
    goal_position = [5.0, 0.0, 2.0]
    env = GymEnv(
        goal_position=goal_position,
        goal_tolerance=0.2,
        flight_dome_size=100.0,
        agent_hz=30,
        render_mode=None
    )
    
    print("=" * 60)
    print("GymEnv Obstacle Avoidance Test (Option 1 Implementation)")
    print("=" * 60)
    print(f"Goal Position: {goal_position}")
    print(f"Goal Tolerance: {env.goal_tolerance}m")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Observation Shape: {env.observation_space.shape}")
    print("=" * 60)
    
    # Reset environment
    obs, info = env.reset()
    print(f"\nInitial Observation Shape: {obs.shape}")
    print(f"Expected Shape: (84,)")
    print(f"Match: {obs.shape == (84,)}")
    
    # Extract components from observation
    ang_vel = obs[0:3]
    quaternion = obs[3:7]
    lin_vel = obs[7:10]
    lin_pos = obs[10:13]
    prev_action = obs[13:16]
    aux_state = obs[16:20]
    lidar = obs[20:80]
    goal_distance = obs[80]       # NEW: Scalar distance
    goal_direction = obs[81:84]   # NEW: Normalized direction
    
    print(f"\nState Components:")
    print(f"  - Angular Velocity: {ang_vel.shape} = {ang_vel}")
    print(f"  - Quaternion: {quaternion.shape} = {quaternion}")
    print(f"  - Linear Velocity: {lin_vel.shape} = {lin_vel}")
    print(f"  - Linear Position: {lin_pos.shape} = {lin_pos}")
    print(f"  - Previous Action: {prev_action.shape} = {prev_action}")
    print(f"  - Auxiliary State: {aux_state.shape} = {aux_state}")
    print(f"  - Lidar Distances: {lidar.shape} (min: {lidar.min():.2f}, max: {lidar.max():.2f})")
    
    print(f"\n🎯 NEW GOAL REPRESENTATION:")
    print(f"  - Goal Distance: {goal_distance:.3f} meters")
    print(f"  - Goal Direction (normalized): {goal_direction}")
    print(f"  - Direction magnitude: {np.linalg.norm(goal_direction):.6f} (should be ~1.0)")
    
    # Verify direction
    actual_vector = np.array(goal_position) - lin_pos
    actual_distance = np.linalg.norm(actual_vector)
    actual_direction = actual_vector / actual_distance if actual_distance > 0 else np.zeros(3)
    
    print(f"\n✅ VERIFICATION:")
    print(f"  Calculated distance: {actual_distance:.3f}m")
    print(f"  Observation distance: {goal_distance:.3f}m")
    print(f"  Distance match: {np.isclose(actual_distance, goal_distance)}")
    print(f"  Direction match: {np.allclose(actual_direction, goal_direction)}")
    
    # Run a few steps with random actions
    print(f"\n{'='*60}")
    print("Running 5 test steps...")
    print(f"{'='*60}")
    
    for step_num in range(5):
        # Random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract new goal info
        current_pos = obs[10:13]
        goal_distance = obs[80]
        goal_direction = obs[81:84]
        
        print(f"\nStep {step_num + 1}:")
        print(f"  Action: {action}")
        print(f"  Current Position: {current_pos}")
        print(f"  Goal Distance: {goal_distance:.3f}m")
        print(f"  Goal Direction: {goal_direction} (mag={np.linalg.norm(goal_direction):.3f})")
        print(f"  Reward: {reward:.4f}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        
        if terminated or truncated:
            print(f"\n  Episode ended!")
            break
    
    # Close environment
    env.close()
    print(f"\n{'='*60}")
    print("✅ Test completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_environment()
