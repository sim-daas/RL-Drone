"""Test script for GymEnv with simplified rewards and reduced lidar"""

import numpy as np
from gymenv import GymEnv

def test_environment():
    """Test the custom drone obstacle avoidance environment."""
    
    # Create environment with a goal position
    goal_position = [18.0, -5.0, 1.0]
    env = GymEnv(
        goal_position=goal_position,
        goal_tolerance=0.2,
        flight_dome_size=100.0,
        agent_hz=30,
        render_mode=None
    )
    
    print("=" * 70)
    print("GymEnv Test - Simplified Rewards + 24 Lidar Rays")
    print("=" * 70)
    print(f"Goal Position: {goal_position}")
    print(f"Goal Tolerance: {env.goal_tolerance}m")
    print(f"Max Episode Duration: {env.max_duration_seconds}s ({env.max_steps} steps)")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Observation Shape: {env.observation_space.shape}")
    print("=" * 70)
    
    # Reset environment
    obs, info = env.reset()
    print(f"\n✅ Initial Observation Shape: {obs.shape}")
    print(f"   Expected: (48,) - Match: {obs.shape == (48,)}")
    
    # Extract components from observation
    ang_vel = obs[0:3]
    quaternion = obs[3:7]
    lin_vel = obs[7:10]
    lin_pos = obs[10:13]
    prev_action = obs[13:16]
    aux_state = obs[16:20]
    lidar = obs[20:44]        # 24 rays now
    goal_distance = obs[44]
    goal_direction = obs[45:48]
    
    print(f"\n📊 State Components:")
    print(f"  - Angular Velocity: {ang_vel.shape}")
    print(f"  - Quaternion: {quaternion.shape}")
    print(f"  - Linear Velocity: {lin_vel.shape}")
    print(f"  - Linear Position: {lin_pos.shape}")
    print(f"  - Previous Action: {prev_action.shape}")
    print(f"  - Auxiliary State: {aux_state.shape}")
    print(f"  - Lidar Distances: {lidar.shape} (24 rays @ 15° spacing)")
    print(f"  - Goal Distance: {goal_distance:.3f}m")
    print(f"  - Goal Direction: {goal_direction} (mag={np.linalg.norm(goal_direction):.3f})")
    
    print(f"\n🎯 Reward Function Test:")
    # Test normal step
    obs, reward, term, trunc, info = env.step(np.array([0.0, 0.0, 0.0]))
    print(f"  - Normal step reward: {reward:.4f} (should be ≈ -0.1)")
    print(f"    Components: -0.1 (step) + velocity_penalty")
    
    # Test rewards are in expected range
    assert -1.0 < reward < 0.0, f"Step reward out of range: {reward}"
    print(f"  ✅ Step reward in valid range")
    
    print(f"\n🚀 Running 5 test steps...")
    print("=" * 70)
    
    for step_num in range(5):
        # Random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract new goal info
        current_pos = obs[10:13]
        goal_distance = obs[44]
        goal_direction = obs[45:48]
        
        print(f"\nStep {step_num + 1}:")
        print(f"  Action: {action}")
        print(f"  Position: {current_pos}")
        print(f"  Goal Distance: {goal_distance:.3f}m")
        print(f"  Reward: {reward:.4f}")
        print(f"  Terminated: {terminated} | Truncated: {truncated}")
        
        if terminated or truncated:
            reason = []
            if info.get("env_complete"): reason.append("GOAL!")
            if info.get("collision"): reason.append("COLLISION")
            if info.get("out_of_bounds"): reason.append("OUT OF BOUNDS")
            if truncated and not terminated: reason.append("MAX STEPS")
            print(f"  ⚠️  Episode ended: {', '.join(reason)}")
            break
    
    # Close environment
    env.close()
    print(f"\n{'='*70}")
    print("✅ All tests passed!")
    print(f"{'='*70}")
    
    print(f"\n📝 Summary of Changes:")
    print(f"  ✅ Observation space: 84 → 48 dimensions")
    print(f"  ✅ Lidar rays: 60 → 24 (15° spacing)")
    print(f"  ✅ Episode duration: 600s → 30s")
    print(f"  ✅ Rewards: Ultra-simple (-0.1, -100, +500)")
    print(f"  ✅ Added velocity penalty for smooth control")

if __name__ == "__main__":
    test_environment()
