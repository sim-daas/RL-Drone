"""Test script for multi-goal training setup"""

import numpy as np
from gymenv import GymEnv

def test_multi_goal():
    """Test the multi-goal reward system."""
    
    # Define 10 goals
    y_values = np.linspace(-5.0, 5.0, 10)
    GOAL_POSITIONS = [[18.0, y, 1.0] for y in y_values]
    
    print("=" * 70)
    print("Multi-Goal Training Test")
    print("=" * 70)
    print(f"\n10 Goal Positions:")
    for i, goal in enumerate(GOAL_POSITIONS):
        print(f"  Goal {i}: x={goal[0]:.1f}, y={goal[1]:.2f}, z={goal[2]:.1f}")
    
    print(f"\n🎯 Testing Improved Reward Function:")
    print("-" * 70)
    
    # Test with random goal
    goal = GOAL_POSITIONS[np.random.randint(0, 10)]
    print(f"\nSelected goal: {goal}")
    
    env = GymEnv(
        goal_position=goal,
        goal_tolerance=0.2,
        flight_dome_size=100.0,
        agent_hz=30,
        render_mode=None
    )
    
    obs, info = env.reset()
    start_distance = obs[45]
    
    print(f"\nReward Logic:")
    print(f"  When distance < 1.5m:")
    print(f"    - Threshold: 0.006m")
    print(f"    - Progress reward: 30 × delta")
    print(f"    - Retreat penalty: -0.01")
    print(f"  When distance ≥ 1.5m:")
    print(f"    - Threshold: 0.001m")
    print(f"    - Progress reward: 20 × delta")
    print(f"    - Retreat penalty: -0.1")
    
    print(f"\nStart distance: {start_distance:.3f}m")
    print(f"Active threshold: 0.001m (far from goal)")
    print(f"Active reward scale: 20x")
    
    print(f"\n🚀 Running 5 test steps:")
    print("=" * 70)
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        
        dist = obs[45]
        pos = obs[10:13]
        
        if reward > 1.0:
            reward_type = "✅ PROGRESS"
        elif reward < -50:
            reward_type = "💥 TERMINAL"
        elif -0.02 < reward < 0:
            reward_type = "⚠️ SMALL PENALTY"
        else:
            reward_type = "⚠️ PENALTY"
        
        print(f"\nStep {step+1}:")
        print(f"  Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        print(f"  Distance: {dist:.3f}m")
        print(f"  Reward: {reward:+.4f} {reward_type}")
        
        if term or trunc:
            print(f"  Episode ended")
            break
    
    env.close()
    
    print(f"\n{'='*70}")
    print("✅ Multi-goal setup ready!")
    print("=" * 70)
    print(f"\nTraining Configuration:")
    print(f"  • 10 goals with y ∈ [-5.0, 5.0]")
    print(f"  • Random goal per episode")
    print(f"  • Improved distance-dependent rewards")
    print(f"  • Continue from existing checkpoint")
    print(f"\nExpected: Better near-goal precision + multi-goal generalization")

if __name__ == "__main__":
    test_multi_goal()
