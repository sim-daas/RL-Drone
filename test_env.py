"""Test Phase 2 Progress-Based Rewards"""

import numpy as np
from gymenv import GymEnv

def test_phase2_rewards():
    """Test progress-based reward system."""
    
    goal_position = [18.0, -2.0, 1.5]
    env = GymEnv(
        goal_position=goal_position,
        goal_tolerance=0.2,
        flight_dome_size=100.0,
        agent_hz=30,
        render_mode=None
    )
    
    print("=" * 70)
    print("Phase 2 Test - Progress-Based Rewards")
    print("=" * 70)
    print(f"Goal: {goal_position}")
    print(f"\nReward Structure:")
    print(f"  • Moving toward goal (delta > 0.006m): +20 × delta")
    print(f"  • Moving away (delta < -0.006m): -0.1")
    print(f"  • Hovering (|delta| ≤ 0.006m): -0.1")
    print(f"  • Goal reached: +500")
    print(f"  • Collision/OOB: -100")
    print("=" * 70)
    
    obs, info = env.reset()
    start_pos = obs[10:13]
    start_distance = obs[45]
    
    print(f"\nStart position: {start_pos}")
    print(f"Start distance: {start_distance:.3f}m")
    
    print(f"\n🧪 Testing Reward Scenarios:")
    print("-" * 70)
    
    # Scenario 1: Hover (no movement)
    print(f"\n1. Hover Test:")
    action = np.array([0.0, 0.0, 0.0, 0.0])  # No movement
    obs, reward, term, trunc, info = env.step(action)
    print(f"   Action: [0, 0, 0, 0] (hover)")
    print(f"   Reward: {reward:.4f} (expected: -0.1)")
    
    # Scenario 2: Move toward goal
    print(f"\n2. Move Toward Goal Test:")
    # Move in positive X direction (toward goal)
    action = np.array([2.0, 0.0, 0.0, 0.0])
    obs, reward, term, trunc, info = env.step(action)
    new_distance = obs[45]
    delta = start_distance - new_distance
    print(f"   Action: [2.0, 0, 0, 0] (toward goal)")
    print(f"   Distance change: {delta:.6f}m")
    print(f"   Reward: {reward:.4f} (expected: ≈{20*delta:.4f})")
    
    # Reset for clean test
    env.reset()
    
    print(f"\n🚀 Running 10 Steps with Random Actions:")
    print("=" * 70)
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        
        pos = obs[10:13]
        dist = obs[45]
        
        # Determine reward type
        if reward > 1.0:
            reward_type = "✅ PROGRESS"
        elif reward < -50:
            reward_type = "💥 TERMINAL"
        else:
            reward_type = "⚠️ PENALTY"
        
        print(f"\nStep {step+1}:")
        print(f"  Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        print(f"  Distance: {dist:.3f}m")
        print(f"  Reward: {reward:+.4f} {reward_type}")
        
        if term or trunc:
            reason = []
            if info.get("env_complete"): reason.append("GOAL!")
            if info.get("collision"): reason.append("COLLISION")
            if info.get("altitude_violation"): reason.append("ALTITUDE")
            if info.get("out_of_bounds"): reason.append("OOB")
            if trunc and not term: reason.append("TIMEOUT")
            print(f"  Episode ended: {', '.join(reason)}")
            break
    
    env.close()
    
    print(f"\n{'='*70}")
    print("✅ Phase 2 Reward System Ready!")
    print("=" * 70)
    print(f"\nExpected Training Improvements:")
    print(f"  → Agent will prefer moving toward goal (+20×delta)")
    print(f"  → Hovering penalized (-0.1) → encourages action")
    print(f"  → Moving away penalized (-0.1) → stays focused")
    print(f"\nNext: Delete old checkpoints and start fresh training!")

if __name__ == "__main__":
    test_phase2_rewards()
