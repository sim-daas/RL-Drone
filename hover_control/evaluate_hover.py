"""
Evaluation/Visualization Script for Trained SAC Hover Control Model
Loads the trained model and runs episodes with rendering enabled
"""

import os
import sys
import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hover_control.gymenv import GymEnv


def evaluate_model(model_path, stats_path=None, n_episodes=5, goal_position=[0.0, 0.0, 2.0]):
    """
    Evaluate a trained SAC model.
    """
    
    print("=" * 60)
    print("🎮 EVALUATING TRAINED HOVER MODEL")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Goal Position: {goal_position}")
    print(f"Episodes: {n_episodes}")
    print("=" * 60)
    
    # Create environment with rendering enabled
    def make_eval_env():
        env = GymEnv(
            goal_position=goal_position,
            goal_tolerance=0.2,
            flight_dome_size=10.0,
            agent_hz=30,
            render_mode="human",  # Enable rendering for visualization
            rl=False,
            track=True
        )
        return env
    
    # Wrap in DummyVecEnv for compatibility
    env = DummyVecEnv([make_eval_env])
    
    # Load VecNormalize stats if available
    if stats_path and os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False  # Disable training mode
        env.norm_reward = False  # Don't normalize rewards during evaluation
        print(f"✅ Loaded VecNormalize stats from: {stats_path}")
    else:
        print("⚠️  No VecNormalize stats found, running without normalization")
    
    # Load model
    model = SAC.load(model_path, env=env, device="cpu")
    print(f"✅ Loaded SAC model")
    
    # Run evaluation episodes
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'='*60}")
        
        while not done:
            # Get action from model (deterministic for evaluation)
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
            # Check termination reasons
            if done:
                if info[0].get("env_complete", False):
                    print(f"✅ Goal reached!")
                elif info[0].get("collision", False):
                    print(f"💥 Collision detected!")
                elif info[0].get("out_of_bounds", False):
                    print(f"🚫 Out of bounds!")
                else:
                    print(f"⏱️  Episode truncated (max steps)")
        
        print(f"Episode Reward: {episode_reward:.2f}")
        print(f"Episode Length: {episode_length} steps")
    
    env.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained SAC hover model')
    parser.add_argument('--model', type=str, default='./logs/hover_sac_v1/final_hover_sac_model.zip',
                        help='Path to model checkpoint (.zip)')
    parser.add_argument('--stats', type=str, default='./logs/hover_sac_v1/final_vec_normalize_stats.pkl',
                        help='Path to VecNormalize stats (.pkl)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to evaluate')
    parser.add_argument('--goal-x', type=float, default=0.0,
                        help='Goal X coordinate')
    parser.add_argument('--goal-y', type=float, default=0.0,
                        help='Goal Y coordinate')
    parser.add_argument('--goal-z', type=float, default=2.0,
                        help='Goal Z coordinate')
    
    args = parser.parse_args()
    
    # Handle both .zip and non-zip extensions for model path
    model_path = args.model
    if not model_path.endswith(".zip") and not os.path.exists(model_path):
        if os.path.exists(model_path + ".zip"):
            model_path += ".zip"

    goal_position = [args.goal_x, args.goal_y, args.goal_z]
    
    evaluate_model(
        model_path=model_path,
        stats_path=args.stats,
        n_episodes=args.episodes,
        goal_position=goal_position
    )
