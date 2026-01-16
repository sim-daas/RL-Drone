import gymnasium as gym
import PyFlyt.gym_envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import os
import numpy as np

# Force CPU (fixes GPU warning)
import torch
torch.set_num_threads(1)

# 1. Create single render env
eval_env = gym.make("PyFlyt/QuadX-Hover-v4", render_mode="human")
eval_env = DummyVecEnv([lambda: eval_env])

LOG_DIR = "./logs/safe_move_v4/"

# 2. Load FINAL training files
final_model = f"{LOG_DIR}final_safe_move_v4.zip"
final_stats = f"{LOG_DIR}final_vec_normalize_stats.pkl"

print("📁 Checking final training files...")
print(f"Model: {os.path.exists(final_model)}")
print(f"Stats: {os.path.exists(final_stats)}")

# Load VecNormalize FIRST
if os.path.exists(final_stats):
    eval_env = VecNormalize.load(final_stats, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    print("✅ VecNormalize loaded")

# Load model (CPU)
model = PPO.load(final_model, device="cpu")
print(f"✅ Model loaded: {final_model}")

# 3. EVALUATION LOOP (CORRECT VecEnv API)
obs = eval_env.reset()
total_reward = 0
episode_count = 0
max_steps = 2000

print("\n🎬 Starting evaluation (deterministic)...")

for step in range(max_steps):
    action, _ = model.predict(obs, deterministic=True)
    
    # VecEnv.step() returns: obs, rewards, dones, infos (4 values)
    obs, rewards, dones, infos = eval_env.step(action)
    
    # Extract scalar values (VecEnv returns arrays)
    reward = rewards[0].item()
    done = dones[0]
    total_reward += reward
    
    # Render
    eval_env.render()
    
    # Episode ended
    if done:
        print(f"Episode {episode_count+1}: {total_reward:.1f} reward ({step+1} steps)")
        total_reward = 0
        episode_count += 1
        obs = eval_env.reset()

if total_reward > 0:
    print(f"Final episode: {total_reward:.1f} reward ({max_steps} steps)")

print(f"\n🏁 Complete: {episode_count} episodes")
eval_env.close()
