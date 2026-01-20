import os
import glob
import re
import torch
import gymnasium as gym
import PyFlyt.gym_envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# 1. OPTIMIZATION: Lock threads to prevent "core clashing" 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

def get_latest_checkpoint(log_dir):
    """Finds checkpoint with HIGHEST STEP COUNT where BOTH files exist."""
    zip_files = glob.glob(os.path.join(log_dir, "safe_move_agent_*.zip"))
    if not zip_files:
        print("❌ No .zip checkpoint files found.")
        return None
    
    valid_checkpoints = []
    for zip_path in zip_files:
        # Extract steps from: safe_move_agent_749880_steps.zip
        match = re.search(r'safe_move_agent_(\d+)_steps\.zip$', os.path.basename(zip_path))
        if not match:
            continue
            
        steps = int(match.group(1))
        # CORRECT PKL FORMAT: safe_move_agent_vecnormalize_749880_steps.pkl
        pkl_path = os.path.join(log_dir, f"safe_move_agent_vecnormalize_{steps}_steps.pkl")
        
        if os.path.exists(pkl_path):
            valid_checkpoints.append((zip_path, pkl_path, steps))
        else:
            print(f"⚠️  Missing PKL: {pkl_path}")
    
    if not valid_checkpoints:
        print("❌ No matching zip/pkl pairs found.")
        return None
    
    # Sort by step count (highest first)
    valid_checkpoints.sort(key=lambda x: x[2], reverse=True)
    latest_zip, latest_pkl, latest_steps = valid_checkpoints[0]
    
    print(f"✅ Found {len(valid_checkpoints)} valid pairs. Using:")
    print(f"   📦 Model: {os.path.basename(latest_zip)} ({latest_steps:,} steps)")
    print(f"   📊 Stats: {os.path.basename(latest_pkl)}")
    return latest_zip, latest_pkl

def train():
    ENV_ID = "PyFlyt/QuadX-Hover-v4"
    LOG_DIR = "./logs/safe_move_v4/"
    N_ENVS = 12
    TOTAL_STEPS = 1_000_000
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"🚀 Initializing {N_ENVS} parallel environments...")
    vec_env = make_vec_env(ENV_ID, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    # AUTO-RESUME
    latest = get_latest_checkpoint(LOG_DIR)
    
    if latest:
        model_path, stats_path = latest
        print(f"\n🔄 Loading checkpoint...")
        
        # Load VecNormalize stats FIRST (critical order!)
        vec_env = VecNormalize.load(stats_path, vec_env)
        print(f"   ✅ VecNormalize loaded: {stats_path}")
        
        # Load PPO model
        model = PPO.load(model_path, env=vec_env, device="cpu")
        current_steps = model.num_timesteps
        remaining_steps = max(0, TOTAL_STEPS - current_steps)
        print(f"✅ RESUME SUCCESS: {current_steps:,} steps → {remaining_steps:,} remaining")
        
    else:
        print("\n🆕 Starting FRESH training...")
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        model = PPO(
            "MlpPolicy", vec_env, device="cpu", verbose=1,
            n_steps=2048, batch_size=256, n_epochs=10
        )
        remaining_steps = TOTAL_STEPS
        current_steps = 0

    # Checkpoint callback (continues from current point)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=LOG_DIR,
        name_prefix="safe_move_agent",
        save_vecnormalize=True
    )

    # Train remaining steps
    print(f"\n🎯 Training {remaining_steps:,} timesteps...")
    model.learn(
        total_timesteps=remaining_steps,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
        progress_bar=True
    )

    # Final save
    model.save(f"{LOG_DIR}final_safe_move_v4")
    vec_env.save(f"{LOG_DIR}final_vec_normalize_stats.pkl")
    print(f"\n💾 COMPLETE: {model.num_timesteps:,} total steps")

if __name__ == '__main__':
    train()
