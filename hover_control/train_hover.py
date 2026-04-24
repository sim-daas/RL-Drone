"""
SAC Training Script for Drone Hover Control (Flight Mode 0)
Uses Stable Baselines3 with parallel environments.
"""

import os
import glob
import re
import sys
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hover_control.gymenv import GymEnv

# Lock to 1 thread per env to prevent core clashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = GymEnv(
            goal_tolerance=0.2,
            flight_dome_size=10.0,
            agent_hz=30,
            render_mode=None
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def get_latest_checkpoint(log_dir):
    """Finds checkpoint with HIGHEST STEP COUNT where BOTH model and normalizer files exist."""
    zip_files = glob.glob(os.path.join(log_dir, "hover_sac_*.zip"))
    if not zip_files:
        return None
    
    valid_checkpoints = []
    for zip_path in zip_files:
        match = re.search(r'hover_sac_(\d+)_steps\.zip$', os.path.basename(zip_path))
        if not match:
            continue
            
        steps = int(match.group(1))
        pkl_path = os.path.join(log_dir, f"hover_sac_vecnormalize_{steps}_steps.pkl")
        
        if os.path.exists(pkl_path):
            valid_checkpoints.append((zip_path, pkl_path, steps))
    
    if not valid_checkpoints:
        return None
    
    valid_checkpoints.sort(key=lambda x: x[2], reverse=True)
    return valid_checkpoints[0][0], valid_checkpoints[0][1]


def train():
    """Main training function."""
    
    # =========================
    # CONFIGURATION
    # =========================
    
    LOG_DIR = "./logs/hover_sac_v1/"
    TENSORBOARD_LOG = "./logs/tensorboard_hover/"
    N_ENVS = 50
    TOTAL_STEPS = 4_000_000
    
    # SAC hyperparameters
    LEARNING_RATE = 3e-4
    BUFFER_SIZE = 1000000
    LEARNING_STARTS = 10000
    BATCH_SIZE = 256
    TAU = 0.005
    GAMMA = 0.99
    TRAIN_FREQ = 1
    GRADIENT_STEPS = 1

    
    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    
    print("=" * 60)
    print("🚁 DRONE HOVER CONTROL TRAINING (FLIGHT MODE 0)")
    print("=" * 60)
    print(f"Parallel Environments: {N_ENVS}")
    print(f"Total Training Steps: {TOTAL_STEPS:,}")
    print("=" * 60)
    
    # =========================
    # CREATE VECTORIZED ENVIRONMENT
    # =========================
    env_fns = [make_env(i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    
    # =========================
    # AUTO-RESUME LOGIC
    # =========================
    latest = get_latest_checkpoint(LOG_DIR)
    
    if latest:
        model_path, stats_path = latest
        print(f"\n🔄 Resuming from {model_path}...")
        vec_env = VecNormalize.load(stats_path, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        
        model = SAC.load(
            model_path,
            env=vec_env,
            device="cpu",
            tensorboard_log=TENSORBOARD_LOG
        )
        remaining_steps = max(0, TOTAL_STEPS - model.num_timesteps)
    else:
        print("\n🆕 Starting FRESH training...")
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            tau=TAU,
            gamma=GAMMA,
            train_freq=TRAIN_FREQ,
            gradient_steps=GRADIENT_STEPS,
            device="cpu",
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG
        )
        remaining_steps = TOTAL_STEPS
    
    # =========================
    # CALLBACKS
    # =========================
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=LOG_DIR,
        name_prefix="hover_sac",
        save_vecnormalize=True,
        verbose=1
    )
    
    # =========================
    # TRAINING
    # =========================
    print(f"\n🎯 Training {remaining_steps:,} timesteps...")
    print(f"📊 TensorBoard logs: {TENSORBOARD_LOG}")
    print(f"💾 Checkpoints: {LOG_DIR}")
    print(f"\nMonitor training with: tensorboard --logdir={TENSORBOARD_LOG}")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=[checkpoint_callback],
            reset_num_timesteps=False,
            progress_bar=True,
            tb_log_name="SAC_HoverControl"
        )
        
        # Final save
        model.save(os.path.join(LOG_DIR, "final_hover_sac_model"))
        vec_env.save(os.path.join(LOG_DIR, "final_vec_normalize_stats.pkl"))
        print("\n✅ TRAINING COMPLETE!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted!")
    finally:
        vec_env.close()


if __name__ == '__main__':
    train()
