"""
SAC Training Script for RL Drone Obstacle Avoidance
Uses Stable Baselines3 with 12 parallel environments on CPU
"""

import os
import glob
import re
import sys
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gymenv import GymEnv

# OPTIMIZATION: Lock threads to prevent "core clashing" on CPU
# IMPORTANT: These settings are for SINGLE MACHINE with multiple parallel envs
# For 64-core Xeon, allow more threads per environment for better CPU utilization

import multiprocessing

# Calculate optimal thread count per environment
# Formula: total_cores / (n_envs * safety_factor)
TOTAL_CORES = multiprocessing.cpu_count()
N_ENVS = 4  # Will be defined later, but we need it here

# On 64-core system with 30 envs: 64 / (30 * 1.2) = ~1.7 → use 2 threads/env
# On 16-core system with 6 envs: 16 / (6 * 1.2) = ~2.2 → use 2 threads/env
THREADS_PER_ENV = max(1, min(4, TOTAL_CORES // (N_ENVS * 2)))

os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_ENV)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_ENV)
torch.set_num_threads(THREADS_PER_ENV)

print(f"🔧 CPU Configuration:")
print(f"   Total Cores: {TOTAL_CORES}")
print(f"   Parallel Envs: {N_ENVS}")
print(f"   Threads per Env: {THREADS_PER_ENV}")
print(f"   Expected Core Usage: ~{N_ENVS * THREADS_PER_ENV}/{TOTAL_CORES}")


def make_env(goal_positions, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    Args:
        goal_positions: List of goal positions to randomly sample from
        rank: Index of the subprocess
        seed: Random seed
    """
    def _init():
        # Randomly select a goal for this episode
        goal_idx = np.random.randint(0, len(goal_positions))
        goal = goal_positions[goal_idx]
        
        env = GymEnv(
            goal_position=goal,
            goal_tolerance=0.2,
            flight_dome_size=100.0,
            agent_hz=30,
            render_mode=None  # No rendering during training
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def get_latest_checkpoint(log_dir):
    """Finds checkpoint with HIGHEST STEP COUNT where BOTH model and normalizer files exist."""
    zip_files = glob.glob(os.path.join(log_dir, "drone_sac_*.zip"))
    if not zip_files:
        print("❌ No .zip checkpoint files found.")
        return None
    
    valid_checkpoints = []
    for zip_path in zip_files:
        # Extract steps from: drone_sac_500000_steps.zip
        match = re.search(r'drone_sac_(\d+)_steps\.zip$', os.path.basename(zip_path))
        if not match:
            continue
            
        steps = int(match.group(1))
        # Corresponding normalizer file: drone_sac_vecnormalize_500000_steps.pkl
        pkl_path = os.path.join(log_dir, f"drone_sac_vecnormalize_{steps}_steps.pkl")
        
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
    """Main training function."""
    
    # =========================
    # CONFIGURATION
    # =========================
    
    # Define 10 diverse goal positions across the warehouse
    # Spread across different x and y coordinates for better generalization
    GOAL_POSITIONS = [
        [8.0, 5.0, 1.0],    # Near warehouse entrance
        [12.0, 5.0, 1.0],   # Middle-front area
        [15.0, 5.0, 1.0],   # Right-front area
        [19.0, 5.0, 1.0],   # Far right-front
        [3.6, 14.0, 1.0],   # Left-back area
        [7.6, 14.0, 1.0],   # Middle-back area
        [18.0, 14.0, 1.0],  # Right-back area
        [-10.0, 10.0, 1.0], # Outside left area
        [-17.0, -10.0, 1.0],# Far outside left-front
        [10.0, -22.0, 1.0], # Outside bottom area
    ]
    
    print(f"Training with {len(GOAL_POSITIONS)} diverse goal positions:")
    for i, goal in enumerate(GOAL_POSITIONS):
        print(f"  Goal {i}: [{goal[0]:.1f}, {goal[1]:.2f}, {goal[2]:.1f}]")
    
    LOG_DIR = "./logs/drone_sac_multi_goal/"
    TENSORBOARD_LOG = "./logs/tensorboard/"
    N_ENVS = 6  # Number of parallel environments
    TOTAL_STEPS = 10_000_000  # 2 million timesteps (1M base + 1M additional)
    
    # SAC hyperparameters
    LEARNING_RATE = 3e-4
    BUFFER_SIZE = 300000  # Replay buffer size
    LEARNING_STARTS = 10000  # Start training after this many steps
    BATCH_SIZE = 256
    TAU = 0.005  # Soft update coefficient
    GAMMA = 0.99  # Discount factor
    TRAIN_FREQ = 1  # Update the model every step
    GRADIENT_STEPS = 1  # Gradient steps per update
    
    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    
    print("=" * 60)
    print("🚁 RL DRONE OBSTACLE AVOIDANCE TRAINING")
    print("=" * 60)
    print(f"Algorithm: SAC (Soft Actor-Critic)")
    print(f"Goal Position: {GOAL_POSITIONS}")
    print(f"Parallel Environments: {N_ENVS}")
    print(f"Total Training Steps: {TOTAL_STEPS:,}")
    print(f"Device: CPU (optimized for MLP policy)")
    print("=" * 60)
    
    # =========================
    # CREATE VECTORIZED ENVIRONMENT
    # =========================
    print(f"\\n🚀 Initializing {N_ENVS} parallel environments...")
    
    # Create parallel environments with random goal selection
    env_fns = [make_env(GOAL_POSITIONS, i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    
    # =========================
    # AUTO-RESUME LOGIC
    # =========================
    latest = get_latest_checkpoint(LOG_DIR)
    
    if latest:
        model_path, stats_path = latest
        print(f"\\n🔄 Loading checkpoint...")
        
        # Load VecNormalize stats FIRST (critical order!)
        vec_env = VecNormalize.load(stats_path, vec_env)
        vec_env.training = True  # Enable training mode
        vec_env.norm_reward = True
        print(f"   ✅ VecNormalize loaded: {stats_path}")
        
        # Load SAC model
        model = SAC.load(
            model_path,
            env=vec_env,
            device="cpu",
            tensorboard_log=TENSORBOARD_LOG
        )
        current_steps = model.num_timesteps
        remaining_steps = max(0, TOTAL_STEPS - current_steps)
        print(f"✅ RESUME SUCCESS: {current_steps:,} steps → {remaining_steps:,} remaining")
        
    else:
        print("\\n🆕 Starting FRESH training...")
        
        # Wrap with VecNormalize for observation and reward normalization
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        
        # Create SAC model
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
        current_steps = 0
        
        print(f"✅ SAC model initialized with MlpPolicy")
    
    # =========================
    # CALLBACKS
    # =========================
    
    # Checkpoint callback - saves every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),  # Frequency per environment
        save_path=LOG_DIR,
        name_prefix="drone_sac",
        save_vecnormalize=True,
        verbose=1
    )
    
    # Combine callbacks
    callbacks = [checkpoint_callback]
    
    # =========================
    # TRAINING
    # =========================
    print(f"\\n🎯 Training {remaining_steps:,} timesteps...")
    print(f"📊 TensorBoard logs: {TENSORBOARD_LOG}")
    print(f"💾 Checkpoints: {LOG_DIR}")
    print(f"\\nMonitor training with: tensorboard --logdir={TENSORBOARD_LOG}")
    print("=" * 60)
    
    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callbacks,
            reset_num_timesteps=False,  # Continue counting from checkpoint
            progress_bar=True,
            tb_log_name="SAC_DroneObstacleAvoidance"
        )
        
        # Final save
        final_model_path = os.path.join(LOG_DIR, "final_drone_sac_model")
        final_stats_path = os.path.join(LOG_DIR, "final_vec_normalize_stats.pkl")
        
        model.save(final_model_path)
        vec_env.save(final_stats_path)
        
        print(f"\\n" + "=" * 60)
        print(f"✅ TRAINING COMPLETE!")
        print(f"💾 Final model saved: {final_model_path}.zip")
        print(f"📊 Final stats saved: {final_stats_path}")
        print(f"🎯 Total timesteps: {model.num_timesteps:,}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print(f"\\n⚠️  Training interrupted by user!")
        print(f"💾 Latest checkpoint saved in: {LOG_DIR}")
    
    finally:
        # Clean up
        vec_env.close()


if __name__ == '__main__':
    train()
