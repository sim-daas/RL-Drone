import gymnasium as gym
import PyFlyt.gym_envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecVideoRecorder
import torch
import os

# 1. Setup Environment with rgb_array
# This mode is mandatory for capturing frames
VIDEO_DIR = "./videos/safe_move_eval/"
os.makedirs(VIDEO_DIR, exist_ok=True)

raw_env = gym.make("PyFlyt/QuadX-Hover-v4", render_mode="human")
eval_env = DummyVecEnv([lambda: raw_env])

# 2. Wrap with VecVideoRecorder
# This captures the rgb_array frames into a video file
eval_env = VecVideoRecorder(
    eval_env, 
    VIDEO_DIR, 
    record_video_trigger=lambda x: x == 0, # Start recording at step 0
    video_length=1000,                      # Record for 1000 steps
    name_prefix="safe-move-agent"
)

# 3. Load Stats and Model
LOG_DIR = "./logs/safe_move_v4/"
final_model = f"{LOG_DIR}final_safe_move_v4.zip"
final_stats = f"{LOG_DIR}final_vec_normalize_stats.pkl"

# Apply VecNormalize stats to ensure correct observation scaling
if os.path.exists(final_stats):
    eval_env = VecNormalize.load(final_stats, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    print("✅ VecNormalize stats loaded")

model = PPO.load(final_model, device="cpu")
print(f"✅ Model loaded: {final_model}")

# 4. RUN EVALUATION
obs = eval_env.reset()
print("🎬 Recording video... (Simulation running in background)")

for step in range(1001):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = eval_env.step(action)
    
    # Note: No need for manual render() call here; 
    # VecVideoRecorder handles it automatically

# Close and save the video file
eval_env.close()
print(f"🏁 Video saved to {VIDEO_DIR}")
