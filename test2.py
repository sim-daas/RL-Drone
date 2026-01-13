import gymnasium as gym
import PyFlyt.gym_envs  # Required to register PyFlyt envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import ProgressBarCallback

# 1. Setup vectorized environments for faster training
# Using 8-16 parallel envs is recommended for modern CPUs
env_id = "PyFlyt/QuadX-Hover-v4"
vec_env = make_vec_env(env_id, n_envs=64)

# 2. Add observation/reward normalization
# This is crucial for "Good" performance in flight environments
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

# 3. Instantiate the PPO agent
# We use an MLP policy (Multi-Layer Perceptron) for vector observations
model = PPO(
    "MlpPolicy", 
    vec_env, 
    verbose=1,
    learning_rate=3e-4,  # Standard for PPO/UAV tasks
    n_steps=2048,        # Steps per update per environment
    batch_size=64,       # Mini-batch size
    n_epochs=10,         # Number of epochs per update
    device="auto"
)

# 4. Train the agent
# Hovering is a relatively simple task; 500k to 1M steps is usually sufficient
print("Starting training...")
model.learn(total_timesteps=1_000_000, callback=ProgressBarCallback())

# 5. Save the model and normalization stats
model.save("ppo_quadx_hover_v4")
vec_env.save("vec_normalize_stats.pkl")
print("Training complete and model saved.")
