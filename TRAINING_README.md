# SAC Training for RL Drone Obstacle Avoidance

## Overview

This training setup uses **Soft Actor-Critic (SAC)** to train a drone to navigate from start to goal while avoiding obstacles in a warehouse environment using lidar sensor readings.

---

## Setup Summary

### ✅ Fixed Issues

1. **Rendering Disabled for Training**
   - Modified [`base.py:21`](file:///home/ubuntu/githubrepos/dronerlsim/base.py#L21): `render=not rl` 
   - When `rl=True`, rendering is disabled for faster training
   - When `rl=False`, rendering is enabled for visualization

2. **Obstacles Loaded for RL**
   - Modified [`base.py:37-57`](file:///home/ubuntu/githubrepos/dronerlsim/base.py#L37-L57)
   - Obstacles now load regardless of `rl` mode
   - Ensures collision detection works during training
   - `self.planeId` stored for reference

3. **GymEnv Configuration**
   - [`gymenv.py:157`](file:///home/ubuntu/githubrepos/dronerlsim/gymenv.py#L157): `self.env = Env(rl=True)`
   - Explicitly sets RL mode during training

---

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Algorithm** | SAC | Off-policy, sample efficient |
| **Device** | CPU | Optimized for MLP policies |
| **Parallel Envs** | 12 | CPU-optimized parallelization |
| **Total Steps** | 1,000,000 | Can be adjusted |
| **Goal Position** | [5.0, 0.0, 2.0] | Fixed position |
| **Goal Tolerance** | 0.2m | Success sphere radius |
| **Flight Dome** | 100.0m | Max distance from origin |

### SAC Hyperparameters

```python
LEARNING_RATE = 3e-4
BUFFER_SIZE = 300000
LEARNING_STARTS = 10000
BATCH_SIZE = 256
TAU = 0.005
GAMMA = 0.99
```

---

## File Structure

```
/home/ubuntu/githubrepos/dronerlsim/
├── base.py                  # ✅ Fixed rendering
├── gymenv.py                # Custom gym environment
├── train_sac.py             # 🆕 Training script
├── evaluate_sac.py          # 🆕 Evaluation script
├── logs/
│   ├── drone_sac_obstacle_avoidance/  # Model checkpoints
│   └── tensorboard/                    # TensorBoard logs
└── train/
    └── train.py             # Reference implementation
```

---

## Usage

### 1. **Start Training**

```bash
cd /home/ubuntu/githubrepos/dronerlsim
python train_sac.py
```

**Features:**
- ✅ Auto-resumes from latest checkpoint
- ✅ Saves every 50k steps
- ✅ Progress bar
- ✅ TensorBoard logging
- ✅ VecNormalize for obs/reward normalization

### 2. **Monitor Training (TensorBoard)**

In a separate terminal:

```bash
cd /home/ubuntu/githubrepos/dronerlsim
tensorboard --logdir=./logs/tensorboard/
```

Then open: `http://localhost:6006`

**Metrics to watch:**
- `rollout/ep_rew_mean` - Average episode reward
- `rollout/ep_len_mean` - Average episode length
- `train/actor_loss` - Actor network loss
- `train/critic_loss` - Critic network loss

### 3. **Evaluate Trained Model**

```bash
# Use default final model
python evaluate_sac.py

# Or specify checkpoint
python evaluate_sac.py --model ./logs/drone_sac_obstacle_avoidance/drone_sac_500000_steps.zip \
                       --stats ./logs/drone_sac_obstacle_avoidance/drone_sac_vecnormalize_500000_steps.pkl \
                       --episodes 10

# Custom goal position
python evaluate_sac.py --goal-x 8.0 --goal-y 3.0 --goal-z 2.5 --episodes 5
```

---

## Reward System

Your custom reward function:

### Goal Success
```python
goal_reward = 3 * (((45 * (1/hz) * d²) / min_vel) + (3 * (1/hz) * d) / 2)
```

### Collision / Out of Bounds
```python
reward = -0.5 * goal_reward  # Terminal
```

### Progress Reward
```python
delta = best_distance - current_distance
if delta > 0:
    r = (delta - penalty) * 10 * (initial_distance - current_distance) / initial_distance
    reward = min(r, 0.1)  # Capped
    best_distance = current_distance
else:
    reward = -penalty
```

Where:
- `penalty = min_velocity / agent_hz`
- `initial_distance` = distance from start to goal at reset

---

## Training Process

### Auto-Resume Feature

The training script automatically:
1. Searches for latest checkpoint in `./logs/drone_sac_obstacle_avoidance/`
2. Loads both model (`.zip`) and normalizer (`.pkl`)
3. Continues training from checkpoint
4. Maintains step count across sessions

### Checkpointing

- Saves every **50,000 steps** (distributed across 12 environments)
- Includes both model and VecNormalize stats
- Format: `drone_sac_<steps>_steps.zip` and `drone_sac_vecnormalize_<steps>_steps.pkl`

### CPU Optimization

```python
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
```

Prevents thread contention across parallel environments.

---

## Observation Space (83 dimensions)

| Component | Dimensions | Range |
|-----------|-----------|-------|
| Angular Velocity | 3 | rad/s |
| Quaternion | 4 | normalized |
| Linear Velocity | 3 | m/s |
| Linear Position | 3 | meters |
| Previous Action | 3 | [-5, 5] m/s |
| Auxiliary State | 4 | varies |
| **Lidar Distances** | 60 | [0, 10] meters |
| **Relative Goal** | 3 | meters |

---

## Action Space (3 dimensions)

Velocity setpoints: `[v_x, v_y, v_z]` in range **[-5.0, 5.0] m/s**

---

## Troubleshooting

### Issue: "No module named 'gymenv'"
```bash
# Make sure you're in the correct directory
cd /home/ubuntu/githubrepos/dronerlsim
python train_sac.py
```

### Issue: Rendering still appears during training
- Check [`base.py:21`](file:///home/ubuntu/githubrepos/dronerlsim/base.py#L21) has `render=not rl`
- Check [`gymenv.py:157`](file:///home/ubuntu/githubrepos/dronerlsim/gymenv.py#L157) has `Env(rl=True)`

### Issue: Collision detection not working
- Verify obstacles are loaded in [`base.py:39-57`](file:///home/ubuntu/githubrepos/dronerlsim/base.py#L39-L57)
- Check `self.planeId` is set

### Issue: Training is slow
- Verify CPU optimization settings in `train_sac.py`
- Check number of parallel environments (reduce if memory constrained)
- Monitor system resources: `htop`

---

## Expected Training Time

With 12 parallel environments:
- **50k steps**: ~10-20 minutes
- **500k steps**: ~2-3 hours  
- **1M steps**: ~4-6 hours

*Times vary based on CPU performance*

---

## Next Steps

1. **Run Training**: `python train_sac.py`
2. **Monitor Progress**: Check TensorBoard
3. **Evaluate**: Test checkpoints with `evaluate_sac.py`
4. **Tune Hyperparameters**: Adjust in `train_sac.py` if needed
5. **Curriculum Learning**: Gradually increase goal distance
6. **Randomize Goals**: Modify goal position selection for generalization

---

## Advanced: Curriculum Learning

To implement curriculum learning (gradually increasing difficulty):

```python
# In make_env function, randomize goal distance
def make_env(rank, seed=0):
    def _init():
        # Start with easy goals, increase distance over time
        difficulty_factor = min(1.0, current_steps / 500000)
        max_distance = 2.0 + difficulty_factor * 8.0  # 2m to 10m
        
        # Random goal within sphere
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi/4)  # Keep in upper hemisphere
        r = np.random.uniform(1.0, max_distance)
        
        goal_x = r * np.sin(phi) * np.cos(theta)
        goal_y = r * np.sin(phi) * np.sin(theta)
        goal_z = r * np.cos(phi) + 1.0
        
        env = GymEnv(goal_position=[goal_x, goal_y, goal_z], ...)
        return env
    return _init
```

---

## Contact & Support

For questions about:
- Environment implementation → Check `gymenv.py` and `base.py`
- Training configuration → Check `train_sac.py`
- Reward function → Check reward logic in `gymenv.py:251-287`

Happy Training! 🚁🎯
