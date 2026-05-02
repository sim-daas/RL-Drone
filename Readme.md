# Drone RL Simulation Suite

A lightweight and high-fidelity suite of Reinforcement Learning environments for autonomous drone operations built on top of the PyFlyt simulation engine. This project features two primary operational tasks trained using Soft Actor-Critic (SAC).

[![Hover Control](https://img.youtube.com/vi/rY_cwEJwi2A/0.jpg)](https://youtu.be/rY_cwEJwi2A)
---

## 🚀 Task 1: Drone Hover Control (Flight Mode 0 & 3)

Hyper-stable static hover control at a fixed coordinate point. This task addresses inner-loop flight stability and the elimination of steady-state drift.

- **V1: Rate Control (Finalized)** 
  - Commands: `[roll_rate, pitch_rate, yaw_rate, thrust]`
  - Features: 16D observation space with 3D Integral Error tracking.
- **V2: Attitude Control (Experimental)**
  - Commands: `[roll, pitch, yaw, altitude]`
- **Sim2Sim Transfer:** Integrated with a custom **X500 quadrotor URDF and YAML** to optimize zero-shot capability for Gazebo deployment.

📄 **Documentation & Results:** [Hover Control Implementation Journey & Results](hover_control_v1_report.md)

---

## 🗺️ Task 2: Point-to-Point (P2P) Navigation with Obstacle Avoidance

Autonomous route planning and obstacle negotiation using simulated rangefinder arrays.

- **Sensor Payload:** Multi-beam LiDAR data integrated into the RL observation state.
- **Environment:** Constrained spatial settings (Warehouse models, dynamic boundaries).

📄 **Documentation:** [P2P Research Summary](RESEARCH_SUMMARY.md) | [Training Guide](TRAINING_README.md)

---

## 🛠️ Quick Start

### 1. Environment Setup
```bash
conda activate dronerl
pip install -r requirements.txt
```

### 2. Execution Commands
**Hover Control V1 (Rate Control):**
```bash
# Training
python3 hover_control/train_hover.py
# Evaluation
python3 hover_control/evaluate_hover.py
```

**P2P Navigation:**
```bash
# Training
python3 train_sac.py
# Evaluation
python3 evaluate_sac.py
```
