# 🚀 Reinforcement Learning with Gymnasium & PPO

A practical hands-on reinforcement learning repository exploring policy-based deep reinforcement learning (**PPO**) across classic control, rocket physics, and computer vision environments using **Gymnasium** and **Stable-Baselines3**.

---

## 📌 Project Overview

This project implements, evaluates, and tunes reinforcement learning agents across three distinct simulation environments:

1. **CartPole-v1**: Classic inverted pendulum balance using vector state observations.
2. **LunarLander-v3**: Lunar descent and pinpoint touchdown between flags utilizing multi-engine physics control, exploration tuning, and persistent model training.
3. **CarRacing-v3**: Continuous track driving combining Box2D vehicle dynamics with camera-based visual inputs and manual keyboard override.

---

## 🎮 The Environments

### 1. 🕹️ CartPole (`CartPole.py`)
- **Objective:** Balance a vertical pole hinged to a cart moving along a frictionless track.
- **Observation Space:** 4 continuous variables:
  - Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity.
- **Action Space:** Discrete (2 actions: Push Left `0` or Push Right `1`).
- **Reward:** `+1` for every frame the pole stays upright (within ~12°).
- **Result:** Fully solved using PPO (`MlpPolicy`), reaching the maximum possible score of **500.0**.

---

### 2. 🌕 LunarLander (`LunarLander.py`) - SOLVED 🎉

- **Objective:** Navigate a lunar lander through gravity, stabilize orientation, and land smoothly between the two target flags.
- **Observation Space:** 8 continuous state variables (coordinates, linear velocities, angle, angular velocity, and ground-contact sensor booleans).
- **Action Space:** Discrete (4 actions: Do nothing `0`, Left thruster `1`, Main engine `2`, Right thruster `3`).
- **Reward:**
  - `+100` to `+140` bonus for landing safely between the target flags.
  - `-100` penalty for crashing.
  - Small negative penalty per frame for firing engines (fuel cost).
- **Key Breakthrough & Hyperparameter Tuning:**
  - Standard PPO can suffer from policy collapse (settling for a "safe" landing outside the flags).
  - Configured `gamma=0.999` (long-horizon discount factor) so the agent values the distant touchdown pad bonus.
  - Configured `ent_coef=0.01` (entropy coefficient) to preserve exploration and prevent premature convergence.
- **Result:** Consistently scores **+200 to +280+**, executing pinpoint soft landings directly on the pad!

#### 🎬 Perfect Landing Demo:
https://github.com/user-attachments/assets/LanderLanding_perfect.mp4

<p align="center">
  <video src="LanderLanding_perfect.mp4" width="100%" controls autoplay loop muted></video>
</p>

> 📹 **Direct video link:** [Watch LanderLanding_perfect.mp4](LanderLanding_perfect.mp4)

---

### 3. 🏎️ CarRacing (`CarRacing.py`)
- **Objective:** Drive a racecar around procedurally generated tracks while staying within road boundaries.
- **Observation Space:** High-dimensional **$96 \times 96 \times 3$ RGB camera image** (27,648 pixels).
- **Action Space:** Continuous controls as a NumPy array: `[steering (-1.0 to 1.0), gas (0.0 to 1.0), brake (0.0 to 1.0)]`.
- **Modes:**
  - **Manual Play:** Drive the car directly using keyboard arrow keys (`pygame`).
  - **Autonomous Training:** Vision-based training using `CnnPolicy`.

---

## 🧠 Key Reinforcement Learning Concepts Explored

- **Policy-Based vs. Value-Based:** Compared **Q-Learning** (estimating action scores via value iteration) against **PPO (Proximal Policy Optimization)** (directly optimizing action probability distributions with safe clipping updates).
- **Headless Training vs. Visual Rendering:**
  - *Training:* Run in headless mode (without `render_mode="human"`) to process physics at **1,500+ FPS**.
  - *Evaluation:* Enable `render_mode="human"` at 50 FPS for human-observable testing and replay.
- **Avoiding Policy Collapse & Local Minima:**
  - Preventing premature convergence by maintaining curiosity via `ent_coef`.
  - Adjusting `gamma` to account for multi-hundred-step episodic horizons.
- **Model Checkpointing & Fine-Tuning:**
  - Serializing neural network weights to disk via `.save()`.
  - Resuming training on pre-trained checkpoints with `PPO.load(..., env=train_env)`.
- **Network Architectures:**
  - **`MlpPolicy`:** Multi-Layer Perceptrons for flat numerical states (CartPole, LunarLander).
  - **`CnnPolicy`:** Convolutional Neural Networks with spatial feature extractors for raw pixel frames (CarRacing).

---

## 📂 Repository Structure

```text
AI_training/
├── CartPole.py                 # CartPole PPO training & testing script
├── LunarLander.py              # LunarLander training, checkpointing & evaluation
├── CarRacing.py                # CarRacing simulation (Manual keyboard & PPO agent)
├── ppo_lunar_lander.zip        # Trained PPO neural network weights for LunarLander
├── LanderLanding_perfect.mp4   # Video recording of the solved landing between flags
└── README.md                   # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Recommended: A Python virtual environment (`.venv`)

### 1. Clone the Repository
```bash
git clone https://github.com/Rehansajid2130/Agent_training.git
cd Agent_training
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Core RL dependencies
pip install gymnasium[classic-control] gymnasium[box2d] stable-baselines3 pygame numpy
```

---

## 🚀 How to Run

### 1. Run CartPole
```bash
python CartPole.py
```
Loads or trains the agent for 25,000 steps and opens a window displaying the balanced cartpole.

### 2. Run LunarLander
```bash
python LunarLander.py
```
- If `ppo_lunar_lander.zip` exists and `TRAIN_MORE = False`, it immediately opens the simulation window to display perfect landings between the flags.
- Set `TRAIN_MORE = True` to continue training the model for additional timesteps.

### 3. Run CarRacing
```bash
python CarRacing.py
```
Run the racetrack simulation. Use the arrow keys (`←`, `→`, `↑`, `↓`) to steer and accelerate manually, or enable autonomous mode to train with `CnnPolicy`.

---

## 📊 Summary of Results

| Environment | Observation Type | Policy | Training Timesteps | Final Score / Status |
| :--- | :--- | :--- | :--- | :--- |
| **CartPole-v1** | 4D Vector | `MlpPolicy` | 25,000 | **500.0** (Perfect Score) |
| **LunarLander-v3** | 8D Vector | `MlpPolicy` | 300,000 | **+200 to +280+** (Solved - Pinpoint Landing) |
| **CarRacing-v3** | 96x96x3 Image | `CnnPolicy` / Manual | Modular | Playable / Continuous Control |

---

## 📜 Acknowledgements & References
- [Gymnasium Documentation (Farama Foundation)](https://gymnasium.farama.org/)
- [Stable-Baselines3 (DLR-RM)](https://stable-baselines3.readthedocs.io/)
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
