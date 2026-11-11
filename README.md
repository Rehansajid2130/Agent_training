# 🚀 Reinforcement Learning with Gymnasium & PPO

A practical hands-on reinforcement learning repository exploring policy-based deep reinforcement learning (**PPO**) across classic control, rocket physics, and computer vision environments using **Gymnasium** and **Stable-Baselines3**.

---

## 📌 Project Overview

This project implements, evaluates, and tunes reinforcement learning agents across three distinct simulation environments:

1. **CartPole-v1**: Classic inverted pendulum balance using vector state observations.
2. **LunarLander-v3**: Lunar descent and pinpoint touchdown between flags utilizing multi-engine physics control, exploration tuning, and persistent model training.
3. **CarRacing-v3**: Continuous track driving combining Box2D vehicle dynamics with camera-based visual inputs and manual keyboard override.
4. **ROS 2 + Gazebo (`ROS/`)**: Autonomous mobile robot perception-action pipeline tracking and following colored objects via OpenCV computer vision and closed-loop proportional control.

---

## 🎮 The Environments

### 1. 🕹️ CartPole (`CartPole.py`) - SOLVED 🎉
- **Objective:** Balance a vertical pole hinged to a cart moving along a frictionless track.
- **Observation Space:** 4 continuous variables:
  - Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity.
- **Action Space:** Discrete (2 actions: Push Left `0` or Push Right `1`).
- **Reward:** `+1` for every frame the pole stays upright (within ~12°).
- **Result:** Fully solved using PPO (`MlpPolicy`), reaching the maximum possible score of **500.0**.

#### 🎬 Balancing Demo:
https://github.com/user-attachments/assets/CartPole.mp4

<p align="center">
  <video src="CartPole.mp4" width="100%" controls autoplay loop muted></video>
</p>

> 📹 **Direct video link:** [Watch CartPole.mp4](CartPole.mp4)

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

### 3. 🏎️ CarRacing (`CarRacing.py`) - Vision-Based Deep RL & CNN Architecture

#### 🎬 Autonomous Driving Demo:
<p align="center">
  <img src="carracing_trained_demo.gif" alt="Trained CarRacing Agent Demo" width="60%">
</p>

#### 🔴 The Problem & Challenges
Autonomous track driving presents three compounding challenges not found in classic control:
1. **High-Dimensional Visual Perception:** The agent perceives a raw $96 \times 96 \times 3$ RGB camera stream (27,648 pixels) rather than pre-computed state vectors.
2. **Vehicle Dynamics & Inertia:** Non-linear tire friction, centrifugal drift on corners, and drastic friction loss on grass surfaces require forward planning and braking control.
3. **The "Reward Hacking" (Suicide Exploit):**
   - Tile crossing yields $\approx +3.3$ points; every frame alive costs $-0.1$ time penalty.
   - Without intervention, naive agents learn to collect the easy straightaway tiles (+100) and **deliberately steer into the black border** to trigger early termination, locking in positive scores while avoiding tricky turns!

---

#### 🧪 Methodology & Architecture

##### 1. Frame Stacking & Temporal Awareness
A single $96 \times 96$ snapshot contains zero velocity information (the agent cannot discern if the car is stationary, accelerating, or drifting sideways). Stacking $k=4$ consecutive frames gives the visual policy **optical flow**, enabling it to infer:
$$\text{State}_t = \{ \text{Frame}_t, \, \text{Frame}_{t-1}, \, \text{Frame}_{t-2}, \, \text{Frame}_{t-3} \}$$

##### 2. Nature CNN Feature Extractor
The policy utilizes a standard deep convolutional backbone designed for pixel-based reinforcement learning:
```
Input: (96, 96, 3) Image Frame
  │
  ├── Conv2D (32 filters, 8x8 kernel, stride 4) + ReLU
  ├── Conv2D (64 filters, 4x4 kernel, stride 2) + ReLU
  ├── Conv2D (64 filters, 3x3 kernel, stride 1) + ReLU
  ├── Flatten
  └── Dense Linear (512 units) + ReLU
        │
        ├── Actor Head  --> Continuous Action: [Steering, Gas, Brake]
        └── Critic Head --> State Value V(s)
```

##### 3. PPO Hyperparameters & Optimization
- **Policy:** `CnnPolicy` (Nature-CNN feature extractor + Gaussian policy head)
- **Parallel Vectorization:** $N=4$ asynchronous environments via `make_vec_env`, boosting CPU training speed from 40 FPS to **180+ FPS**
- **Discount Factor ($\gamma$):** $0.99$
- **GAE Parameter ($\lambda$):** $0.95$
- **Clip Range ($\epsilon$):** $0.2$
- **Learning Rate:** $3 \times 10^{-4}$

##### 4. Anti-Suicide Reward Shaping & Zoom-In Bypass
- **Zoom-In Fast Forward:** Skips the initial 40-step camera zoom animation with neutral inputs so the car begins with the road clearly ahead.
- **Out-of-Bounds Penalty:** Applies a severe **$-100.0$ penalty** on early border crash termination. This mathematically destroys the suicide exploit and forces the policy to learn braking and cornering.

---

#### ⚖️ Architectural Comparison: MLP vs. CNN

| Characteristic | Multi-Layer Perceptron (`MlpPolicy`) | Convolutional Neural Network (`CnnPolicy`) |
| :--- | :--- | :--- |
| **Input Representation** | 1D flattened vector (27,648 values) | 2D/3D tensor $(C, H, W)$ |
| **Spatial Locality** | ❌ Completely lost (pixels treated as unrelated inputs) | ✅ Preserved via local receptive fields ($8\times8, 4\times4$) |
| **Translation Invariance** | ❌ None (curve shifting 2 pixels looks like an entirely new state) | ✅ Robust to track shifts and camera repositioning |
| **Edge & Road Detection** | ❌ Struggles to segment asphalt borders from grass | ✅ Hierarchical filters capture road edges, curbs, and corners |
| **Driving Performance** | 💥 **Failed:** Crashes on turn 1 (Score: $-82$) | 🏆 **Success:** Navigates sharp turns & drifts (Score: $+150$ to $+250+$) |

---

#### 📈 Learning-Curve Analysis
<p align="center">
  <img src="carracing_learning_curve.png" alt="CarRacing PPO Learning Curves" width="90%">
</p>

- **MLP Baseline (Red):** Fails to extract 2D road geometry, flatlining around $-60$ to $-80$.
- **Standard CNN (Orange):** Learns straightaways rapidly but plateaus around $+130$ once it exploits the border suicide loophole.
- **CNN + Anti-Suicide Wrapper (Green):** Penalizing out-of-bounds driving forces the agent through turn-braking mastery, climbing steadily past $+250+$.

---

#### 🎮 Available Operational Modes in `CarRacing.py`
- **`MODE = "WATCH"`:** Loads `ppo_carracing.zip` and watches the trained CNN navigate laps.
- **`MODE = "MANUAL"`:** Drive manually with keyboard arrow keys (`←`, `→`, `↑`, `↓`).
- **`MODE = "TRAIN"`:** Launches 4-track vectorized parallel PPO training.

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
├── ppo_carracing.zip           # Trained PPO CNN weights for CarRacing
├── CartPole.mp4                # Video recording of CartPole balancing
├── LanderLanding_perfect.mp4   # Video recording of the solved landing between flags
├── carracing_trained_demo.gif  # GIF recording of the trained CNN racing
├── carracing_learning_curve.png# Learning-curve plot comparing MLP vs CNN architectures
├── ROS/                        # ROS 2 + Gazebo autonomous robotics package
│   ├── ball_follower_node.py   # OpenCV perception & proportional tracking node
│   ├── controller.py           # Basic open-loop velocity publisher demo
│   ├── launch/                 # Simulation & node launch scripts
│   ├── urdf/                   # Differential drive camera robot URDF
│   ├── worlds/                 # Gazebo simulation world with target ball
│   └── README.md               # ROS 2 architecture & execution guide
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
Easily configure `MODE` in `CarRacing.py`:
- `MODE = "WATCH"`: Watch your trained CNN navigate laps.
- `MODE = "MANUAL"`: Drive manually with arrow keys.
- `MODE = "TRAIN"`: Launch high-speed multi-core parallel training.

---

## 📊 Summary of Results

| Environment | Observation Type | Policy | Training Timesteps | Final Score / Status |
| :--- | :--- | :--- | :--- | :--- |
| **CartPole-v1** | 4D Vector | `MlpPolicy` | 25,000 | **500.0** (Perfect Score) |
| **LunarLander-v3** | 8D Vector | `MlpPolicy` | 300,000 | **+200 to +280+** (Solved - Pinpoint Landing) |
| **CarRacing-v3** | 96x96x3 Image | `CnnPolicy` | 150,000+ | **+148.4+** (Autonomous Track Navigation) |

---

## 📜 Acknowledgements & References
- [Gymnasium Documentation (Farama Foundation)](https://gymnasium.farama.org/)
- [Stable-Baselines3 (DLR-RM)](https://stable-baselines3.readthedocs.io/)
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
