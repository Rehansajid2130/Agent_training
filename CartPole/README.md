# 🕹️ CartPole-v1: Inverted Pendulum Balancing with PPO

A Deep Reinforcement Learning agent trained using **Proximal Policy Optimization (PPO)** with Stable-Baselines3 to solve the classic Gymnasium `CartPole-v1` environment.

---

## 🎬 Balancing Demo
<p align="center">
  <video src="CartPole.mp4" width="85%" controls autoplay loop muted></video>
</p>

---

## 📊 Environment Specifications
- **Observation Space:** 4 continuous variables (Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity).
- **Action Space:** Discrete (`0`: Push Left, `1`: Push Right).
- **Reward Function:** `+1.0` for every step the pole angle remains upright (within $\approx 12^\circ$).
- **Success Criteria:** Solved when average score across 100 consecutive episodes is $\ge 475.0$.
- **Performance:** Reaches the maximum possible score of **500.0**.

---

## 🚀 How to Run

```bash
# Activate virtual environment
# Windows: .venv\Scripts\activate | Linux: source .venv/bin/activate

# Run evaluation with trained model weights (ppo_cartpole.zip)
python CartPole.py
```
To train from scratch or fine-tune, toggle `TRAIN_MORE = True` inside `CartPole.py`.

