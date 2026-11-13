# 🚀 LunarLander-v3: Precision Lander with PPO

A continuous-physics rocket lander agent trained in Gymnasium's Box2D `LunarLander-v3` environment using **Proximal Policy Optimization (PPO)**.

---

## 🎬 Landing Demo
<p align="center">
  <video src="LanderLanding_perfect.mp4" width="85%" controls autoplay loop muted></video>
</p>

---

## 📊 Environment & Training Specifications
- **Observation Space:** 8 continuous state variables (Coordinates $(x, y)$, Linear velocities $(v_x, v_y)$, Angle $\theta$, Angular velocity $\omega$, Left leg contact boolean, Right leg contact boolean).
- **Action Space:** Discrete (0: Do nothing, 1: Fire left orientation engine, 2: Fire main engine, 3: Fire right orientation engine).
- **Key Hyperparameters:**
  - Policy: `MlpPolicy`
  - Discount Factor ($\gamma$): `0.999` (essential for long-horizon descent planning)
  - Entropy Coefficient (`ent_coef`): `0.01` (prevents premature policy collapse)
- **Performance:** Consistently scores between **+200 and +280+**, landing smoothly between the flags without tipping or hard impact.

---

## 🚀 How to Run

```bash
# Activate virtual environment
# Windows: .venv\Scripts\activate | Linux: source .venv/bin/activate

# Run evaluation with trained model weights (ppo_lunar_lander.zip)
python LunarLander.py
```
To train further, toggle `TRAIN_MORE = True` inside `LunarLander.py`.

