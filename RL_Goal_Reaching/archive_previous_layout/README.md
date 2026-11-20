# Learning to Reach a Goal with a 2D Mobile Robot

**Author:** Rehan Sajid  
**Project type:** Reinforcement learning and robotics simulation  
**Method:** PPO compared with random actions and a proportional programmed controller

## Abstract

This project investigates whether a reinforcement-learning agent can learn a simple goal-reaching behaviour from distance and heading information. A custom Gymnasium environment models an ideal mobile robot on an empty 2D plane. The robot chooses continuous forward and turning speeds, receives a reward for approaching the goal, and receives an additional reward for reaching it. Three PPO agents are trained with different random seeds and evaluated against random actions and a programmed controller on the same held-out scenarios. The implementation includes saved models, episode-level training logs, a learning curve, raw evaluation data, and a side-by-side demonstration. It uses Stable-Baselines3's PPO implementation rather than implementing PPO from scratch.

## 1. Research Question

Can PPO learn reliable goal reaching in a simple kinematic environment, and how does its success rate, episode length, and travelled distance compare with random actions and a simple programmed controller?

The purpose is to understand the relationship between observations, actions, reward design, training, and evaluation. No new reinforcement-learning algorithm is proposed.

## 2. Run the Project

Double-click **Launch Demo.bat** to view the three controllers on the same starting scenario. Trained models are included; no training is required to use the demo.

- **Next scenario:** choose the next fixed test seed for all three controllers.
- **Replay:** reset the same scenario, including the random controller's action sequence.
- **Pause / Resume:** freeze or continue all three views.

Each panel automatically scales to show its own trajectory; apparent on-screen path length is not a common scale. The displayed numerical path length uses the same units in every panel. One step represents 0.1 simulated seconds; animation speed is for presentation rather than real-time control.

The project uses the normal Windows Python installation, which already has the required learning libraries. The older `AI_training/.venv` does not contain PPO/PyTorch and is not used by the launchers. If using a different machine, install `requirements.txt` and ensure Python includes Tkinter.

## 3. Environment Design

### Robot motion

The robot is represented by its position `(x, y)` and heading `theta`. Its motion follows a discrete kinematic model:

```text
x_next     = x + v * cos(theta) * dt
y_next     = y + v * sin(theta) * dt
theta_next = wrap(theta + omega * dt)
```

Here `dt = 0.1 s`, maximum forward speed is 0.8 units/s, and maximum turning speed is 2.5 rad/s. This approximates differential-drive motion at the commanded body-speed level. It does not simulate individual wheel motors, acceleration, mass, friction, or wheel slip.

Start and goal positions are sampled from `[-2, 2] x [-2, 2]`, with an initial separation of at least 0.8 units. The heading is sampled from `[-pi, pi]`. These sampling limits are not physical walls: motion takes place on an unbounded, obstacle-free plane.

### Observations

The agent receives three numbers:

```text
[distance_to_goal / 6, sin(goal_bearing_error), cos(goal_bearing_error)]
```

Sine and cosine represent heading error without a discontinuity at the -pi/pi boundary. Distance is scaled by 6, but is not clipped to one if the robot travels far away. Observations come directly from the simulator; there is no camera, SLAM, localisation error, or sensor noise.

### Actions

Both action components are in `[-1, 1]`:

```text
forward_speed = (action[0] + 1) / 2 * 0.8
turning_speed = action[1] * 2.5
```

Thus `action[0] = -1` means stop, not reverse; the robot can turn in place. Positive turning rotates counterclockwise.

### Reward and episode ending

```text
reward = 10 * (previous_distance - current_distance) - 0.01
if goal reached:
    reward += 10
```

Approaching the goal gives positive progress reward; moving away gives negative progress reward. The step penalty favours shorter episodes. An episode succeeds when the centre of the robot comes within 0.15 units of the goal. Otherwise it is truncated after 250 steps, equivalent to 25 simulated seconds.

This is a shaped reward, deliberately chosen to make the first project straightforward to learn. Its benefit over a sparse reward has not been measured in an ablation experiment.

## 4. Controllers Compared

| Controller | Decision rule |
|---|---|
| Random | Independently sample both action components uniformly at each step |
| Programmed | Turn proportionally toward the goal; reduce forward speed when misaligned or close |
| PPO | Use a neural-network policy trained from interaction and rewards |

The programmed controller uses the same three observations and action limits as PPO. Its turn gain is 2.0; its forward-speed fraction is `min(1, distance) * max(0, cos(bearing_error))`. It is a transparent reference, not a proven time-optimal or exhaustively tuned controller. Random actions are an exploratory baseline, not a serious navigation method.

## 5. Training Protocol

PPO is trained separately using seeds **7, 17, and 27**. Each run requests 100,000 environment steps. With eight environments and 256 steps per rollout, Stable-Baselines3 rounds the actual budget up to a complete rollout; exact totals and durations are stored in each metadata file.

| Setting | Value |
|---|---|
| Policy | MLP, two hidden layers of 64 units for policy and value networks |
| Vector environments | 8, sequential DummyVecEnv |
| Rollout length per environment | 256 |
| Batch size | 128 |
| Optimisation epochs per rollout | 5 |
| Learning rate | 0.0003 |
| Discount factor | 0.99 |
| Entropy coefficient | 0.005 |
| Compute | CPU; one PyTorch thread |
| Checkpoint used | Final model from each fixed-budget run |

Other PPO settings use the installed Stable-Baselines3 defaults. Versions, seeds, actual timesteps, and training durations are recorded in `results/metadata_seed_*.json`.

The [learning curve](results/training_curve.svg) shows rolling training success over up to the most recent 100 completed episodes for each agent. This is a training diagnostic, not a held-out performance curve. Episode rewards, lengths, and success flags are retained in `results/training_seed_*.csv`.

## 6. Held-Out Evaluation

Each method is evaluated on the same 100 scenario seeds, **100000-100099**, separate from training environment seed initialisation. The start position, heading, and goal match across all methods for a given seed. These are new samples from the same distribution, not out-of-distribution tests. PPO uses deterministic action selection. No test-based checkpoint selection is performed.

The evaluation reports:

- **Success rate:** fraction of episodes ending inside the goal radius.
- **Mean steps across all episodes:** includes failures that run to the time limit.
- **Mean steps among successes:** excludes failures and must be interpreted alongside success rate.
- **Travelled path length:** sum of the commanded forward travel in simulation units, reported for all episodes and for successful episodes.
- **Episode return:** total shaped reward; this is specific to the chosen reward design.

Read the generated [evaluation report](results/evaluation_report.md), [summary](results/evaluation_summary.json), and [raw episode measurements](results/evaluation_episodes.csv). The report also gives mean success and sample standard deviation across the three PPO training seeds. Test scenarios are shared, so results are paired rather than independent across controllers.

### Recorded result snapshot

The initial completed experiment used 100,352 training steps per PPO agent and 100 evaluation scenarios per method:

| Method | Success rate | Mean steps, all episodes | Mean path, successful episodes |
|---|---:|---:|---:|
| Random actions | 7% | 237.56 | 2.951 |
| Programmed controller | 100% | 45.07 | 2.249 |
| PPO, seed 7 | 100% | 31.57 | 2.320 |
| PPO, seed 17 | 100% | 31.66 | 2.333 |
| PPO, seed 27 | 100% | 31.70 | 2.357 |

PPO reached the goal in fewer steps than this programmed baseline, but travelled slightly longer paths. The programmed controller reduces speed near the target; PPO can approach more quickly under the chosen reward. This is a useful trade-off to discuss, not evidence that PPO is generally better. Random-controller successful-path averages cover only its seven successes, so they are not directly comparable with the other methods' 100 successes. After rerunning experiments, use the generated evaluation report for current results and update this snapshot if needed.

## 7. Reproduction

From a terminal in this folder:

```powershell
python check_environment.py
python project.py train
python project.py evaluate
python project.py demo
```

`Train Agents.bat` runs training; `Evaluate Agents.bat` regenerates the comparison. Training replaces the three generated models and logs. Evaluation replaces its generated CSV, JSON, and report. Copy the `results` folder first if preserving an earlier experiment. After retraining, rerun evaluation to avoid mixing new models with an old report.

To change the budget or episode count:

```powershell
python project.py train --steps 100000
python project.py evaluate --episodes 100
python project.py demo --seed 17
```

The environment checks cover API compatibility, deterministic seeding, motion direction, reward, successful termination, timeout truncation, invalid actions, and the programmed controller on 20 separate scenarios. The demo defaults to seed 7, selected in advance rather than chosen for the best test score.

## 8. Limitations and Interpretation

This deliberately small task has exact state information, no obstacles, no dynamics, and no sensing or actuation uncertainty. Success demonstrates learning in this custom simulation, not readiness for a physical robot or general autonomous navigation.

Three training seeds provide a small check on training variability, not a comprehensive statistical study. A 100% observed success rate on 100 finite scenarios is not a guarantee of reliability. The programmed controller may perform better or worse under different gains; this comparison does not establish that reinforcement learning is necessary or generally superior for a task that has an easy analytical solution.

Training success and evaluation success are separate measurements. Comparing only successful path lengths can also be misleading when methods solve different subsets of scenarios. No reward ablation, noise test, obstacle experiment, or transfer to ROS/Gazebo is included.

## 9. Interview Preparation

1. **What is the agent learning?** A mapping from distance and goal direction to forward and turning speeds.
2. **What is PPO?** A policy-learning algorithm that collects experience and updates the action policy using a clipped objective intended to limit overly large updates. This project uses the library implementation.
3. **Why not use only programmed control?** That is a valid solution here. The project compares it with a learned policy to understand their behaviour and trade-offs.
4. **Why sine and cosine?** They represent an angle continuously at the wraparound boundary.
5. **What makes the reward useful?** It provides feedback for progress before the goal is reached, with a bonus at success and a cost per step.
6. **What was implemented for this project?** The kinematic environment, reward, programmed baseline, training/evaluation pipeline, and visual comparison, using Gymnasium and Stable-Baselines3.
7. **What would be a sensible next experiment?** Add small observation noise and compare both controllers under the same noise conditions. Obstacle avoidance should be treated as a separate extension.

## 10. Files and References

| File | Purpose |
|---|---|
| `goal_env.py` | Environment dynamics, observations, reward, and programmed controller |
| `project.py` | Training, held-out evaluation, learning-curve export, and demo |
| `check_environment.py` | Runnable environment checks |
| `requirements.txt` | Direct library versions used |
| `results/ppo_seed_*.zip` | Three trained policy checkpoints |
| `results/training_seed_*.csv` | Training episode records |
| `results/training_curve.svg` | Rolling training-success plot |
| `results/evaluation_episodes.csv` | Episode-level comparison data |
| `results/evaluation_report.md` | Generated comparison table |

Implementation references: [Gymnasium custom environments](https://gymnasium.farama.org/introduction/create_custom_env/), [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), and [evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).
