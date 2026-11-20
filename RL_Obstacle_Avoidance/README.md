# Learning Goal Reaching Around a Fixed Obstacle with PPO

**Author:** Rehan Sajid  
**Project type:** Reinforcement learning and robotics simulation  
**Research focus:** Compare an agent before and after training on a simple collision-avoidance task

## Abstract

This project extends an earlier obstacle-free goal-reaching environment by introducing one fixed circular obstacle and a robot with a finite collision radius. A PPO agent observes the relative distance and direction of both the goal and obstacle, chooses forward and turning speeds, and receives rewards for goal progress and successful arrival, together with collision and proximity penalties. Three agents are trained from separate random initialisations. Their saved initial policies and final trained policies are evaluated on the same held-out scenarios. Random actions and a goal-only programmed controller provide additional reference conditions. Saved models, training logs, a learning curve, raw evaluation data, and a visual before-and-after demonstration accompany the implementation.

This is an educational simulation investigation using the Stable-Baselines3 PPO implementation. It does not introduce a new algorithm or establish real-robot collision-avoidance reliability.

## 1. Research Questions

1. Does PPO training improve successful goal reaching compared with the same policy's initial weights?
2. Does training reduce collisions and timeouts on this fixed-obstacle task?
3. Can the learned behaviour complete goals that a controller ignoring the obstacle fails to reach safely?

This project adds obstacle observations and explicit collision outcomes to the earlier `RL_Goal_Reaching` project. The earlier three-observation models are not reused: the new policy receives six observations and is trained from scratch.

## 2. Quick Start

Run `python RL_Obstacle_Avoidance.py`, or open that file in your Python editor and press Run. The demo shows:

| Panel | Behaviour |
|---|---|
| Untrained PPO | Deterministic actions from the saved initial network |
| Goal-only controller | Programmed steering toward the goal, deliberately ignoring the obstacle |
| Trained PPO | Deterministic actions from the final trained network |

All three panels use the same start, initial heading, goal, and obstacle. The default displayed model seed is 7, chosen in advance rather than selected for its evaluation score.

Use **Next scenario**, **Replay**, and **Pause / Resume** to inspect trajectories. The red disk is the obstacle; its dashed outline includes the robot radius. Green marks the goal region. Views auto-scale independently, so compare numerical path lengths rather than apparent screen distances. The animation advances one 0.1-second simulation step per frame; display speed is not a real-time control measurement.

The script uses the normal Windows Python installation with the existing learning libraries. They do not activate the older `AI_training/.venv`. On another machine, install `gymnasium`, `stable-baselines3`, `numpy`, and `torch` and ensure Python includes Tkinter.

## 3. Environment and Robot Model

### 3.1 Geometry and scenario sampling

- One fixed obstacle is centred at `(0, 0)` with radius **0.45 units**.
- The robot is a disk with radius **0.08 units**.
- Contact occurs when the robot centre reaches the combined radius of **0.53 units**.
- The robot starts at a random angle and radial distance between **1.2 and 2.4 units** from the obstacle.
- The goal is sampled on approximately the opposite side, with an angular offset of up to **0.5 radians**, and radial distance between **1.2 and 2.4 units**.
- Initial heading is uniform over `[-pi, pi]`.

This deliberately samples routes for which the obstacle is relevant. It does not represent arbitrary maps. There are no physical walls or other obstacles; the plane is unbounded.

### 3.2 Motion

The kinematic update is:

```text
x_next     = x + v * cos(theta) * dt
y_next     = y + v * sin(theta) * dt
theta_next = wrap(theta + omega * dt)
```

The time step is **0.1 s**, maximum forward speed is **0.8 units/s**, and maximum turning speed is **2.5 rad/s**. The model represents body-level motion, rather than wheel motors, inertia, acceleration, friction, or slip.

### 3.3 Observations and actions

The six observations are:

```text
[goal_distance / 6,
 sin(goal_bearing_error), cos(goal_bearing_error),
 obstacle_centre_distance / 6,
 sin(obstacle_bearing_error), cos(obstacle_bearing_error)]
```

Bearings are relative to the robot's heading. Sine and cosine avoid an angular wraparound discontinuity. Distances are scaled but not clipped. Training and the original comparison use exact simulator state. The observation-noise experiment below perturbs policy inputs only; these are not camera or LiDAR measurements. Because obstacle and robot radii are fixed, the obstacle-centre distance also determines clearance.

The policy outputs two numbers in `[-1, 1]`:

```text
forward_speed = (action[0] + 1) / 2 * 0.8
turning_speed = action[1] * 2.5
```

The robot can stop and turn in place but cannot reverse. Positive turning is counterclockwise.

## 4. Collision Detection and Reward

For every movement, the environment finds the closest point on the entire segment between the old and proposed robot positions to the obstacle centre. It checks that distance against the combined obstacle and robot radius. This detects grazing crossings that an endpoint-only check can miss.

```text
clearance = closest_segment_distance - obstacle_radius - robot_radius
collision = clearance <= 0

proximity_penalty = 0.05 * max(0, 1 - max(0, clearance) / 0.3)
reward = 10 * (previous_goal_distance - new_goal_distance)
         - 0.02 - proximity_penalty

if successful arrival:
    reward += 20
elif collision:
    reward -= 15
```

Progress provides feedback before arrival, the per-step cost encourages shorter episodes, and the proximity penalty discourages passing very close to the obstacle. These are deliberately designed shaping terms; their individual benefits have not been isolated by an ablation study.

An episode ends in one of three outcomes:

- **Success:** robot centre within **0.15 units** of the goal, with no collision in that step.
- **Collision:** movement segment contacts the obstacle after accounting for robot size.
- **Timeout:** neither terminal outcome occurs within **250 steps**, or 25 simulated seconds.

Collision takes priority over success. A collision ends the episode at the proposed step endpoint; the environment does not resolve contact forces or interpolate a physical stopping position. Logged travel is the proposed kinematic movement up to episode termination.

## 5. Learning Method

PPO learns a neural-network action policy from repeated interactions and rewards. This project uses the library's clipped policy-optimisation implementation; it does not program obstacle-avoidance actions into the learned policy.

| Training setting | Value |
|---|---|
| Independent training seeds | 7, 17, 27 |
| Requested environment steps per seed | 300,000 |
| Policy/value networks | Separate two-layer MLPs, 64 units per layer |
| Vector environments | 8, sequential DummyVecEnv |
| Rollout steps per environment | 256 |
| Batch size | 128 |
| Optimisation epochs per rollout | 5 |
| Learning rate | 0.0003 |
| Discount factor | 0.99 |
| Entropy coefficient | 0.005 |
| Compute | CPU, one PyTorch thread |
| Evaluation checkpoint | Final fixed-budget checkpoint |

Other PPO parameters use the installed library defaults. The actual budget is rounded up to complete rollouts and recorded with library versions and training durations in `results/metadata_seed_*.json`.

Each initial network is saved **before learning** as `untrained_seed_*.zip`. The final network is saved as `ppo_seed_*.zip`. Thus the untrained reference is the same seed's initialisation, not an unrelated trained model. Deterministic untrained actions are not the same as uniformly random actions.

Episode rewards, lengths, success and collision flags were saved in training CSV files during the original three-seed experiment. The [learning curve](results/training_curve.svg) shows rolling training success across up to 100 recent completed episodes; it is not a held-out success curve.

## 6. Evaluation Protocol

Eight methods are evaluated: random actions, the goal-only controller, three untrained networks, and their three trained counterparts. Every method uses the same **100 test scenario seeds, 100000-100099**. These scenarios are separate seeded samples from the training distribution; they do not test unseen obstacle locations or map types.

Neural policies use deterministic inference. The random baseline samples both actions independently at each step. The programmed controller uses only the goal observations, making it an intentionally obstacle-unaware reference rather than a competitive obstacle-avoidance planner.

No test-based checkpoint selection is performed. The comparison reports:

| Metric | Definition |
|---|---|
| Success rate | Fraction of episodes reaching the goal without collision |
| Collision rate | Fraction ending in contact |
| Timeout rate | Fraction reaching the step limit |
| Successful episode length | Mean steps among successful episodes |
| Successful path length | Mean travelled distance among successful episodes |
| Minimum clearance | Smallest swept obstacle-to-robot surface clearance in an episode |
| Episode return | Sum of the specified shaped rewards |

Success, collision, and timeout percentages partition all evaluated episodes. Successful-only means may represent different subsets of scenarios across methods and must be interpreted alongside success rates.

The original [evaluation report](results/evaluation_report.md) contains the results. The [summary JSON](results/evaluation_summary.json) and [episode CSV](results/evaluation_episodes.csv) retain the numerical evidence. The report also includes mean trained success and sample standard deviation across the three training seeds.

### Recorded result snapshot

The initial completed experiment used **301,056 actual training steps per seed**, rounded up from the requested budget. Each method was tested on 100 shared scenarios:

| Method | Success | Collision | Timeout |
|---|---:|---:|---:|
| Random actions | 0% | 11% | 89% |
| Goal-only controller | 11% | 89% | 0% |
| Untrained PPO, seed 7 | 0% | 7% | 93% |
| Untrained PPO, seed 17 | 0% | 7% | 93% |
| Untrained PPO, seed 27 | 0% | 7% | 93% |
| Trained PPO, seed 7 | 93% | 7% | 0% |
| Trained PPO, seed 17 | 99% | 1% | 0% |
| Trained PPO, seed 27 | 92% | 8% | 0% |

Mean trained success was **94.67%**, with sample standard deviation **3.79 percentage points** across three training seeds. Initial policies frequently travelled away from the target and timed out. Thus their low collision rate alone did not indicate useful navigation. Training eliminated timeouts in this evaluation but did not eliminate collisions; seed 27's collision rate was slightly higher than its own initial policy's rate. The result supports improved goal completion, not an unconditional claim of safer motion.

The demo remains on seed 7 by default despite seed 17's higher test score. This avoids selecting the default model based on the final evaluation. These values are a historical snapshot; after rerunning, consult the generated report and update this section before presenting new results.

## 6.1 Observation-Noise Experiment

**Practical question:** Does a trained controller still reach its goal when the distances and directions it receives are slightly inaccurate?

This experiment keeps the saved policy weights fixed. It adds Gaussian measurement errors to goal and obstacle distances and relative bearings before passing observations to the policy. Sine/cosine inputs are rebuilt from the noisy angle. Negative noisy distances are clamped to zero. The simulator's actual positions, motion, rewards, collisions and success checks remain unchanged.

| Condition | Distance error standard deviation | Bearing error standard deviation |
|---|---:|---:|
| Clean | 0 units | 0 degrees |
| Small noise | 0.02 units | 2 degrees |
| Moderate noise | 0.05 units | 5 degrees |

Errors are resampled at every observation, independently for the goal and obstacle. These settings describe simulated errors in arbitrary length units, not measured sensor accuracy. Gaussian standard deviations are not maximum error bounds.

All three saved trained policies are tested on the same 100 scenario seeds, giving 900 policy-condition episodes. The noise generator is separate from the environment generator and uses seed `500000 + scenario_seed`. Small and moderate conditions use the same standard-normal draws at each step, scaled to their specified noise levels. This makes reruns reproducible and holds the scenario sampling fixed. It does not provide independent repeated noise trials.

Read [the original noise experiment report](results/noise_report.md) and [raw episode measurements](results/noise_evaluation.csv). The clean case is rerun using the current saved models. Model fingerprints are checked before and after evaluation to verify that no learning occurs during this experiment.

To see the effect visually, run the usual single Python file and tick **Compare observation noise**. The three panels then show the same trained policy with clean, small-noise and moderate-noise inputs. Replay resets both the scenario and noise sequence. Untick the box to return to the original before/after comparison. The visual display shows the true simulated positions; noise affects what the policy receives, not where the objects are drawn.

### Recorded observation

Across the three saved policies, clean-input success averaged **94.67%**. Both noisy conditions averaged **94.33%**, a decrease of **0.33 percentage points**. Seeds 7 and 17 retained their success rates of 93% and 99%; seed 27 declined from 92% to 91%. No timeouts occurred in these conditions; unsuccessful episodes ended in collisions.

The measured effect on success was small under these settings. This is a valid experimental result, rather than a reason to claim that noise has no effect or increase it until a dramatic failure appears. Paths and actions can still differ when final outcome counts are similar. More noise repeats and a broader range of scenarios would be needed to assess sensitivity with greater confidence. These values are a historical snapshot. Current reruns write a latest-noise CSV for the selected seed rather than updating the original three-seed report.

### What this demonstrates

The contribution is the practical work of choosing a limitation, changing one part of the input pipeline, checking that true-state evaluation stays valid, running a controlled comparison and explaining the result. A suitable description is: "Extended a simulated PPO robot controller with reproducible observation-noise tests and compared goal success, collisions and timeouts on shared scenarios."

## 7. Reproduction

The active script has five main parts: settings and imports, the robot environment, observation noise, training/evaluation, and the visual demo. Automatic research-report generation, metadata export, learning-curve drawing, and startup test routines have been removed from the runnable script.

At the top of `RL_Obstacle_Avoidance.py`:

```python
TRAIN_AGENT = False
EVALUATE_AGENT = False
RUN_NOISE_TEST = False
SEED = 7
TRAINING_STEPS = 300000
EPISODES = 100
```

- **Normal use:** leave the three switches false to load saved models and show the demo.
- **Train:** set `TRAIN_AGENT = True` to train one fresh agent for the selected `SEED`, save its initial and final weights, then show the demo. It does not resume previous training or train all three seeds automatically.
- **Evaluate:** set `EVALUATE_AGENT = True` to print a before/after table and save `results/latest_evaluation_seed_7.csv` for seed 7.
- **Test noise:** set `RUN_NOISE_TEST = True` to print clean/small/moderate-noise results and save `results/latest_noise_seed_7.csv` for seed 7.
- **Choose a saved agent:** change `SEED` to 7, 17 or 27. The CSV filename follows that choice.

Evaluation is separate from training: turn on its switch if you want to evaluate immediately after retraining. The demo always opens after the selected work. Its checkbox switches between before/after and observation-noise comparisons.

```powershell
python RL_Obstacle_Avoidance.py
```

The script uses nine import statements, each once at the top: standard Python utilities and the existing simulation/learning libraries. PyTorch is used only to limit CPU threads; it does not add a separate hand-written neural-network implementation.

### Existing research evidence

The earlier reports, training logs, metadata and learning curve in `results` are preserved records of the original experiments. The simplified script does not regenerate those reports or plots. Its new evaluations are the `latest_*.csv` files and printed tables for the selected seed. Do not interpret an old three-seed report as a new result after retraining one model. Copy `results` before retraining if preserving the earlier experiment.

The original experiments documented above used all three seeds. To repeat that comparison with this script, select and evaluate each seed separately. The earlier multi-file layout remains in `archive_previous_layout` as a historical backup; it is not required to run the current project.

## 8. Limitations

- One known, fixed circular obstacle is substantially simpler than a cluttered or changing environment.
- Training uses exact geometry. Evaluation now includes artificial observation noise, but no calibrated real-sensor model or actuation noise.
- The robot model has no dynamics or physical collision response.
- Three training seeds and 100 shared test scenarios provide limited evidence, not a safety guarantee.
- The programmed reference ignores the obstacle; no comparison with A*, model predictive control, or an obstacle-aware hand-designed controller is made.
- Before/after differences concern this reward, model, sampling distribution, and training budget. They do not establish general superiority of reinforcement learning.
- Successful-path averages can hide failures. Always report all outcome rates.
- ROS/Gazebo integration and physical robot testing have not been performed.

A sensible next improvement is to repeat the noise experiment with several independent noise seeds per scenario. Varying obstacle position, size, or count would require a separately defined training/evaluation task.

## 9. Interview Notes

1. **What changed from the previous project?** A fixed obstacle, finite robot radius, obstacle observations, collision termination, proximity/collision penalties, and an explicit before-training comparison.
2. **Where does learning happen?** PPO updates the neural-network policy from environment interactions. The reward encourages progress and penalises collisions; it does not supply a route.
3. **Why save initial weights?** To compare each trained policy against its own pre-training behaviour.
4. **Why not check only the new position for collision?** A movement segment can graze the obstacle even when both endpoints are outside it.
5. **Can the robot handle any environment?** No. Evaluation covers one fixed circular obstacle with new start/goal samples.
6. **What does a failure tell us?** Collisions indicate insufficient clearance; timeouts may indicate hesitation, circling, or movement away from the goal. The raw outcomes distinguish these cases.
7. **Did you implement PPO from scratch?** No. The custom environment, reward, tests, evaluation and demo use Stable-Baselines3's implementation.

## 10. File Guide and References

| Item | Purpose |
|---|---|
| `RL_Obstacle_Avoidance.py` | Complete runnable project with training switch at the top |
| `README.md` | Research method, results, limitations and usage |
| `results/` | Saved policy weights, training logs, metadata, learning curve, evaluation evidence and observation-noise results |
| `archive_previous_layout/` | Previous scripts and launchers, retained as a backup |

References: [Gymnasium custom environments](https://gymnasium.farama.org/introduction/create_custom_env/), [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), and [evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).
