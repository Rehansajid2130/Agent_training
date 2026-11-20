"""A robot learns to reach a goal while avoiding one circular obstacle."""

import math
import csv
from pathlib import Path
import tkinter as tk

import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Change these settings, then run this file.
TRAIN_AGENT = True       # Train a fresh agent for SEED and replace its saved weights.
EVALUATE_AGENT = False    # Print and save before/after results for SEED.
RUN_NOISE_TEST = False    # Print and save clean/noisy results for SEED.
SEED = 7                 # Existing saved agents: 7, 17, 27.
TRAINING_STEPS = 300000
EPISODES = 100

RESULTS = Path(__file__).resolve().parent / 'results'
NOISE_LEVELS = [('clean', 0.0, 0.0), ('small', 0.02, 2.0), ('moderate', 0.05, 5.0)]
torch.set_num_threads(1)


class GoalEnv(gym.Env):
    DT = 0.1
    MAX_SPEED = 0.8
    MAX_TURN = 2.5
    GOAL_RADIUS = 0.15
    MAX_STEPS = 250
    OBSTACLE_RADIUS = 0.45
    ROBOT_RADIUS = 0.08

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.obstacle = np.array([0., 0.])
        self.observation_space = gym.spaces.Box(
            np.array([0, -1, -1, 0, -1, -1], dtype=np.float32),
            np.array([np.inf, 1, 1, np.inf, 1, 1], dtype=np.float32))

    def observation(self):
        offset = self.goal - self.position
        distance = float(np.linalg.norm(offset))
        bearing = math.atan2(offset[1], offset[0]) - self.heading
        offset_obstacle = self.obstacle - self.position
        obstacle_distance = float(np.linalg.norm(offset_obstacle))
        obstacle_bearing = math.atan2(offset_obstacle[1], offset_obstacle[0]) - self.heading
        return np.array([distance / 6.0, math.sin(bearing), math.cos(bearing),
                         obstacle_distance / 6.0, math.sin(obstacle_bearing),
                         math.cos(obstacle_bearing)], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Start and goal lie on roughly opposite sides of the fixed obstacle.
        angle = self.np_random.uniform(-math.pi, math.pi)
        goal_angle = angle + math.pi + self.np_random.uniform(-0.5, 0.5)
        self.position = self.np_random.uniform(1.2, 2.4) * np.array([math.cos(angle), math.sin(angle)])
        self.goal = self.np_random.uniform(1.2, 2.4) * np.array([math.cos(goal_angle), math.sin(goal_angle)])
        self.heading = float(self.np_random.uniform(-math.pi, math.pi))
        self.steps, self.path_length = 0, 0.0
        self.initial_distance = float(np.linalg.norm(self.goal - self.position))
        self.min_clearance = float(np.linalg.norm(self.position-self.obstacle)) - self.OBSTACLE_RADIUS - self.ROBOT_RADIUS
        return self.observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError('Action must contain two finite numbers: speed and turning.')
        action = np.clip(action, -1, 1)
        speed = float((action[0] + 1) * 0.5) * self.MAX_SPEED
        turn = float(action[1]) * self.MAX_TURN
        old_distance = float(np.linalg.norm(self.goal - self.position))
        # Euler kinematics: move along the current heading, then update heading.
        travel = speed * self.DT
        previous = self.position.copy()
        self.position += travel * np.array([math.cos(self.heading), math.sin(self.heading)])
        self.heading = (self.heading + turn*self.DT + math.pi) % (2*math.pi) - math.pi
        self.steps += 1
        self.path_length += travel
        distance = float(np.linalg.norm(self.goal - self.position))
        # Sweep the entire movement segment to detect collisions, including grazing crossings.
        segment = self.position - previous
        fraction = float(np.clip(np.dot(self.obstacle-previous, segment) /
                                 max(float(np.dot(segment, segment)), 1e-20), 0, 1))
        closest = previous + fraction*segment
        clearance = float(np.linalg.norm(closest-self.obstacle)) - self.OBSTACLE_RADIUS - self.ROBOT_RADIUS
        self.min_clearance = min(self.min_clearance, clearance)
        collision = clearance <= 0
        success = distance <= self.GOAL_RADIUS and not collision
        near_penalty = 0.05 * max(0.0, 1.0 - max(0.0, clearance)/0.3)
        reward = 10*(old_distance-distance) - 0.02 - near_penalty
        reward += 20.0 if success else (-15.0 if collision else 0.0)
        info = dict(is_success=success, distance=distance, path_length=self.path_length,
                    initial_distance=self.initial_distance, collision=collision,
                    min_clearance=self.min_clearance)
        terminated = success or collision
        return self.observation(), float(reward), terminated, self.steps >= self.MAX_STEPS and not terminated, info


def rule_action(observation):
    """Turn toward the goal and reduce forward speed when misaligned or close."""
    distance, sine, cosine = map(float, observation[:3])
    angle = math.atan2(sine, cosine)
    speed_fraction = min(1.0, distance*6.0) * max(0.0, cosine)
    return np.array([2*speed_fraction - 1, np.clip(2*angle / GoalEnv.MAX_TURN, -1, 1)],
                    dtype=np.float32)


def noisy_observation(observation, rng, distance_std, angle_std_degrees):
    """Add Gaussian noise to physical distances/bearings, then rebuild the six inputs."""
    if distance_std == 0 and angle_std_degrees == 0:
        return observation.copy()
    measured = observation.copy()
    distance_errors = rng.normal(0, distance_std, size=2)
    angle_errors = rng.normal(0, math.radians(angle_std_degrees), size=2)
    for index, offset in enumerate((0, 3)):
        measured[offset] = max(0.0, float(observation[offset])*6 + distance_errors[index]) / 6
        bearing = math.atan2(float(observation[offset+1]), float(observation[offset+2])) + angle_errors[index]
        measured[offset+1] = math.sin(bearing)
        measured[offset+2] = math.cos(bearing)
    return measured


def load_model(untrained=False):
    name = 'untrained' if untrained else 'ppo'
    path = RESULTS / f'{name}_seed_{SEED}.zip'
    if not path.exists():
        raise FileNotFoundError(f'{path.name} is missing. Set TRAIN_AGENT = True first.')
    return PPO.load(path, device='cpu')


def train():
    env = make_vec_env(GoalEnv, n_envs=8, seed=SEED)
    model = PPO('MlpPolicy', env, seed=SEED, device='cpu', n_steps=256,
                batch_size=128, n_epochs=5, learning_rate=0.0003,
                gamma=0.99, ent_coef=0.005, verbose=1)
    RESULTS.mkdir(exist_ok=True)
    model.save(RESULTS / f'untrained_seed_{SEED}.zip')
    model.learn(total_timesteps=TRAINING_STEPS)
    model.save(RESULTS / f'ppo_seed_{SEED}.zip')
    env.close()


def evaluate(noise=False):
    model = load_model()
    # Each test starts from the same scenarios. Only the controller's inputs change.
    if noise:
        tests = [(name, model, distance, angle) for name, distance, angle in NOISE_LEVELS]
    else:
        tests = [('untrained', load_model(True), 0, 0),
                 ('goal_only', None, 0, 0), ('trained', model, 0, 0)]
    rows = []
    print('\nCondition       Success   Collision   Timeout')
    for name, policy, distance_std, angle_std in tests:
        counts = {'success': 0, 'collision': 0, 'timeout': 0}
        env = GoalEnv()
        for episode in range(EPISODES):
            scenario = 100000 + episode
            observation, _ = env.reset(seed=scenario)
            rng = np.random.default_rng(500000 + scenario)
            for step in range(env.MAX_STEPS):
                measured = noisy_observation(observation, rng, distance_std, angle_std)
                action = rule_action(measured) if policy is None else policy.predict(measured, deterministic=True)[0]
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            outcome = 'success' if info['is_success'] else 'collision' if info['collision'] else 'timeout'
            counts[outcome] += 1
            rows.append([SEED, scenario, name, distance_std, angle_std, outcome,
                         step+1, info['path_length'], info['min_clearance']])
        rates = [100*counts[key]/EPISODES for key in ('success', 'collision', 'timeout')]
        print(f'{name:<15} {rates[0]:>5.1f}%      {rates[1]:>5.1f}%     {rates[2]:>5.1f}%')
        env.close()
    RESULTS.mkdir(exist_ok=True)
    filename = f'latest_noise_seed_{SEED}.csv' if noise else f'latest_evaluation_seed_{SEED}.csv'
    with (RESULTS / filename).open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['model_seed', 'scenario_seed', 'condition', 'distance_std_units',
                         'bearing_std_degrees', 'outcome', 'steps', 'path_length', 'min_clearance'])
        writer.writerows(rows)
    print(f'Saved: {filename}\n')


def demo():
    model, untrained = load_model(), load_model(True)
    root = tk.Tk()
    root.title('Simple Robot Obstacle Avoidance')
    root.resizable(False, False)
    noise_mode = tk.BooleanVar(value=False)
    header = tk.Label(root, font=('Segoe UI', 12))
    header.pack(pady=10)
    titles, canvases = [], []
    panels = tk.Frame(root)
    panels.pack()
    for index in range(3):
        frame = tk.Frame(panels)
        frame.grid(row=0, column=index, padx=5)
        title = tk.Label(frame, font=('Segoe UI', 12, 'bold'))
        title.pack()
        canvas = tk.Canvas(frame, width=320, height=340, bg='white')
        canvas.pack()
        titles.append(title)
        canvases.append(canvas)
    envs = [GoalEnv() for _ in range(3)]
    scenario, paused = 100000, False
    observations, finished, outcomes, paths, generators = [], [], [], [], []

    def reset(next_scenario=False):
        nonlocal scenario, observations, finished, outcomes, paths, generators
        scenario += int(next_scenario)
        observations = [env.reset(seed=scenario)[0] for env in envs]
        finished, outcomes = [False]*3, ['Running']*3
        paths = [[env.position.copy()] for env in envs]
        generators = [np.random.default_rng(500000+scenario) for _ in envs]
        names = ['Clean inputs', 'Small noise', 'Moderate noise'] if noise_mode.get() else ['Before training', 'Goal-only controller', 'After training']
        for title, name in zip(titles, names):
            title.config(text=name)
        header.config(text=f'Scenario {scenario} | Model seed {SEED} | Same start and goal in every panel')

    def draw(index):
        env, canvas, path = envs[index], canvases[index], paths[index]
        canvas.delete('all')
        points = np.vstack([path, env.goal, env.obstacle+0.6, env.obstacle-0.6])
        low, high = points.min(axis=0)-0.5, points.max(axis=0)+0.5
        middle, span = (low+high)/2, max(4.5, float(max(high-low)))

        def pixel(point):
            return 160+(point[0]-middle[0])*260/span, 145-(point[1]-middle[1])*260/span

        def circle(point, radius, colour):
            x, y = pixel(point)
            radius *= 260/span
            canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill=colour)

        circle(env.obstacle, env.OBSTACLE_RADIUS+env.ROBOT_RADIUS, '#fee2e2')
        circle(env.obstacle, env.OBSTACLE_RADIUS, '#ef4444')
        circle(env.goal, env.GOAL_RADIUS, '#22c55e')
        if len(path) > 1:
            canvas.create_line(*[value for point in path for value in pixel(point)], fill='#94a3b8')
        circle(env.position, env.ROBOT_RADIUS, '#2563eb')
        x, y = pixel(env.position)
        canvas.create_line(x, y, x+15*math.cos(env.heading), y-15*math.sin(env.heading), arrow=tk.LAST)
        canvas.create_text(160, 305, text=f'{outcomes[index]} | Steps: {env.steps}')
        canvas.create_text(160, 325, text=f'Path: {env.path_length:.2f} units')

    def tick():
        if not paused:
            for index, env in enumerate(envs):
                if finished[index]:
                    continue
                observation = observations[index]
                if noise_mode.get():
                    _, distance_std, angle_std = NOISE_LEVELS[index]
                    measured = noisy_observation(observation, generators[index], distance_std, angle_std)
                    action = model.predict(measured, deterministic=True)[0]
                elif index == 1:
                    action = rule_action(observation)
                else:
                    policy = untrained if index == 0 else model
                    action = policy.predict(observation, deterministic=True)[0]
                observations[index], _, terminated, truncated, info = env.step(action)
                finished[index] = terminated or truncated
                if finished[index]:
                    outcomes[index] = 'Goal reached' if info['is_success'] else 'Collision' if info['collision'] else 'Time limit'
                paths[index].append(env.position.copy())
        for index in range(3):
            draw(index)
        root.after(40, tick)

    def pause():
        nonlocal paused
        paused = not paused

    controls = tk.Frame(root)
    controls.pack(pady=10)
    tk.Checkbutton(controls, text='Compare observation noise', variable=noise_mode, command=reset).pack(side='left')
    tk.Button(controls, text='Next scenario', command=lambda: reset(True)).pack(side='left', padx=5)
    tk.Button(controls, text='Replay', command=reset).pack(side='left', padx=5)
    tk.Button(controls, text='Pause / Resume', command=pause).pack(side='left', padx=5)
    tk.Label(root, text='Red: obstacle | Green: goal | Blue: robot | Each panel auto-scales').pack(pady=5)
    reset()
    tick()
    root.mainloop()


def main():
    if TRAIN_AGENT:
        train()
    if EVALUATE_AGENT:
        evaluate()
    if RUN_NOISE_TEST:
        evaluate(noise=True)
    demo()


if __name__ == '__main__':
    main()
