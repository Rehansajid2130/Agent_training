"""Train and compare PPO before/after learning to avoid one fixed obstacle."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

from goal_env import GoalEnv, rule_action

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
TRAIN_SEEDS = (7, 17, 27)
TEST_SEED_START = 100000


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_model(seed, untrained=False):
    from stable_baselines3 import PPO
    prefix = 'untrained' if untrained else 'ppo'
    path = RESULTS / f'{prefix}_seed_{seed}.zip'
    if not path.exists():
        raise FileNotFoundError(f'Train first: missing {path.name}')
    return PPO.load(path, device='cpu')


def train(steps):
    import gymnasium
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env

    torch.set_num_threads(1)
    RESULTS.mkdir(exist_ok=True)
    for seed in TRAIN_SEEDS:
        rows = []

        class EpisodeLog(BaseCallback):
            def _on_step(self):
                for info in self.locals['infos']:
                    if 'episode' in info:
                        rows.append(dict(timesteps=self.num_timesteps,
                                         reward=info['episode']['r'],
                                         length=info['episode']['l'], success=int(info['is_success']),
                                         collision=int(info['collision'])))
                if self.n_calls % 2500 == 0:
                    recent = rows[-100:]
                    if recent:
                        rate = statistics.mean(r['success'] for r in recent)*100
                        print(f'Seed {seed}: {self.num_timesteps} steps; recent training success {rate:.1f}%', flush=True)
                return True

        started = time.monotonic()
        env = make_vec_env(GoalEnv, n_envs=8, seed=seed)
        model = PPO('MlpPolicy', env, seed=seed, device='cpu', n_steps=256,
                    batch_size=128, n_epochs=5, learning_rate=3e-4,
                    gamma=0.99, ent_coef=0.005, verbose=0)
        model.save(RESULTS / f'untrained_seed_{seed}.zip')
        model.learn(total_timesteps=steps, callback=EpisodeLog())
        model.save(RESULTS / f'ppo_seed_{seed}.zip')
        write_csv(RESULTS / f'training_seed_{seed}.csv', rows)
        metadata = dict(seed=seed, requested_timesteps=steps, actual_timesteps=model.num_timesteps,
                        wall_seconds=round(time.monotonic()-started, 2), n_envs=8,
                        n_steps=256, batch_size=128, n_epochs=5, learning_rate=3e-4,
                        gamma=0.99, ent_coef=0.005, policy='MlpPolicy [64,64]',
                        gymnasium=gymnasium.__version__, stable_baselines3=stable_baselines3.__version__,
                        torch=torch.__version__, numpy=np.__version__)
        (RESULTS / f'metadata_seed_{seed}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        env.close()
        print(f'Saved seed {seed}; {metadata["wall_seconds"]} seconds.', flush=True)
    plot_training()


def plot_training():
    """Write a portable SVG learning curve without another plotting dependency."""
    colours = ['#2563eb', '#059669', '#9333ea']
    curves = []
    for seed in TRAIN_SEEDS:
        with (RESULTS / f'training_seed_{seed}.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        curves.append([(int(row['timesteps']), statistics.mean(int(r['success']) for r in rows[max(0,i-99):i+1]))
                       for i, row in enumerate(rows)])
    maximum = max(x for curve in curves for x, y in curve)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="460" viewBox="0 0 900 460">',
           '<rect width="900" height="460" fill="white"/>',
           '<g font-family="Arial" fill="#243247">',
           '<text x="65" y="30" font-size="20">PPO training: rolling success over up to 100 episodes</text>']
    for percentage in (0, 25, 50, 75, 100):
        y = 370 - percentage*3
        svg.append(f'<path d="M65 {y} H850" stroke="#e2e8f0"/><text x="25" y="{y+5}">{percentage}%</text>')
    for index, (seed, curve) in enumerate(zip(TRAIN_SEEDS, curves)):
        points = ' '.join(f'{65+x/maximum*785:.1f},{370-y*300:.1f}' for x,y in curve)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{colours[index]}" stroke-width="2"/>')
        svg.append(f'<text x="{150+index*220}" y="440" fill="{colours[index]}">Training seed {seed}</text>')
    for fraction in (0, .25, .5, .75, 1):
        x = 65 + fraction*785
        svg.append(f'<text x="{x-12}" y="395">{int(maximum*fraction):,}</text>')
    svg.append('<text x="370" y="418">Environment steps (all 8 environments)</text></g></svg>')
    (RESULTS / 'training_curve.svg').write_text('\n'.join(svg), encoding='utf-8')


def evaluate(episodes):
    import torch
    torch.set_num_threads(1)
    RESULTS.mkdir(exist_ok=True)
    methods = [('random', None), ('goal_only_rule', None)]
    methods += [(f'untrained_seed_{s}', load_model(s, True)) for s in TRAIN_SEEDS]
    methods += [(f'ppo_seed_{s}', load_model(s)) for s in TRAIN_SEEDS]
    rows, summaries = [], []
    for method, model in methods:
        env = GoalEnv()
        method_rows = []
        for episode in range(episodes):
            scenario_seed = TEST_SEED_START + episode
            obs, _ = env.reset(seed=scenario_seed)
            action_rng = np.random.default_rng(200000 + episode)
            reward_sum = 0.0
            for step in range(env.MAX_STEPS):
                if method == 'random':
                    action = action_rng.uniform(-1, 1, size=2)
                elif method == 'goal_only_rule':
                    action = rule_action(obs)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                reward_sum += reward
                if terminated or truncated:
                    break
            method_rows.append(dict(method=method, scenario_seed=scenario_seed,
                                    success=int(info['is_success']), steps=step+1,
                                    collision=int(info['collision']), timeout=int(truncated),
                                    path_length=info['path_length'], final_distance=info['distance'],
                                    initial_distance=info['initial_distance'], episode_return=reward_sum,
                                    min_clearance=info['min_clearance']))
        rows.extend(method_rows)
        successes = [r for r in method_rows if r['success']]
        summary = dict(method=method, episodes=episodes,
                       success_rate_pct=100*len(successes)/episodes,
                       collision_rate_pct=100*sum(r['collision'] for r in method_rows)/episodes,
                       timeout_rate_pct=100*sum(r['timeout'] for r in method_rows)/episodes,
                       mean_steps_all=statistics.mean(r['steps'] for r in method_rows),
                       mean_steps_success=statistics.mean(r['steps'] for r in successes) if successes else None,
                       mean_path_all=statistics.mean(r['path_length'] for r in method_rows),
                       mean_path_success=statistics.mean(r['path_length'] for r in successes) if successes else None,
                       mean_return=statistics.mean(r['episode_return'] for r in method_rows))
        summaries.append(summary)
        print(f'{method}: {summary["success_rate_pct"]:.1f}% success; '
              f'{summary["collision_rate_pct"]:.1f}% collisions; '
              f'{summary["timeout_rate_pct"]:.1f}% timeouts', flush=True)
        env.close()
    write_csv(RESULTS / 'evaluation_episodes.csv', rows)
    (RESULTS / 'evaluation_summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    report = ['# Held-Out Evaluation Results', '',
              f'Each method was evaluated on the same {episodes} scenario seeds, starting at {TEST_SEED_START}.',
              'PPO used deterministic actions from the final fixed-budget checkpoints. No test-based checkpoint selection was performed.', '',
              '| Method | Success | Collisions | Timeouts | Mean steps, successes | Mean path, successes |',
              '|---|---:|---:|---:|---:|---:|']
    for summary in summaries:
        steps = summary['mean_steps_success']
        path = summary['mean_path_success']
        steps_text = f'{steps:.2f}' if steps is not None else 'N/A'
        path_text = f'{path:.3f}' if path is not None else 'N/A'
        report.append(f'| {summary["method"]} | {summary["success_rate_pct"]:.1f}% | '
                      f'{summary["collision_rate_pct"]:.1f}% | {summary["timeout_rate_pct"]:.1f}% | '
                      f'{steps_text} | {path_text} |')
    rates = [s['success_rate_pct'] for s in summaries if s['method'].startswith('ppo')]
    report.extend(['', f'PPO success across {len(rates)} training seeds: mean {statistics.mean(rates):.2f}%, '
                   f'sample standard deviation {statistics.stdev(rates):.2f} percentage points.', '',
                   'Steps are simulation steps; path lengths are arbitrary simulation units. Successful-only metrics exclude failures and may involve different subsets of scenarios.',
                   'The goal-only programmed controller deliberately ignores obstacle observations; it is not an obstacle-avoidance baseline. '
                   'Untrained policies use deterministic actions from their saved initial weights, not uniformly random actions. '
                   'High success on this small, noise-free task does not establish real-robot reliability or unseen-map generalization.', '',
                   'Raw measurements: [evaluation_episodes.csv](evaluation_episodes.csv). '
                   'Summary: [evaluation_summary.json](evaluation_summary.json).'])
    (RESULTS / 'evaluation_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    return summaries


def demo(seed):
    import tkinter as tk
    from tkinter import ttk
    import torch
    torch.set_num_threads(1)
    model = load_model(seed)
    untrained = load_model(seed, True)
    root = tk.Tk()
    root.title('Obstacle Avoidance | Before Training vs Goal-Only Control vs Trained PPO')
    root.resizable(False, False)
    header = ttk.Label(root, font=('Segoe UI', 13), padding=12)
    header.pack()
    container = ttk.Frame(root)
    container.pack()
    canvases = []
    names = [f'Untrained PPO | seed {seed}', 'Goal-only controller', f'Trained PPO | seed {seed}']
    for index, name in enumerate(names):
        frame = ttk.Frame(container, padding=6)
        frame.grid(row=0, column=index)
        ttk.Label(frame, text=name, font=('Segoe UI', 13, 'bold')).pack()
        canvas = tk.Canvas(frame, width=340, height=350, bg='#f5f7fb', highlightthickness=0)
        canvas.pack()
        canvases.append(canvas)
    envs = [GoalEnv() for _ in names]
    state = dict(scenario=TEST_SEED_START, paused=False)

    def reset(increment=0):
        state['scenario'] += increment
        state['obs'] = [env.reset(seed=state['scenario'])[0] for env in envs]
        state['done'] = [False]*3
        state['success'] = [False]*3
        state['collision'] = [False]*3
        state['paths'] = [[env.position.copy()] for env in envs]
        state['rng'] = np.random.default_rng(200000 + state['scenario'] - TEST_SEED_START)
        header.config(text=f'Same scenario: {state["scenario"]} | One fixed obstacle | 1 step = 0.1 simulated seconds')

    def draw(index):
        canvas, env, path = canvases[index], envs[index], state['paths'][index]
        canvas.delete('all')
        points = np.vstack([path, env.goal, env.obstacle + 0.6, env.obstacle - 0.6])
        low, high = points.min(axis=0)-0.5, points.max(axis=0)+0.5
        span = max(4.5, float(max(high-low)))
        middle = (low+high)*0.5

        def pixel(point):
            return 170+(point[0]-middle[0])*280/span, 155-(point[1]-middle[1])*280/span

        ox, oy = pixel(env.obstacle)
        obstacle_radius = env.OBSTACLE_RADIUS*280/span
        clearance_radius = (env.OBSTACLE_RADIUS+env.ROBOT_RADIUS)*280/span
        canvas.create_oval(ox-clearance_radius, oy-clearance_radius, ox+clearance_radius,
                           oy+clearance_radius, outline='#dc2626', dash=(3, 3))
        canvas.create_oval(ox-obstacle_radius, oy-obstacle_radius, ox+obstacle_radius,
                           oy+obstacle_radius, fill='#f87171', outline='#b91c1c')
        gx, gy = pixel(env.goal)
        radius = env.GOAL_RADIUS*280/span
        canvas.create_oval(gx-radius, gy-radius, gx+radius, gy+radius, fill='#bbf7d0', outline='#16a34a')
        canvas.create_text(gx, gy-15, text='Goal', fill='#166534')
        if len(path) > 1:
            canvas.create_line(*[coordinate for p in path for coordinate in pixel(p)], fill='#97b4d6', width=2)
        sx, sy = pixel(path[0])
        canvas.create_oval(sx-3, sy-3, sx+3, sy+3, fill='#64748b')
        x, y = pixel(env.position)
        robot_radius = env.ROBOT_RADIUS*280/span
        canvas.create_oval(x-robot_radius, y-robot_radius, x+robot_radius, y+robot_radius, fill='#2563eb', outline='white')
        canvas.create_line(x, y, x+17*math.cos(env.heading), y-17*math.sin(env.heading), width=3, fill='#172554', arrow=tk.LAST)
        status = ('COLLISION' if state['collision'][index] else 'GOAL REACHED' if state['success'][index]
                  else 'TIME LIMIT' if state['done'][index] else 'Running')
        canvas.create_text(170, 315, text=f'{status} | Steps: {env.steps} | Path: {env.path_length:.2f} units')
        canvas.create_text(170, 337, text=f'Distance: {np.linalg.norm(env.goal-env.position):.2f} | View span: {span:.1f} units')

    def tick():
        if not state['paused']:
            for index, env in enumerate(envs):
                if not state['done'][index]:
                    obs = state['obs'][index]
                    if index == 0:
                        action, _ = untrained.predict(obs, deterministic=True)
                    elif index == 1:
                        action = rule_action(obs)
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    state['obs'][index] = obs
                    state['done'][index] = terminated or truncated
                    state['success'][index] = info['is_success']
                    state['collision'][index] = info['collision']
                    state['paths'][index].append(env.position.copy())
        for index in range(3):
            draw(index)
        root.after(40, tick)

    controls = ttk.Frame(root, padding=12)
    controls.pack()
    ttk.Button(controls, text='Next scenario', command=lambda: reset(1)).pack(side='left', padx=8)
    ttk.Button(controls, text='Replay', command=reset).pack(side='left', padx=8)
    ttk.Button(controls, text='Pause / Resume', command=lambda: state.update(paused=not state['paused'])).pack(side='left', padx=8)
    ttk.Label(root, text='Views auto-scale independently. Red: obstacle. Dashed ring: obstacle plus robot radius. Green: goal.', padding=8).pack()
    reset()
    tick()
    root.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate', 'demo'])
    parser.add_argument('--steps', type=int, default=300000)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7, choices=TRAIN_SEEDS)
    args = parser.parse_args()
    if args.steps < 1 or args.episodes < 1:
        parser.error('Steps and episodes must be positive.')
    if args.command == 'train':
        train(args.steps)
    elif args.command == 'evaluate':
        evaluate(args.episodes)
    else:
        demo(args.seed)
