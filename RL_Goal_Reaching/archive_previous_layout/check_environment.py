"""Small reproducible checks for the environment's actual learning task."""

import numpy as np
from stable_baselines3.common.env_checker import check_env

from goal_env import GoalEnv, rule_action

env = GoalEnv()
check_env(env)
a, _ = env.reset(seed=42)
b, _ = env.reset(seed=42)
assert np.array_equal(a, b), 'Reset must be repeatable'
env.position = np.array([0., 0.])
env.goal = np.array([1., 0.])
env.heading = 0.0
obs, reward, done, truncated, _ = env.step([1, 0])
assert np.allclose(env.position, [0.08, 0]) and reward > 0
env.step([-1, 1])
assert np.isclose(env.heading, .25), 'Positive turning must rotate counterclockwise'
env.position = np.array([.9, 0.])
_, reward, done, truncated, _ = env.step([-1, 0])
assert done and not truncated and reward > 9
env.reset(seed=42)
env.steps = env.MAX_STEPS - 1
_, _, done, truncated, _ = env.step([-1, 0])
assert truncated and not done
for action in ([0], [float('nan'), 0]):
    try:
        env.step(action)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid action accepted')
for seed in range(20):
    obs, _ = env.reset(seed=seed)
    for _ in range(env.MAX_STEPS):
        obs, _, done, truncated, _ = env.step(rule_action(obs))
        if done or truncated:
            break
    assert done, f'Programmed controller failed on seed {seed}'
print('Passed: Gymnasium/SB3 contract, seeding, forward motion, turning, reward, success, timeout, invalid input, and 20 programmed-controller scenarios.')
