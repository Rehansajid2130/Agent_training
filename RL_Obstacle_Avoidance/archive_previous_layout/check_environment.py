"""Focused checks for geometry, collision, termination and API correctness."""

import numpy as np
from stable_baselines3.common.env_checker import check_env
from goal_env import GoalEnv

env = GoalEnv()
check_env(env)
a, _ = env.reset(seed=42)
b, _ = env.reset(seed=42)
assert np.array_equal(a, b) and len(a) == 6
for seed in range(100):
    env.reset(seed=seed)
    assert np.linalg.norm(env.position-env.obstacle) > 1.19
    assert np.linalg.norm(env.goal-env.obstacle) > 1.19

env.reset(seed=0)
env.position = np.array([1., 1.])
env.goal = np.array([2., 1.])
env.heading = 0.
_, reward, done, trunc, info = env.step([1, 0])
assert np.allclose(env.position, [1.08, 1]) and reward > 0 and not done
env.step([-1, 1])
assert np.isclose(env.heading, .25)

env.reset(seed=0)
env.position = np.array([.57, 0.])
env.goal = np.array([-1.5, 0.])
env.heading = np.pi
_, reward, done, trunc, info = env.step([1, 0])
assert done and info['collision'] and not info['is_success'] and reward < 0

# Grazing segment: both endpoints are outside the expanded obstacle, but the line crosses it.
env.reset(seed=0)
env.position = np.array([-.04, .529])
env.goal = np.array([2., 1.])
env.heading = 0.
_, _, done, _, info = env.step([1, 0])
assert done and info['collision'], 'Swept collision check missed a grazing crossing'

env.reset(seed=0)
env.position = np.array([1.9, 1.])
env.goal = np.array([2., 1.])
_, reward, done, trunc, info = env.step([-1, 0])
assert done and info['is_success'] and not info['collision'] and reward > 19
env.reset(seed=0)
env.steps = env.MAX_STEPS-1
_, _, done, trunc, _ = env.step([-1, 0])
assert trunc and not done
for action in ([1], [0, float('nan')]):
    try:
        env.step(action)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid action accepted')
print('Passed: API contract, seeding, safe reset, forward/turn motion, swept collision, collision penalty, success bonus, timeout and invalid inputs.')
