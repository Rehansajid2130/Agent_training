"""Minimal differential-drive-style goal reaching on an obstacle-free 2D plane."""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GoalEnv(gym.Env):
    DT = 0.1
    MAX_SPEED = 0.8
    MAX_TURN = 2.5
    GOAL_RADIUS = 0.15
    MAX_STEPS = 250

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            np.array([0, -1, -1], dtype=np.float32),
            np.array([np.inf, 1, 1], dtype=np.float32))

    def observation(self):
        offset = self.goal - self.position
        distance = float(np.linalg.norm(offset))
        bearing = math.atan2(offset[1], offset[0]) - self.heading
        return np.array([distance / 6.0, math.sin(bearing), math.cos(bearing)], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.np_random.uniform(-2, 2, size=2)
        self.goal = self.np_random.uniform(-2, 2, size=2)
        while np.linalg.norm(self.goal - self.position) < 0.8:
            self.goal = self.np_random.uniform(-2, 2, size=2)
        self.heading = float(self.np_random.uniform(-math.pi, math.pi))
        self.steps, self.path_length = 0, 0.0
        self.initial_distance = float(np.linalg.norm(self.goal - self.position))
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
        self.position += travel * np.array([math.cos(self.heading), math.sin(self.heading)])
        self.heading = (self.heading + turn*self.DT + math.pi) % (2*math.pi) - math.pi
        self.steps += 1
        self.path_length += travel
        distance = float(np.linalg.norm(self.goal - self.position))
        success = distance <= self.GOAL_RADIUS
        reward = 10 * (old_distance - distance) - 0.01 + (10.0 if success else 0.0)
        info = dict(is_success=success, distance=distance, path_length=self.path_length,
                    initial_distance=self.initial_distance)
        return self.observation(), float(reward), success, self.steps >= self.MAX_STEPS and not success, info


def rule_action(observation):
    """Turn toward the goal and reduce forward speed when misaligned or close."""
    distance, sine, cosine = map(float, observation)
    angle = math.atan2(sine, cosine)
    speed_fraction = min(1.0, distance*6.0) * max(0.0, cosine)
    return np.array([2*speed_fraction - 1, np.clip(2*angle / GoalEnv.MAX_TURN, -1, 1)],
                    dtype=np.float32)
