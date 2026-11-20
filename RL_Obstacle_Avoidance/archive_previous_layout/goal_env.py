"""Goal reaching with one fixed circular obstacle and a finite-sized robot."""

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
    OBSTACLE_RADIUS = 0.45
    ROBOT_RADIUS = 0.08

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.obstacle = np.array([0., 0.])
        self.observation_space = spaces.Box(
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
