import os
import time
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "ppo_carracing.zip")

# =============================================================
# CHOOSE YOUR MODE:
# "WATCH"  -> Watch the trained AI CNN drive on screen
# "MANUAL" -> Drive yourself with arrow keys (Up=Gas, Down=Brake, Left/Right=Steer)
# "TRAIN"  -> Train the AI with 4 parallel tracks at high FPS
# =============================================================
MODE = "WATCH"

TIMESTEPS_TO_TRAIN = 100000  # Steps to train if MODE == "TRAIN"
NUM_PARALLEL_ENVS = 4        # Parallel environments for fast CPU training
EVAL_EPISODES = 3            # Number of laps to run in WATCH / MANUAL mode

class AntiSuicideCarRacingWrapper(gym.Wrapper):
    """
    1. Skips the blind zoom-in animation at the start.
    2. Penalizes driving into the black border with -100 points
       so the AI cannot commit suicide to protect its score.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
       
        no_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(40):
            obs, _, _, _, info = self.env.step(no_action)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
       
        if terminated:
            reward -= 100.0
        return obs, reward, terminated, truncated, info


def make_fixed_env(render_mode=None):
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    return AntiSuicideCarRacingWrapper(env)


def main():
    if MODE == "TRAIN":
        print(f"🏎️ Launching parallel training across {NUM_PARALLEL_ENVS} tracks (with Anti-Suicide penalty)...")
        train_env = make_vec_env(lambda: make_fixed_env(), n_envs=NUM_PARALLEL_ENVS)

        if os.path.exists(MODEL_PATH):
            print(f"Found existing model at {MODEL_PATH}! Resuming training...")
            model = PPO.load(MODEL_PATH, env=train_env)
        else:
            print("No saved model found. Initializing fresh CnnPolicy...")
            model = PPO("CnnPolicy", train_env, verbose=1)

        print(f"Training for {TIMESTEPS_TO_TRAIN} timesteps...")
        model.learn(total_timesteps=TIMESTEPS_TO_TRAIN)
        model.save(MODEL_PATH)
        train_env.close()
        print("Training finished and model saved successfully!")

    else:
        # WATCH or MANUAL Mode
        model = None
        if MODE == "WATCH":
            if os.path.exists(MODEL_PATH):
                print(f"Loading trained AI model from {MODEL_PATH}...")
                model = PPO.load(MODEL_PATH)
            else:
                print("No trained model found! Please set MODE = 'TRAIN' first.")
                return

        print(f"Starting CarRacing on screen in {MODE} mode...")
        test_env = make_fixed_env(render_mode="human")

        for episode in range(EVAL_EPISODES):
            observation, info = test_env.reset()
            episode_over = False
            total_reward = 0

            while not episode_over:
                # Keep window responsive and handle close button
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        episode_over = True

                if MODE == "MANUAL":
                    # Read keyboard inputs
                    keys = pygame.key.get_pressed()
                    steering = 0.0
                    gas = 0.0
                    brake = 0.0
                    if keys[pygame.K_LEFT]:
                        steering = -1.0
                    if keys[pygame.K_RIGHT]:
                        steering = 1.0
                    if keys[pygame.K_UP]:
                        gas = 0.6
                    if keys[pygame.K_DOWN]:
                        brake = 0.8
                    action = np.array([steering, gas, brake], dtype=np.float32)
                else:
                    # AI drives using its CNN policy
                    action, _ = model.predict(observation, deterministic=True)

                observation, reward, terminated, truncated, info = test_env.step(action)
                total_reward += reward
                episode_over = terminated or truncated

            print(f"Lap {episode + 1} finished! Score: {total_reward:.1f}")
            time.sleep(1)

        test_env.close()


if __name__ == "__main__":
    main()
