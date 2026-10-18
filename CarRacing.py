import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO

print("Training CarRacing....")
test_env = gym.make("CarRacing-v3"  )

# normal reset, this changes the colour scheme by default

model = PPO("CnnPolicy",test_env,verbose=1)
model.learn(total_timesteps=20000)

test_env.close()
print("Testing completed")

total_reward = 0
# reset with colour scheme change
print("Testing on screen")
test_env = gym.make("CarRacing-v3", render_mode="human")
observation, info = test_env.reset()
episode_over = False
total_reward = 0
while not episode_over:
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = test_env.step(action)
    total_reward += reward
    episode_over = terminated or truncated


# user input program 

# while not episode_over:
#     # action = env.action_space.sample()

#     # obs, reward, terminated, truncated, info = env.step(action)
    
#     # total_reward += reward
#     # episode_over = terminated or truncated
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             episode_over = True

#     steering = 0.0
#     gas = 0.0
#     brake = 0.0

#     keys = pygame.key.get_pressed()
#     if keys[pygame.K_LEFT]:
#         steering = -1.0
#     if keys[pygame.K_RIGHT]:
#         steering = 1.0
#     if keys[pygame.K_UP]:
#         gas = 0.5
#     if keys[pygame.K_DOWN]:
#         brake = 0.8

#     action = np.array([steering, gas, brake])
#     obs, reward, terminated, truncated, info = env.step(action)

#     total_reward += reward
#     episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
test_env.close()
