import os
import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = "ppo_cartpole.zip"
TRAIN_MORE = True

if os.path.exists(MODEL_PATH):
    print("Model data found")
    if TRAIN_MORE:
        train_env = gym.make("CartPole-v1")
        model = PPO.load(MODEL_PATH, env=train_env)
        print("Training model")
        model.learn(total_timesteps=25000)
        model.save(MODEL_PATH)
        train_env.close()
        print("Training complete")
    else:
        model = PPO.load(MODEL_PATH)
else:
    print("Training model")
    train_env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save(MODEL_PATH)
    train_env.close()
    print("Training complete")

# Reset environment to start a new episode
# observation, info = env.reset()
# observation: what the agent can "see" - cart position, velocity, pole angle, etc.
# info: extra debugging information (usually not needed for basic learning)
print("Testing the training model..")
# print(f"Starting observation: {observation}")
# Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
test_env= gym.make("CartPole-v1",render_mode="human")

for episode in range(5):
    observation, _ = test_env.reset()
    episode_over = False
    total_reward = 0

    while not episode_over:

        action, _states = model.predict(observation, deterministic=True)
        # Choose an action: 0 = push cart left, 1 = push cart right
        # action = env.action_space.sample()  # Random action for now - real agents will be smarter!
        # pole_angle = observation[2]
        # if pole_angle > 0:
        #     action = 1
        # else:
        #     action = 0
    # Take the action and see what happens  
        observation, reward, terminated, truncated, info = test_env.step(action)

    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)

        total_reward += reward
        episode_over = terminated or truncated

print(f"Episode {episode + 1} finished! Total reward: {total_reward}")
test_env.close()