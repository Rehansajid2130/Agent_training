import time
import gymnasium as gym
import os
from stable_baselines3 import PPO

MODEL_PATH ="ppo_lunar_lander.zip"
TRAIN_MORE = True

if os.path.exists(MODEL_PATH):
    print("Model data found")
    if TRAIN_MORE:
        train_env = gym.make("LunarLander-v3")
        model = PPO.load(MODEL_PATH,env=train_env)

        print("Training LunarLander..")
        model.learn(total_timesteps=150000)
        model.save("ppo_lunar_lander.zip")

        train_env.close()
        print("updated")
    else:
        model =PPO.load(MODEL_PATH)

else:
    print("no model found")
    train_env = gym.make("LunarLander-v3")
    model = PPO("MlpPolicy", train_env)
    model.learn(total_timesteps=15000)
    model.save(MODEL_PATH)
    train_env.close()
    print("trained and saved")





print("Showing lander")
test_env = gym.make("LunarLander-v3",render_mode="human")

for episode in range(5):
    observation,info = test_env.reset()
    episode_over = False
    total_reward = 0

    while not episode_over:
        action, _states = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        episode_over = terminated or truncated 
    
    print(f"Total reward: {total_reward}")
    time.sleep(1)
test_env.close()