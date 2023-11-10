# stable baselines installation
# !apt-get install ffmpeg freeglut3-dev xvfb  # For visualization
# !pip install "stable-baselines3[extra]>=2.0.0a4"

# KEEP IT AS IMPORT GYM SINCE THE LIBRARY DOESNT WORK WITH GYMNASIUM
import gym
# FOR SOME REASON, ADVERSARIAL BATTLESHIP DOESNT WORK, WILL TRY FIX SOON
import gym_battleship
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('Battleship-v0')

model = PPO("MlpPolicy", env, verbose=0)


#Modified from https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/1_getting_started.ipynb#scrollTo=63M8mSKR-6Zt

def evaluate(model, env, num_episodes = 100, deterministic = True):
    """
    Evaluate an RL agent for `num_episodes`.

    :param model: the RL Agent
    :param num_episodes: number of episodes to evaluate it
    :param deterministic: Whether to use deterministic or stochastic actions
    :return: Mean reward for the last `num_episodes`
    """
    # This function will only work for a single environment
    #vec_env = model.get_env()
    #obs = vec_env.reset()
    #model.learn(total_timesteps = 10)
    obs = env.reset()
    all_episode_rewards = []
    #print(vec_env.action_space)
    print(env.action_space)
    for _ in range(num_episodes):
        episode_rewards = []
        done = False
        # Note: SB3 VecEnv resets automatically:
        # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            # `deterministic` is to use deterministic actions
            action, _states = model.predict(obs, deterministic=deterministic)

            print(f"action = {action}")


            action = int(action)
            obs, reward, done, _info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    print(all_episode_rewards)
    mean_episode_reward = np.mean(all_episode_rewards)
    print(f"Mean reward: {mean_episode_reward:.2f} - Num episodes: {num_episodes}")

    return mean_episode_reward


mean_reward = evaluate(model, env, num_episodes=100) #random policy

print(f"mean_reward: {mean_reward:.2f} +/- {mean_reward:.2f}")  #shoul just be random policy prior to training
