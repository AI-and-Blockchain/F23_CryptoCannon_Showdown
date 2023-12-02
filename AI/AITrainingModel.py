import gym
import stable_baselines3
from gym import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from battleship_enviroment import BattleshipEnv
from train_model import evaluate 

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, episode_interval: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_interval = episode_interval
        self.log_dir = log_dir
        self.save_path = os.path.join('./gym/', 'best_model5x5.pkl')
        print("SAVE PATH: ", self.save_path)
        self.best_mean_reward = -np.inf


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # NOTE: when done is True, timesteps are counted and reported to the log_dir
                mean_reward = np.mean(y[-self.episode_interval:]) # mean reward over previous episode_interval episodes
                mean_moves = np.mean(np.diff(x[-self.episode_interval:])) # mean moves over previous 100 episodes
                if self.verbose > 0:
                    print(x[-1], 'timesteps') # closest to step_interval step number
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f} - Last mean moves per episode: {:.2f}".format(self.best_mean_reward, 
                                                                                                   mean_reward, mean_moves))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model")
                    self.model.save('best_model5x5.pkl')

        return True


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, window = 100, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=window)
    y_moves = moving_average(np.diff(x), window = window) 
    # Truncate x
    x = x[len(x) - len(y):]
    x_moves = x[len(x) - len(y_moves):]

    title = 'Smoothed Learning Curve of Rewards (every ' + str(window) +' steps)'
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()

    title = 'Smoothed Learning Curve of Moves (every ' + str(window) +' steps)'
    fig = plt.figure(title)
    plt.plot(x_moves, y_moves)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Moves')
    plt.title(title)
    plt.show()


# https://towardsdatascience.com/an-artificial-intelligence-learns-to-play-battleship-ebd2cf9adb01#89b2
# ships -- keep only one kind for 5x5 grid
ships = {}
ships['carrier'] = 5
ships['battleship'] = 4
ships['cruiser'] = 3
ships['submarine'] = 3
ships['destroyer'] = 2

num_timesteps = 100
grid_size = 10
num_timesteps = 5000000 # this is number of moves and not number of episodes


best_mean_reward, n_steps, step_interval, episode_interval = -np.inf, 0, 10000, 10000

# Instantiate the env
env = BattleshipEnv(enemy_board=None, ship_locs={}, grid_size=grid_size, ships=ships)


# wrap it
log_dir = "./gym/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, filename=log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

callback = SaveOnBestTrainingRewardCallback(check_freq=100000, episode_interval=episode_interval, log_dir=log_dir)

#print(stable_baselines3.__version__)
# Train the agent - Note: best model is not save in Callback function for PPO2; save manually
model = PPO('MlpPolicy', env, verbose=0)

#See how well the model improves by learning
evaluate(model, env)
model.learn(total_timesteps=num_timesteps, callback=callback)
evaluate(model, env)
plot_results(log_dir, 1000)

model.save("./models/model")
#plot_results(log_dir,1000)
