from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from battleship_enviroment import BattleshipEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import gymnasium as gym
import os
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
        self.save_path = os.path.join('./gym/', 'best_model.pkl')
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
                    self.model.save('best_model.pkl')

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


def make_env():
    ships = {}
    ships['cruiser'] = 3
    ships['carrier'] = 5
    ships['submarine'] = 3
    ships['destroyer'] = 2
    ships['battleship'] = 4
    newEnv = BattleshipEnv(enemy_board=None, ship_locs={}, grid_size=10, ships=ships)
    return newEnv

# Agent hyperparameter optimization
def objective(space):
    

    env_copies = space['env_copies']    
    num_timesteps = space['num_timesteps']
    gamma = space['gamma']
    n_steps = space['n_steps']
    vf_coef = space['vf_coef']
    ent_coef = space['ent_coef']
    max_grad_norm = space['max_grad_norm']
    learning_rate = space['learning_rate']
    alpha = space['alpha']
    epsilon = space['epsilon']
    lr_schedule = space['lr_schedule']
    
    print('space:', space)
    
    # ships
    ships = {}
    ships['cruiser'] = 3
    ships['carrier'] = 5
    ships['submarine'] = 3
    ships['destroyer'] = 2
    ships['battleship'] = 4

    log_dir = "./gym/"
    best_mean_reward, step_interval, episode_interval = -np.inf, 10000, 10000

    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, episode_interval=episode_interval, log_dir=log_dir)

    grid_size = 10

    # Instantiate the env
    env = BattleshipEnv(enemy_board=None, ship_locs={}, grid_size=grid_size, ships=ships)
    env = Monitor(env, filename=log_dir, allow_early_resets=True)
    #print(gym.envs.registry.keys())
    env = DummyVecEnv([make_env]*env_copies)

    model = PPO('MlpPolicy', env, verbose=0, 
                 gamma=gamma,
                 n_steps=n_steps,
                 ent_coef=ent_coef,
                 learning_rate=learning_rate,
                 vf_coef=vf_coef,
                 max_grad_norm=max_grad_norm,
               ).learn(total_timesteps=num_timesteps, callback=callback)

    '''
    model = A2C.load("best_model.pkl")
    model.set_env(env)
    '''
    rewards_mean = []
    moves_mean = []
    n_episodes = 100
    for ep in range(n_episodes):
        reward_env = []
        moves_env = []
        for env_i in env.envs:
            #print(env_i)
            obs, _ = env_i.reset()
            #print(obs)
            done = False
            rewards_sum = 0
            moves = 0
            while not done:
                action, obs = model.predict(obs, deterministic=True)
                obs, reward, done , _, _ = env_i.step(action)
                rewards_sum += reward # total reward for this episode
                moves += 1
            # print(done)
            reward_env.append(rewards_sum)
            moves_env.append(moves)
        rewards_mean.append(np.min(reward_env)) # avg environment reward 
        moves_mean.append(np.mean(moves_env)) # avg environment reward 
    rewards_mean = np.mean(rewards_mean)
    moves_mean = np.mean(moves_mean)

    print('reward', rewards_mean, 'moves', moves_mean)
    
    # hyperopt will minimize objective, number of moves in this case
    return{'loss': moves_mean, 'status': STATUS_OK }


space = {
    'env_copies': hp.choice('env_copies', [10]),
    'num_timesteps': hp.choice('num_timesteps', [1000000]), #np.arange(1000000, 1000001, 1000000, dtype=int)
    'gamma': hp.choice('gamma', [0.99, 0.95, 0.9]),
    'n_steps': hp.choice('n_steps', [5, 1, 10]),
    'vf_coef': hp.choice('vf_coef', [0.25, 0.1, 0.5]),
    'ent_coef': hp.choice('ent_coef', [0.01, 0.1]), 
    'learning_rate': hp.choice('learning_rate', [0.0007]),
    'max_grad_norm': hp.choice('max_grad_norm', [0.5, 0.2, 0.7]), 
    'alpha': hp.choice('lam', [0.99, 0.95, 0.9]), 
    'epsilon': hp.choice('epsilon', [1e-5, 1e-3, 1e-4]), 
    'lr_schedule': hp.choice('lr_schedule', ['constant', 'linear'])
}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=30, 
            trials=trials, verbose=1)