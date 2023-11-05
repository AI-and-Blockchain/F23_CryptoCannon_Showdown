# KEEP IT AS IMPORT GYM SINCE THE LIBRARY DOESNT WORK WITH GYMNASIUM
import gym
# FOR SOME REASON, ADVERSARIAL BATTLESHIP DOESNT WORK, WILL TRY FIX SOON
import gym_battleship

# this is mainly for testing out the Battleship env
reward_dict = {
    'win': 100,
    'missed': 0,
    'touched': 1,
    'repeat_missed': -1,
    'repeat_touched': -0.5
}
env = gym.make('Battleship-v0', board_size=(5,5), reward_dictionary=reward_dict, episode_steps=21)
obs = env.reset()
print(obs)

for i in range(5):
    newobs, reward, done, info = env.step(env.action_space.sample())
    obs = newobs
    #print(obs)
    #print(env.reward_range)
    env.render()

# obs prints out two list of lists
# one for hit ships (top)
# one for missed ships (bottom)
print(obs)
print(reward_dict)