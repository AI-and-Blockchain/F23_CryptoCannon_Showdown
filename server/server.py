import sys
from flask import Flask, jsonify, request
from stable_baselines3 import PPO
import numpy as np
sys.path.append("../AI")
from battleship_enviroment import BattleshipEnv

app = Flask(__name__)

# *This will change to 10x10 grid*
ships = {}
ships['carrier'] = 5
ships['battleship'] = 4
ships['cruiser'] = 3
ships['submarine'] = 3
ships['destroyer'] = 2

grid_size = 10

env = BattleshipEnv(enemy_board=None, ship_locs={}, grid_size=grid_size, ships=ships)

# Load model
model = PPO.load("../AI/models/model.zip")
print(model)

obs, _ = env.reset()


@app.route('/get_move', methods=['GET'])
def get_move():
    global obs

    # Use the PPO model to predict the next move
    action, _ = model.predict(obs)
    action = int(action)

    #i is the row that is being hit, j is the column
    i, j = np.unravel_index(action, (grid_size,grid_size))

    i,j = int(i), int(j)

    #take next step in env so it's up to date
    obs, _, _, _, _ = env.step(action)

    # Return the move as JSON
    return jsonify({'move': (i,j)})

if __name__ == '__main__':
    app.run(debug=True)
