import sys
from flask import Flask, jsonify, request, session
from flask_cors import CORS
from stable_baselines3 import PPO
import numpy as np
sys.path.append("../AI")
from battleship_enviroment import BattleshipEnv

app = Flask(__name__)
CORS(app)


grid_size = 10

# Load model
model = PPO.load("../AI/models/model.zip")


# Frontend must send post request with obs of current game state so that the model can make best move
# EXAMPLE:
# obs= [[0, 0, -1, 0, 0, 0, 0, 0, -1, 0],
#       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, -1, 0, 0, 0, 0, -1, 0],
#       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
# Where 0 is a unknown spot, -1 is a miss, and 1 is a hit
#
# returned action is the index of where the model wants to hit
@app.route('/get_move', methods=['POST'])
def get_move():

    data = request.json
    #print(data['gameState'])

    if 'gameState' not in data:
        return jsonify({'error': 'Missing gameObject parameter'}), 400
    
    info_dict = data['gameState']
    cel_info = info_dict['cells']
    obs = []
    for cel in cel_info:
        inner_obs = []
        for num in cel:
            if num == 1:
                num = 0
                inner_obs.append(num)
            elif num == 2:
                num = -1
                inner_obs.append(num)
            elif num == 3:
                num = 1
                inner_obs.append(num)
            elif num == 4:
                num = 1
                inner_obs.append(num)
            else:
                inner_obs.append(num)
        obs.append(inner_obs)
    #print(obs)
    # Use the PPO model to predict the next move]
    action, _ = model.predict(obs)
    action = int(action)

    #i is the row that is being hit, j is the column
    i, j = np.unravel_index(action, (grid_size,grid_size))

    i,j = int(i), int(j)
    #print(i, j)

    # Return the move as JSON
    return jsonify({'move': (i,j)})

if __name__ == '__main__':
    app.run(debug=True)
