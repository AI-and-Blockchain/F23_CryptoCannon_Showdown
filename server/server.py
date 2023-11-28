import sys
from flask import Flask, jsonify, request, session
from stable_baselines3 import PPO
import numpy as np
sys.path.append("../AI")
from battleship_enviroment import BattleshipEnv

app = Flask(__name__)


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

    if 'gameState' not in data:
        return jsonify({'error': 'Missing gameState parameter'}), 400
    
    obs = data['gameState']

    # Use the PPO model to predict the next move
    action, _ = model.predict(obs)
    action = int(action)

    #i is the row that is being hit, j is the column
    i, j = np.unravel_index(action, (grid_size,grid_size))

    i,j = int(i), int(j)

    # Return the move as JSON
    return jsonify({'move': (i,j)})

if __name__ == '__main__':
    app.run(debug=True)
