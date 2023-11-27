from flask import Flask, jsonify, request
from stable_baselines3 import PPO
import numpy as np
from AI.battleship_enviroment import BattleshipEnv

app = Flask(__name__)

# *This will change to 10x10 grid*
ships = {}
ships['cruiser'] = 3

grid_size = 5

env = BattleshipEnv(enemy_board=None, ship_locs={}, grid_size=grid_size, ships=ships)

# Load model
model = PPO.load("path/to/your/model.zip")


@app.route('/get_move', methods=['POST'])
def get_move():

    #request should send the user's previous move
    # Get the game state from the request
    data = request.json
    if 'prevMove' not in data:
        return jsonify({'error': 'Missing prevMove parameter'}), 400

    prev_move = data['prevMove']

    obs, _, _, _, _ = env.step(prev_move)

    # Use the PPO model to predict the next move
    action, _ = model.predict(obs)

    #take next step in env so it's up to date
    obs, _, _, _, _ = env.step(action)

    # Return the move as JSON
    return jsonify({'move': action})

if __name__ == '__main__':
    app.run(debug=True)
