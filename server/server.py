import sys
from flask import Flask, jsonify, request, session
from flask_cors import CORS
from stable_baselines3 import PPO
import numpy as np

sys.path.append("../AI")
from battleship_enviroment import BattleshipEnv

sys.path.append("../smart_contract")
import deploy
import gamemanager
import beaker

# deploy contract and load it
contract_id, contract_address = deploy.deploy()
account = beaker.localnet.get_accounts()[0]
app_client = beaker.client.ApplicationClient(
    client=beaker.localnet.get_algod_client(),
    app=gamemanager.app,
    signer=account.signer,
    app_id=contract_id,
)


app = Flask(__name__)
CORS(app)


grid_size = 10

# Load model
model = PPO.load("../AI/models/model.zip")


# gets current ship locations in gamemanager smart contract form
def get_position_from_board(board_state):
    pass


@app.route("/setup_board", methods=["POST"])
def setup_board():
    data = request.json

    if "user_board" not in data or "ai_board" not in data:
        return (
            jsonify({"error": "Missing user_board or ai_board or both parameter"}),
            400,
        )

    user_board = data["user_board"]
    ai_board = data["ai_board"]

    user_pos, user_rot = get_position_from_board(user_board)
    ai_pos, ai_rot = get_position_from_board(ai_board)

    app_client.call(
        gamemanager.new_game,
        o_ship1_pos=user_pos[0],
        o_ship1_rot=user_rot[0],
        o_ship2_pos=user_pos[1],
        o_ship2_rot=user_rot[1],
        o_ship3_pos=user_pos[2],
        o_ship3_rot=user_rot[2],
        o_ship4_pos=user_pos[3],
        o_ship4_rot=user_rot[3],
        o_ship5_pos=user_pos[4],
        o_ship5_rot=user_rot[4],
    )

    app_client.call(
        gamemanager.submit_player_ship_positions,
        p_ship1_pos=ai_pos[0],
        p_ship1_rot=ai_rot[0],
        p_ship2_pos=ai_pos[1],
        p_ship2_rot=ai_rot[1],
        p_ship3_pos=ai_pos[2],
        p_ship3_rot=ai_rot[2],
        p_ship4_pos=ai_pos[3],
        p_ship4_rot=ai_rot[3],
        p_ship5_pos=ai_pos[4],
        p_ship5_rot=ai_rot[4],
    )


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
@app.route("/get_move", methods=["POST"])
def get_move():
    data = request.json
    # print(data['gameState'])

    if "gameState" not in data:
        return jsonify({"error": "Missing gameObject parameter"}), 400

    info_dict = data["gameState"]
    cel_info = info_dict["cells"]
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
    # print(obs)
    # Use the PPO model to predict the next move]
    action, _ = model.predict(obs)
    action = int(action)

    # i is the row that is being hit, j is the column
    i, j = np.unravel_index(action, (grid_size, grid_size))

    i, j = int(i), int(j)
    # print(i, j)

    # Return the move as JSON
    return jsonify({"move": (i, j)})


if __name__ == "__main__":
    app.run(debug=True)
