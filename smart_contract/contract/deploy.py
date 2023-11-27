import beaker.localnet as localnet
import beaker.client as client
import dotenv
import typing
import os

import gamemanager


def deploy() -> None:
    algorand_node_address = (
        os.environ.get("ALGORAND_NODE") or localnet.clients.DEFAULT_ALGOD_ADDRESS
    )

    account = localnet.get_accounts()[0]

    print(f"connecting to algorand node at {algorand_node_address}")
    # Create an Application client
    app_client = client.ApplicationClient(
        # Get localnet algod client
        client=localnet.get_algod_client(address=algorand_node_address),
        # Pass instance of app to client
        app=gamemanager.app,
        # Get acct from localnet and pass the signer
        signer=account.signer,
    )

    # Deploy the app on-chain
    app_id, app_addr, txid = app_client.create()
    print(
        f"""Deployed app in txid {txid}
        App ID: {app_id} 
        Address: {app_addr} 
    """
    )

    app_client.opt_in(sender=account.address)

    print(app_client.call(gamemanager.is_in_game).return_value)
    app_client.call(
        gamemanager.new_game,
        o_ship1_pos=0,
        o_ship1_rot=True,
        o_ship2_pos=11,
        o_ship2_rot=False,
        o_ship3_pos=35,
        o_ship3_rot=False,
        o_ship4_pos=48,
        o_ship4_rot=True,
        o_ship5_pos=92,
        o_ship5_rot=False,
    )

    app_client.call(
        gamemanager.submit_player_ship_positions,
        p_ship1_pos=5,
        p_ship1_rot=True,
        p_ship2_pos=6,
        p_ship2_rot=False,
        p_ship3_pos=35,
        p_ship3_rot=False,
        p_ship4_pos=63,
        p_ship4_rot=False,
        p_ship5_pos=50,
        p_ship5_rot=True,
    )

    p_ret = app_client.call(gamemanager.current_player_board).return_value
    o_ret = app_client.call(gamemanager.current_opponent_board).return_value

    print("player \t\t\t opponent")
    for i in range(10):
        player_row = ""
        opponent_row = ""
        for j in range(10):
            player_row += f"{p_ret[i * 10 + j]} "
            opponent_row += f"{o_ret[i * 10 + j]} "
        print(f"{player_row} \t {opponent_row}")


if __name__ == "__main__":
    dotenv.load_dotenv()
    deploy()
