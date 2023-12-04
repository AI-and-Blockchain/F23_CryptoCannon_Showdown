import algosdk
import algosdk.atomic_transaction_composer
import dotenv
import os
import sys
import gamemanager
import beaker


def test(address: str, private_key: str, app_id: int, contract_addr: str):
    algod_client = algosdk.v2client.algod.AlgodClient(
        os.environ.get("ALGORAND_TOKEN") or "a" * 64,
        os.environ.get("ALGORAND_NODE") or "http://localhost:4001",
    )

    print("Address:", address)
    print("Private key:", private_key)

    app_client = beaker.client.ApplicationClient(
        client=algod_client,
        app=gamemanager.app,
        signer=algosdk.atomic_transaction_composer.AccountTransactionSigner(
            private_key
        ),
        app_id=app_id,
    )

    app_client.opt_in(address)

    print(
        app_client.call(
            gamemanager.is_in_game,
        ).return_value
    )
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

    # p_ret = app_client.call(gamemanager.current_player_board).return_value

    print(
        app_client.call(
            gamemanager.current_player_score,
        ).return_value
    )

    print(
        app_client.call(
            gamemanager.player_shoot,
            pos=0,
        ).return_value
    )

    print(
        app_client.call(
            gamemanager.player_shoot,
            pos=55,
        ).return_value
    )

    print(
        app_client.call(
            gamemanager.current_player_score,
        ).return_value
    )

    print_board(
        app_client.call(
            gamemanager.current_opponent_board,
        ).return_value
    )


def print_board(*boards: list[int], labels: list[str] = []):
    for label in labels:
        print(f"{label}\t\t\t", end="")
    print()

    for i in range(10):
        rows = [[] for _ in range(len(boards))]
        for j in range(10):
            for b in range(len(boards)):
                rows[b].append(f"{boards[b][i * 10 + j]} ")

        for row in rows:
            for c in row:
                print(c, end="")
            print("\t", end="")
        print()


if __name__ == "__main__":
    dotenv.load_dotenv()

    import deploy

    app_id, contract_addr = deploy.deploy()

    account = beaker.localnet.get_accounts()[1]

    test(account.address, account.private_key, app_id, contract_addr)
