import beaker.localnet as localnet
import beaker.client as client
import dotenv
import typing
import os

import gamemanager


def deploy():
    algorand_node_address = (
        os.environ.get("ALGORAND_NODE") or localnet.clients.DEFAULT_ALGOD_ADDRESS
    )

    owner = localnet.get_accounts()[0]

    print(f"connecting to algorand node at {algorand_node_address}")
    # Create an Application client
    app_client = client.ApplicationClient(
        # Get localnet algod client
        client=localnet.get_algod_client(address=algorand_node_address),
        # Pass instance of app to client
        app=gamemanager.app,
        # Get acct from localnet and pass the signer
        signer=owner.signer,
    )

    # Deploy the app on-chain
    app_id, app_addr, txid = app_client.create()
    print(
        f"""Deployed app in txid {txid}
        App ID: {app_id} 
        Address: {app_addr} 
    """
    )

    return app_id, app_addr


if __name__ == "__main__":
    dotenv.load_dotenv()
    print(deploy())
