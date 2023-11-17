import beaker
import beaker.application
import beaker.lib.storage
import pyteal as pt

class GameManagerState:
    current_game = beaker.LocalStateValue(
        stack_type=pt.TealType.uint64,
        default= beaker.lib.storage.BoxList(pt.abi.Uint8, 100, "tiles").create(),
        descr="In-progress game for a user"
    )
    in_game = beaker.LocalStateValue(
        stack_type=pt.TealType.uint64,
        default=pt.Int(0),
        descr="If the user is currently in a game"
    )


app = beaker.application.Application("gamemanager", state=GameManagerState())

@app.external()
def is_in_game(output: pt.abi.Bool) -> pt.Expr:
    return output.set(app.state.in_game.get())


if __name__ == "__main__":
    gamemanager_spec = app.build()
    print(gamemanager_spec.to_json())
