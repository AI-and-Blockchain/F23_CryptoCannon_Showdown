from os import path
import typing
import beaker
import beaker.application
import beaker.lib.storage
import pyteal as pt

from gamestate import GameState


MAX_SCORE = 18


class GameManagerState:
    player_board = beaker.LocalStateValue(
        stack_type=pt.TealType.bytes,
        default=pt.Bytes(b"\0" * 100),
        descr="user's current game (0 means empty, 1 means ship present, 2 means miss, 3 means hit)",
    )

    player_score = beaker.LocalStateValue(
        stack_type=pt.TealType.uint64,
        default=pt.Int(0),
        descr="Amount of hits the player has",
    )

    opponent_board = beaker.LocalStateValue(
        stack_type=pt.TealType.bytes,
        default=pt.Bytes(b"\0" * 100),
        descr="user's current game (0 means empty, 1 means ship present, 2 means miss, 3 means hit)",
    )

    opponent_score = beaker.LocalStateValue(
        stack_type=pt.TealType.uint64,
        default=pt.Int(0),
        descr="Amount of hits the opponent has",
    )

    game_status = beaker.LocalStateValue(
        stack_type=pt.TealType.uint64,
        default=pt.Int(GameState.NOT_IN_GAME.value),
        descr="Current game status",
    )


app = (
    beaker.application.Application("gamemanager", state=GameManagerState())
    .apply(beaker.unconditional_create_approval, initialize_global_state=True)
    .apply(beaker.unconditional_opt_in_approval, initialize_local_state=True)
)


@app.external(read_only=True, authorize=beaker.Authorize.opted_in())
def is_in_game(*, output: pt.abi.Bool) -> pt.Expr:
    return output.set(
        (app.state.game_status.get() == pt.Int(GameState.IN_GAME_TURN_PLAYER.value))
        or (
            app.state.game_status.get() == pt.Int(GameState.IN_GAME_TURN_OPPONENT.value)
        )
    )


def assert_in_game(adr: pt.abi.Address) -> pt.Expr:
    return pt.Assert(
        (
            app.state.game_status[adr.get()].get()
            == pt.Int(GameState.IN_GAME_TURN_PLAYER.value)
        )
        or (
            app.state.game_status[adr.get()].get()
            == pt.Int(GameState.IN_GAME_TURN_OPPONENT.value)
        )
    )


@pt.Subroutine(pt.TealType.uint64)
def only_in_game(sdr: pt.Expr) -> pt.Expr:
    return (pt.App.optedIn(sdr, pt.App.id())) and (
        app.state.game_status[sdr].get() == pt.Int(GameState.IN_GAME_TURN_PLAYER.value)
        or app.state.game_status[sdr].get()
        == pt.Int(GameState.IN_GAME_TURN_OPPONENT.value)
    )


@pt.Subroutine(pt.TealType.uint64)
def player_turn(sdr: pt.Expr) -> pt.Expr:
    return (pt.App.optedIn(sdr, pt.App.id())) and (
        app.state.game_status[sdr].get() == pt.Int(GameState.IN_GAME_TURN_PLAYER.value)
    )


@app.external(read_only=True, authorize=only_in_game)
def current_player_score(*, output: pt.abi.Uint8):
    return output.set(app.state.player_score.get())


@app.external(read_only=True, authorize=beaker.Authorize.only_creator())
def current_opponent_score(game: pt.abi.Address, *, output: pt.abi.Uint8):
    return pt.Seq(
        assert_in_game(game), output.set(app.state.opponent_score[game.get()].get())
    )


@app.external(read_only=True, authorize=beaker.Authorize.opted_in())
def new_game(
    o_ship1_pos: pt.abi.Uint8,
    o_ship1_rot: pt.abi.Bool,
    o_ship2_pos: pt.abi.Uint8,
    o_ship2_rot: pt.abi.Bool,
    o_ship3_pos: pt.abi.Uint8,
    o_ship3_rot: pt.abi.Bool,
    o_ship4_pos: pt.abi.Uint8,
    o_ship4_rot: pt.abi.Bool,
    o_ship5_pos: pt.abi.Uint8,
    o_ship5_rot: pt.abi.Bool,
) -> pt.Expr:
    return pt.Seq(
        pt.Assert(
            set_ship_positions(
                app.state.opponent_board,
                o_ship1_pos.get(),
                o_ship1_rot.get(),
                o_ship2_pos.get(),
                o_ship2_rot.get(),
                o_ship3_pos.get(),
                o_ship3_rot.get(),
                o_ship4_pos.get(),
                o_ship4_rot.get(),
                o_ship5_pos.get(),
                o_ship5_rot.get(),
            )
        ),
        app.state.game_status.set(pt.Int(1)),
        app.state.player_score.set_default(),
        app.state.opponent_score.set_default(),
    )


@app.external(authorize=beaker.Authorize.opted_in())
def submit_player_ship_positions(
    p_ship1_pos: pt.abi.Uint8,
    p_ship1_rot: pt.abi.Bool,
    p_ship2_pos: pt.abi.Uint8,
    p_ship2_rot: pt.abi.Bool,
    p_ship3_pos: pt.abi.Uint8,
    p_ship3_rot: pt.abi.Bool,
    p_ship4_pos: pt.abi.Uint8,
    p_ship4_rot: pt.abi.Bool,
    p_ship5_pos: pt.abi.Uint8,
    p_ship5_rot: pt.abi.Bool,
) -> pt.Expr:
    return pt.Seq(
        pt.Assert(
            app.state.game_status.get() == pt.Int(1),
            comment="Game is not expecting player ship positions",
        ),
        pt.Assert(
            set_ship_positions(
                app.state.player_board,
                p_ship1_pos.get(),
                p_ship1_rot.get(),
                p_ship2_pos.get(),
                p_ship2_rot.get(),
                p_ship3_pos.get(),
                p_ship3_rot.get(),
                p_ship4_pos.get(),
                p_ship4_rot.get(),
                p_ship5_pos.get(),
                p_ship5_rot.get(),
            )
        ),
        app.state.game_status.set(pt.Int(2)),
    )


def set_ship_positions(
    board: beaker.LocalStateValue,
    ship1_pos: pt.Expr,
    ship1_rot: pt.Expr,
    ship2_pos: pt.Expr,
    ship2_rot: pt.Expr,
    ship3_pos: pt.Expr,
    ship3_rot: pt.Expr,
    ship4_pos: pt.Expr,
    ship4_rot: pt.Expr,
    ship5_pos: pt.Expr,
    ship5_rot: pt.Expr,
) -> pt.Expr:
    all_ship_pos = [ship1_pos, ship2_pos, ship3_pos, ship4_pos, ship5_pos]
    all_ship_rot = [ship1_rot, ship2_rot, ship3_rot, ship4_rot, ship5_rot]
    all_ship_len = [2, 3, 4, 4, 5]
    return pt.Seq(
        *[
            pt.Assert(e < pt.Int(100), comment=f"invalid ship position of ship {i+1}")
            for i, e in enumerate(all_ship_pos)
        ],
        *[
            pt.If(
                all_ship_rot[ship_num],
                set_ship(starting_pos, True, all_ship_len[ship_num], board),
                set_ship(starting_pos, False, all_ship_len[ship_num], board),
            )
            for ship_num, starting_pos in enumerate(all_ship_pos)
        ],
        pt.Int(1),
    )


def set_ship(
    starting_pos: pt.Expr, vertical: bool, ship_len: int, board: beaker.LocalStateValue
) -> pt.Expr:
    mult = 0
    if vertical:
        mult = 10
    else:
        mult = 1

    return pt.Seq(
        *[
            pt.Seq(
                pt.Assert(
                    (starting_pos + pt.Int(i * mult)) < pt.Int(100),
                    comment="invalid ship position",
                ),
                pt.Assert(
                    pt.GetByte(board.get(), starting_pos + pt.Int(i * mult))
                    == pt.Int(0),
                    comment="ship overlaps another ship",
                ),
                board.set(
                    pt.SetByte(
                        board.get(),
                        starting_pos + pt.Int(i * mult),
                        pt.Int(1),
                    )
                ),
            )
            for i in range(ship_len)
        ]
    )


@app.external(read_only=True, authorize=beaker.Authorize.opted_in())
def current_player_board(*, output: pt.abi.StaticBytes[typing.Literal[100]]) -> pt.Expr:
    return output.set(app.state.player_board)


@app.external(read_only=True, authorize=beaker.Authorize.only_creator())
def current_opponent_board(
    game: pt.abi.Address, *, output: pt.abi.StaticBytes[typing.Literal[100]]
) -> pt.Expr:
    return pt.Seq(
        assert_in_game(game), output.set(app.state.opponent_board[game.get()])
    )


@app.external(authorize=player_turn)
def player_shoot(pos: pt.abi.Uint8, *, output: pt.abi.Bool) -> pt.Expr:
    return pt.Seq(
        output.set(shoot(pos, app.state.opponent_board, app.state.player_score)),
        pt.If(
            app.state.player_score.get() >= pt.Int(MAX_SCORE),
            app.state.game_status.set(pt.Int(3)),
        ),
    )


@app.external(authorize=beaker.Authorize.only_creator())
def opponent_shoot(
    game: pt.abi.Address, pos: pt.abi.Uint8, *, output: pt.abi.Bool
) -> pt.Expr:
    return pt.Seq(
        assert_in_game(game),
        output.set(shoot(pos, app.state.player_board, app.state.opponent_score)),
        pt.If(
            app.state.opponent_score.get() >= pt.Int(MAX_SCORE),
            app.state.game_status.set(pt.Int(4)),
        ),
    )


def shoot(
    pos: pt.abi.Uint8, board: beaker.LocalStateValue, score: beaker.LocalStateValue
) -> pt.Expr:
    return pt.Seq(
        pt.Assert(pt.GetByte(board.get(), pos.get()) <= pt.Int(1)),
        pt.If(
            pt.GetByte(board.get(), pos.get()) == pt.Int(1),
            pt.Seq(
                board.set(pt.SetByte(board.get(), pos.get(), pt.Int(3))),
                score.set(score.get() + pt.Int(1)),
                pt.Int(1),
            ),
            pt.Seq(board.set(pt.SetByte(board.get(), pos.get(), pt.Int(2))), pt.Int(0)),
        ),
    )


if __name__ == "__main__":
    import sys
    import os

    gamemanager_spec = app.build()

    path = sys.argv[0].removesuffix("/gamemanager.py")

    os.mkdir(path + "/contract_abi")
    gamemanager_spec.export(path + "/contract_abi")
