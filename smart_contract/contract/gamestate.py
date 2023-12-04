from enum import IntEnum


class GameState(IntEnum):
    NOT_IN_GAME = 0
    WAITING_FOR_PLAYER_SHIP_PLACEMENT = 1
    IN_GAME_TURN_PLAYER = 2
    IN_GAME_TURN_OPPONENT = 3
    WIN_PLAYER = 4
    WIN_OPPONENT = 5
