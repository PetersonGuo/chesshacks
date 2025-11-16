import io

import chess.pgn

import c_helpers
from src import engine


def build_state_from_pgn(pgn: str) -> c_helpers.BitboardState:
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None, f"Failed to parse PGN: {pgn}"
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return engine._board_to_bitboard_state(board)


def search_with_bitboard(state: c_helpers.BitboardState, depth: int = 4) -> str:
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()
    counters = c_helpers.CounterMoveTable()
    return c_helpers.get_best_move_uci_builtin_state(
        state, depth, tt, 0, killers, history, counters
    )


def test_short_white_sequence_regression():
    # Regression for sequences sent from the devtools like:
    # 1. e4 Nh6 2. d4 Rg8 3. Nf3 Rh8 4. Ne5 Rg8 5. Bxh6 Rh8 6. Nc3
    pgn = "1. e4 Nh6 2. d4 Rg8 3. Nf3 Rh8 4. Ne5 Rg8 5. Bxh6 Rh8 6. Nc3"
    state = build_state_from_pgn(pgn)
    move = search_with_bitboard(state)
    assert isinstance(move, str)
    assert len(move) in (4, 5)


def test_basic_opening_with_knight_development():
    # Regression for the reported 'e4, Nc3' failure path
    pgn = "1. e4 e5 2. Nc3"
    state = build_state_from_pgn(pgn)
    move = search_with_bitboard(state)
    assert isinstance(move, str)
    assert len(move) in (4, 5)

