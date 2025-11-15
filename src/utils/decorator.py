from typing import Callable
from dataclasses import dataclass
from chess import Board, Move
from chess.pgn import read_game
import contextlib
import io
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


@dataclass(frozen=True)
class GameContext:
    board: Board
    timeLeft: int  # in milliseconds
    logProbabilities: Callable[[dict[Move, float]], None]


class ChessManager:
    def __init__(self):
        self._ctx = GameContext(
            board=Board(),
            timeLeft=0,
            logProbabilities=self.update_move_probabilities,
        )
        self._func = None
        self._reset_func = None
        self._logger = logging.getLogger(__name__)
        self._move_probabilities = {}

    def entrypoint(self, func: Callable[[GameContext], Move]):
        """
        The function bound to the entrypoint will be called when its time to make a move.
        The function will be passed the game context and is expected to return a Move object.
        """
        def wrapper(ctx: GameContext):
            # Forward the provided context to the model function
            return func(ctx)

        if (self._func is not None):
            raise ValueError("Entrypoint cannot be set twice")

        self._func = wrapper

        return wrapper

    def reset(self, func: Callable[[GameContext], None]):
        """
        Register a function that should run when a new game begins.
        Call `chess_manager.call_reset()` (after updating context) to invoke it.
        """

        def wrapper(ctx: GameContext):
            func(ctx)

        if (self._reset_func is not None):
            raise ValueError("Reset handler cannot be set twice")

        self._reset_func = wrapper

        return wrapper

    def set_context(self, pgn: str, timeleft: int):

        game = read_game(io.StringIO(pgn))

        if game == None:
            raise ValueError("Invalid PGN")

        # Reconstruct the board at the end of the mainline
        board_at_end = game.board()
        for move in game.mainline_moves():
            board_at_end.push(move)

        self._ctx = GameContext(
            board=board_at_end,
            timeLeft=timeleft,
            logProbabilities=self.update_move_probabilities,
        )

    def call_reset(self):
        if self._reset_func is None:
            return

        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                self._reset_func(self._ctx)
        except Exception:
            captured_output = buffer.getvalue()
            if captured_output:
                self._logger.info("Model stdout/stderr:\n%s", captured_output)
            raise

        captured_output = buffer.getvalue()
        if captured_output:
            self._logger.info("Model stdout/stderr:\n%s", captured_output)

    def get_model_move(self) -> tuple[Move, dict[Move, float], str]:

        if (self._func is None):
            raise ValueError("No entrypoint set")

        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                result = self._func(self._ctx)
        except Exception:
            captured_output = buffer.getvalue()
            if captured_output:
                self._logger.info("Model stdout/stderr:\n%s", captured_output)
            raise

        captured_output = buffer.getvalue()
        if captured_output:
            self._logger.info("Model stdout/stderr:\n%s", captured_output)

        return result, self._move_probabilities, captured_output

    def update_move_probabilities(self, probabilities: dict[Move, float]):
        self._move_probabilities = probabilities

# Lie to lsp's about type of the decorator


@dataclass
class ChessManagerType:
    entrypoint: Callable[[Callable[[GameContext], Move]],
                         Callable[[GameContext], Move]]
    reset: Callable[[Callable[[GameContext], None]],
                    Callable[[GameContext], None]]


chess_manager: ChessManagerType = ChessManager()

__all__ = ["chess_manager", "GameContext"]
