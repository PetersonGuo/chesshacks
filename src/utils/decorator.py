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
            # Debug: Get original stderr before any potential redirects
            import sys
            original_stderr = sys.__stderr__ if hasattr(sys, '__stderr__') else sys.stderr
            
            original_stderr.write(f"[WRAPPER] wrapper() called with context\n")
            original_stderr.write(f"[WRAPPER] Context FEN: {ctx.board.fen()}\n")
            original_stderr.write(f"[WRAPPER] About to call func: {func}\n")
            original_stderr.flush()
            
            # Forward the provided context to the model function
            try:
                original_stderr.write(f"[WRAPPER] Calling func(ctx) now...\n")
                original_stderr.flush()
                result = func(ctx)
                original_stderr.write(f"[WRAPPER] func returned: {result}\n")
                original_stderr.flush()
            except Exception as e:
                original_stderr.write(f"[WRAPPER] Exception in func: {type(e).__name__}: {e}\n")
                import traceback
                original_stderr.write(f"[WRAPPER] Traceback:\n{traceback.format_exc()}\n")
                original_stderr.flush()
                raise
            
            return result

        if (self._func is not None):
            raise ValueError("Entrypoint cannot be set twice")

        self._func = wrapper
        
        # Debug
        import sys
        sys.stderr.write(f"[ENTRYPOINT] Registered entrypoint wrapper for function: {func}\n")
        sys.stderr.flush()

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

        # Debug: Write to actual stderr before redirecting
        import sys
        sys.stderr.write(f"[CHESS_MANAGER] get_model_move() called, about to invoke entrypoint\n")
        sys.stderr.write(f"[CHESS_MANAGER] Entrypoint function: {self._func}\n")
        sys.stderr.write(f"[CHESS_MANAGER] Context board FEN: {self._ctx.board.fen()}\n")
        sys.stderr.flush()

        buffer = io.StringIO()
        try:
            sys.stderr.write(f"[CHESS_MANAGER] About to redirect stdout/stderr and call entrypoint...\n")
            sys.stderr.flush()
            
            # Note: stderr writes INSIDE the redirect context will be captured
            # So we write to the original stderr before entering
            original_stderr = sys.stderr
            sys.stderr.write(f"[CHESS_MANAGER] About to enter redirect context...\n")
            sys.stderr.flush()
            
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                # These writes will be captured, but we can still use original_stderr
                original_stderr.write(f"[CHESS_MANAGER] Inside redirect context, calling self._func(self._ctx)...\n")
                original_stderr.flush()
                result = self._func(self._ctx)
                original_stderr.write(f"[CHESS_MANAGER] Entrypoint returned: {result}\n")
                original_stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[CHESS_MANAGER] Exception in entrypoint: {type(e).__name__}: {e}\n")
            sys.stderr.flush()
            import traceback
            sys.stderr.write(f"[CHESS_MANAGER] Traceback:\n{traceback.format_exc()}\n")
            sys.stderr.flush()
            captured_output = buffer.getvalue()
            if captured_output:
                self._logger.info("Model stdout/stderr:\n%s", captured_output)
            raise

        sys.stderr.write(f"[CHESS_MANAGER] Entrypoint completed successfully\n")
        sys.stderr.flush()
        
        captured_output = buffer.getvalue()
        if captured_output:
            self._logger.info("Model stdout/stderr:\n%s", captured_output)
            sys.stderr.write(f"[CHESS_MANAGER] Captured output length: {len(captured_output)} chars\n")
            sys.stderr.write(f"[CHESS_MANAGER] First 500 chars of captured output:\n{captured_output[:500]}\n")
            sys.stderr.flush()

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
