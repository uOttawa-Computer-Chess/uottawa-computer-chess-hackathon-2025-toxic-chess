"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
import chess
from chess.engine import PlayResult, Limit
import random
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE
import logging


# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""


# Bot names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


class ComboEngine(ExampleEngine):
    """
    Get a move using multiple different methods.

    This engine demonstrates how one can use `time_limit`, `draw_offered`, and `root_moves`.
    """

    def search(self,
               board: chess.Board,
               time_limit: Limit,
               ponder: bool,  # noqa: ARG002
               draw_offered: bool,
               root_moves: MOVE) -> PlayResult:
        """
        Choose a move using multiple different methods.

        :param board: The current position.
        :param time_limit: Conditions for how long the engine can search (e.g. we have 10 seconds and search up to depth 10).
        :param ponder: Whether the engine can ponder after playing a move.
        :param draw_offered: Whether the bot was offered a draw.
        :param root_moves: If it is a list, the engine should only play a move that is in `root_moves`.
        :return: The move to play.
        """
        if isinstance(time_limit.time, int):
            my_time = time_limit.time
            my_inc = 0
        elif board.turn == chess.WHITE:
            my_time = time_limit.white_clock if isinstance(time_limit.white_clock, int) else 0
            my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, int) else 0
        else:
            my_time = time_limit.black_clock if isinstance(time_limit.black_clock, int) else 0
            my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, int) else 0

        possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)

        if my_time / 60 + my_inc > 10:
            # Choose a random move.
            move = random.choice(possible_moves)
        else:
            # Choose the first move alphabetically in uci representation.
            possible_moves.sort(key=str)
            move = possible_moves[0]
        return PlayResult(move, None, draw_offered=draw_offered)

    
class MyBot(ExampleEngine):
    """Template code for hackathon participants to modify.

    This is intentionally a very small, simple, and weak example engine
    meant for learning and quick prototyping only.

    Key limitations:
    - Fixed-depth search with only a very naive time-to-depth mapping (no true time management).
    - Plain minimax: no alpha-beta pruning, so the search is much slower than it
      could be for the same depth.
    - No iterative deepening: the engine does not progressively deepen and use PV-based ordering.
    - No move ordering or capture heuristics: moves are searched in arbitrary order.
    - No transposition table or caching: repeated positions are re-searched.
    - Evaluation is material-only and very simplistic; positional factors are ignored.

    Use this as a starting point: replace minimax with alpha-beta, add
    iterative deepening, quiescence search, move ordering (MVV/LVA, history),
    transposition table, and a richer evaluator to make it competitive.
    """

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        # NOTE: The sections below are intentionally simple to keep the example short.
        # They demonstrate the structure of a search but also highlight the engine's
        # weaknesses (fixed depth, naive time handling, no pruning, no quiescence, etc.).

        # --- very simple time-based depth selection (naive) ---
        # Expect args to be (time_limit: Limit, ponder: bool, draw_offered: bool, root_moves: MOVE)
        time_limit = args[0] if (args and isinstance(args[0], Limit)) else None
        my_time = my_inc = None
        if time_limit is not None:
            if isinstance(time_limit.time, (int, float)):
                my_time = time_limit.time
                my_inc = 0
            elif board.turn == chess.WHITE:
                my_time = time_limit.white_clock if isinstance(time_limit.white_clock, (int, float)) else 0
                my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, (int, float)) else 0
            else:
                my_time = time_limit.black_clock if isinstance(time_limit.black_clock, (int, float)) else 0
                my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, (int, float)) else 0

        # Map a rough time budget to a coarse fixed depth.
        # Examples:
        # - >= 60s: depth 4
        # - >= 20s: depth 3
        # - >= 5s:  depth 2
        # - else:   depth 1
        remaining = my_time if isinstance(my_time, (int, float)) else None
        inc = my_inc if isinstance(my_inc, (int, float)) else 0
        budget = (remaining or 0) + 2 * inc  # crude increment bonus
        if remaining is None:
            total_depth = 4
        elif budget >= 60:
            total_depth = 4
        elif budget >= 20:
            total_depth = 3
        elif budget >= 5:
            total_depth = 2
        else:
            total_depth = 1
        total_depth = max(1, int(total_depth))

        # --- simple material evaluator (White-positive score) ---
        def evaluate(b: chess.Board) -> int:
            # Large score for terminal outcomes
            if b.is_game_over():
                outcome = b.outcome()
                if outcome is None or outcome.winner is None:
                    return 0  # draw
                return 10_000_000 if outcome.winner is chess.WHITE else -10_000_000

            values = {
                chess.PAWN: 100,
                chess.KNIGHT: 320,
                chess.BISHOP: 330,
                chess.ROOK: 500,
                chess.QUEEN: 900,
                chess.KING: 0,  # king material ignored (checkmates handled above)
            }
            score = 0
            for pt, v in values.items():
                score += v * (len(b.pieces(pt, chess.WHITE)) - len(b.pieces(pt, chess.BLACK)))
            return score

        # --- plain minimax (no alpha-beta) ---
        def minimax(b: chess.Board, depth: int, maximizing: bool) -> int:
            if depth == 0 or b.is_game_over():
                return evaluate(b)

            if maximizing:
                best = -10**12
                for m in b.legal_moves:
                    b.push(m)
                    val = minimax(b, depth - 1, False)
                    b.pop()
                    if val > best:
                        best = val
                return best
            else:
                best = 10**12
                for m in b.legal_moves:
                    b.push(m)
                    val = minimax(b, depth - 1, True)
                    b.pop()
                    if val < best:
                        best = val
                return best

        # --- root move selection ---
        legal = list(board.legal_moves)
        if not legal:
            # Should not happen during normal play; fall back defensively
            return PlayResult(random.choice(list(board.legal_moves)), None)

        maximizing = board.turn == chess.WHITE
        best_move = None
        best_eval = -10**12 if maximizing else 10**12

        # Lookahead depth chosen by the simple time heuristic; subtract one for the root move
        for m in legal:
            board.push(m)
            val = minimax(board, total_depth - 1, not maximizing)
            board.pop()

            if maximizing and val > best_eval:
                best_eval, best_move = val, m
            elif not maximizing and val < best_eval:
                best_eval, best_move = val, m

        # Fallback in rare cases (shouldn't trigger)
        if best_move is None:
            best_move = legal[0]

        return PlayResult(best_move, None)


import chess
import time
import random
from collections.abc import Sequence
from typing import TypeAlias

# --- Type Aliases (from original context) ---
MOVE: TypeAlias = chess.Move
PlayResult: TypeAlias = tuple[MOVE, None]
Limit: TypeAlias = object
ExampleEngine: TypeAlias = object
HOMEMADE_ARGS_TYPE: TypeAlias = Sequence[Limit | bool | list[MOVE]]

# --- Transposition Table Flags ---
EXACT_SCORE = 0
LOWER_BOUND = 1 # Failed low (alpha was raised), score is at least this good
UPPER_BOUND = 2 # Failed high (beta cutoff), score is at most this good

# --- Evaluation Constants & Piece-Square Tables (Middlegame & Endgame) ---
# Tapered evaluation interpolates between middlegame and endgame scores.

piece_values_mg = { chess.PAWN: 82, chess.KNIGHT: 337, chess.BISHOP: 365, chess.ROOK: 477, chess.QUEEN: 1025, chess.KING: 0 }
piece_values_eg = { chess.PAWN: 94, chess.KNIGHT: 281, chess.BISHOP: 297, chess.ROOK: 512, chess.QUEEN: 936, chess.KING: 0 }

# These tables are now doubled: one for middlegame, one for endgame.
# Sourced from well-known engine configurations.
pawn_mg = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
]
pawn_eg = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]
# ... (Similar MG/EG tables for Knight, Bishop, Rook, Queen, King) ...
# To keep the code concise, I will generate them programmatically from one set.
# In a real engine, these would be meticulously tuned separately.
def generate_simple_eg_table(mg_table): return [int(i * 0.8) for i in mg_table]
knight_mg = [-167, -89, -34, -49,  61, -97, -15, -107] # Simplified one-rank representation
bishop_mg = [ -29,   4, -82, -37, -25, -42,   7,  -8]
rook_mg   = [  32,  42,  32,  51,  63,   9,  31,  43]
queen_mg  = [ -28,   0,  29,  12,  59,  44,  43,  45]
king_mg   = [ -65,  23,  16, -15, -56, -34,   2,  13]
king_eg   = [ -74, -35, -18, -18, -11,  15,   4, -17]

def expand_table(rank): return [v for v in rank for _ in range(8)]
knight_mg_full, bishop_mg_full, rook_mg_full, queen_mg_full, king_mg_full = map(expand_table, [knight_mg, bishop_mg, rook_mg, queen_mg, king_mg])
king_eg_full = expand_table(king_eg)

knight_eg_full, bishop_eg_full, rook_eg_full, queen_eg_full = map(generate_simple_eg_table, [knight_mg_full, bishop_mg_full, rook_mg_full, queen_mg_full])

pst_mg = { chess.PAWN: pawn_mg, chess.KNIGHT: knight_mg_full, chess.BISHOP: bishop_mg_full, chess.ROOK: rook_mg_full, chess.QUEEN: queen_mg_full, chess.KING: king_mg_full }
pst_eg = { chess.PAWN: pawn_eg, chess.KNIGHT: knight_eg_full, chess.BISHOP: bishop_eg_full, chess.ROOK: rook_eg_full, chess.QUEEN: queen_eg_full, chess.KING: king_eg_full }

# --- Advanced Engine Class ---
class MyBot(ExampleEngine):
    def __init__(self, max_depth=64):
        self.transposition_table = {}
        self.MAX_DEPTH = max_depth
        self.killer_moves = [[None, None] for _ in range(max_depth)]
        self.history_scores = [[0] * 64 for _ in range(12)] # [piece][to_square]
        self.pv_table = {}
        self.nodes_searched = 0
        self.start_time = 0
        self.time_limit = 0

    def get_time_budget(self, board: chess.Board, time_limit: Limit) -> float:
        if isinstance(time_limit.time, (int, float)): return time_limit.time / 30
        if board.turn == chess.WHITE:
            my_time = time_limit.white_clock
            my_inc = time_limit.white_inc
        else:
            my_time = time_limit.black_clock
            my_inc = time_limit.black_inc
        return (my_time / 40) + (my_inc * 0.75)

    def evaluate(self, board: chess.Board) -> int:
        if board.is_checkmate(): return -30000
        if board.is_game_over(): return 0

        # Game Phase calculation (0=Endgame, 24=Middlegame)
        phase = sum(piece_values_mg[pt] > 400 for color in chess.COLORS for pt in piece_values_mg for _ in board.pieces(pt, color))
        phase = min(phase, 24)

        score_mg, score_eg = 0, 0
        
        for piece_type in piece_values_mg:
            for color in chess.COLORS:
                sign = 1 if color == board.turn else -1
                for square in board.pieces(piece_type, color):
                    # Material score
                    score_mg += sign * piece_values_mg[piece_type]
                    score_eg += sign * piece_values_eg[piece_type]

                    # Positional Score
                    idx = square if color == chess.WHITE else chess.square_mirror(square)
                    score_mg += sign * pst_mg[piece_type][idx]
                    score_eg += sign * pst_eg[piece_type][idx]
        
        # Tapered score
        final_score = (score_mg * phase + score_eg * (24 - phase)) // 24
        
        # Additional terms can be added here (pawn structure, mobility, etc.)
        return final_score

    def order_moves(self, board: chess.Board, depth: int) -> list[chess.Move]:
        scores = {}
        # 1. PV Move (from previous iteration)
        pv_move = self.pv_table.get(chess.zobrist_hash(board))
        if pv_move: scores[pv_move] = 100000

        for move in board.legal_moves:
            if move in scores: continue
            score = 0
            # 2. Captures (MVV/LVA)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score = piece_values_mg[victim.piece_type] - piece_values_mg[attacker.piece_type] + 50000
            else:
                # 3. Killer Moves
                if move in self.killer_moves[depth]:
                    score = 25000 + self.killer_moves[depth].index(move) # Prefer first killer
                # 4. History Heuristic
                else:
                    piece = board.piece_at(move.from_square)
                    if piece:
                        score = self.history_scores[piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)][move.to_square]
            scores[move] = score

        return sorted(scores, key=scores.get, reverse=True)

    def pvs_search(self, board: chess.Board, depth: int, alpha: float, beta: float, is_pv_node: bool) -> tuple[int, chess.Move | None]:
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError

        self.nodes_searched += 1

        if depth <= 0:
            return self.evaluate(board), None

        board_hash = chess.zobrist_hash(board)
        if board_hash in self.transposition_table:
            entry = self.transposition_table[board_hash]
            if entry['depth'] >= depth:
                score, flag, move = entry['score'], entry['flag'], entry['move']
                if flag == EXACT_SCORE: return score, move
                if flag == LOWER_BOUND: alpha = max(alpha, score)
                elif flag == UPPER_BOUND: beta = min(beta, score)
                if alpha >= beta: return score, move

        best_move = None
        
        # Null Move Pruning
        if not is_pv_node and not board.is_check() and depth >= 3 and any(board.pieces(pt, board.turn) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]):
            board.push(chess.Move.null())
            null_score, _ = self.pvs_search(board, depth - 3, -beta, -beta + 1, False)
            null_score = -null_score
            board.pop()
            if null_score >= beta:
                return beta, None

        moves_searched = 0
        ordered_moves = self.order_moves(board, depth)
        best_move = ordered_moves[0] if ordered_moves else None

        for move in ordered_moves:
            # Late Move Reductions (LMR)
            reduction = 0
            if depth >= 3 and moves_searched > (3 if is_pv_node else 1) and not board.is_capture(move) and not move.promotion:
                reduction = 1 + (moves_searched > 5) + (moves_searched > 10)
                reduction = min(reduction, depth - 1)

            board.push(move)
            
            # Principal Variation Search (PVS) logic
            if moves_searched == 0: # First move is the PV move
                score, _ = self.pvs_search(board, depth - 1, -beta, -alpha, True)
            else:
                # Zero-window search for non-PV moves
                score, _ = self.pvs_search(board, depth - 1 - reduction, -alpha - 1, -alpha, False)
                if score > alpha and score < beta: # Re-search if it fails high
                    score, _ = self.pvs_search(board, depth - 1, -beta, -alpha, True)
            
            score = -score
            board.pop()
            moves_searched += 1

            if score > alpha:
                alpha = score
                best_move = move
                self.pv_table[board_hash] = move
                if alpha >= beta:
                    # Beta cutoff
                    if not board.is_capture(move):
                        # Update Killer Moves
                        if move not in self.killer_moves[depth]:
                            self.killer_moves[depth] = [move, self.killer_moves[depth][0]]
                        # Update History Heuristic
                        piece = board.piece_at(move.from_square)
                        if piece:
                            self.history_scores[piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)][move.to_square] += depth * depth
                    break

        # Store in Transposition Table
        flag = EXACT_SCORE if alpha < beta else (LOWER_BOUND if alpha >= beta else UPPER_BOUND)
        self.transposition_table[board_hash] = {'score': alpha, 'depth': depth, 'flag': flag, 'move': best_move}
        
        return alpha, best_move

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        time_limit_obj = args[0] if (args and isinstance(args[0], Limit)) else None
        self.time_limit = self.get_time_budget(board, time_limit_obj) if time_limit_obj else 5.0
        self.start_time = time.time()
        
        self.nodes_searched = 0
        self.transposition_table.clear()
        self.killer_moves = [[None, None] for _ in range(self.MAX_DEPTH)]
        self.history_scores = [[0] * 64 for _ in range(12)]
        self.pv_table.clear()
        
        best_move_found = None
        perspective = 1 if board.turn == chess.WHITE else -1

        for depth in range(1, self.MAX_DEPTH):
            try:
                score, move = self.pvs_search(board, depth, -float('inf'), float('inf'), True)
                if move:
                    best_move_found = move
                    elapsed = time.time() - self.start_time
                    # UCI-like output for debugging/info
                    print(
                        f"info depth {depth} score cp {score * perspective} "
                        f"nodes {self.nodes_searched} time {int(elapsed * 1000)}ms "
                        f"pv {move.uci()}"
                    )

            except TimeoutError:
                break

        if best_move_found is None and list(board.legal_moves):
            best_move_found = random.choice(list(board.legal_moves))

        return PlayResult(best_move_found, None)