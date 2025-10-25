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
from typing import Optional, Dict
from collections import namedtuple
import time
import random


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


# --- Constants for Search and Evaluation ---
INF = float('inf')
MATE_SCORE = 1_000_000
MAX_DEPTH = 64  # A practical limit for iterative deepening and killer moves table

# Transposition Table entry flags
TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2
TTEntry = namedtuple('TTEntry', ['depth', 'score', 'flag', 'best_move'])


class MyBot(ExampleEngine):
    """A vastly improved chess engine with modern search and evaluation techniques.

    This bot addresses the key limitations of the original template by implementing:
    - Iterative Deepening with smart Time Management.
    - Alpha-Beta Pruning for efficient search.
    - A Transposition Table to cache results and avoid re-computation.
    - Quiescence Search to stabilize evaluation at the search horizon.
    - Advanced Move Ordering (MVV-LVA) to maximize pruning effectiveness.
    - A richer Evaluation Function using Piece-Square Tables for positional awareness.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transposition_table: Dict[int, TTEntry] = {}
        self.killer_moves = [[None, None] for _ in range(MAX_DEPTH)]
        self.nodes_searched = 0
        self.start_time = 0.0
        self.stop_time = 0.0
        self._init_evaluation_tables()

    def search(self,
             board: chess.Board,
             time_limit: Limit,
             ponder: bool,
             draw_offered: bool,
             root_moves: MOVE) -> PlayResult:
        """
        Entry point for the bot's search. Implements an iterative deepening framework
        with time management.
        """
        self.nodes_searched = 0
        self.transposition_table.clear() # Clear cache for new search

        # --- Smart Time Management ---
        self._calculate_time_budget(board, time_limit)

        best_move = None
        
        try:
            for depth in range(1, MAX_DEPTH):
                # Start search for the current depth
                score = self._alphabeta(board, depth, -INF, INF, 0)
                
                # After a full search, the TT entry for the root contains the best move
                tt_entry = self.transposition_table.get(chess.polyglot.zobrist_hash(board))
                
                # Always store the best move from the last fully completed depth
                if tt_entry and tt_entry.best_move:
                    best_move = tt_entry.best_move

                # Optional: log engine info (similar to UCI)
                elapsed = time.monotonic() - self.start_time
                logger.debug(f"info depth {depth} score cp {int(score)} nodes {self.nodes_searched} time {int(elapsed*1000)} pv {best_move}")

                # If we've found a mate, no need to search deeper
                if abs(score) >= MATE_SCORE - MAX_DEPTH:
                    break

        except TimeoutError:
            # Time is up, exit the loop and use the last completed search's result
            logger.debug("Timeout reached, using best move from last completed depth.")
            pass

        # Fallback if no move is found (should be very rare)
        if best_move is None:
            possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)
            best_move = random.choice(possible_moves)

        return PlayResult(best_move, None)

    ## ------------------ ##
    ## CORE SEARCH LOGIC  ##
    ## ------------------ ##

    def _alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int) -> float:
        """The core alpha-beta search algorithm with integrated transposition table."""
        self.nodes_searched += 1

        # Check for timeout every 2048 nodes
        if self.nodes_searched & 2047 == 0 and time.monotonic() > self.stop_time:
            raise TimeoutError

        # Check for terminal nodes first
        if board.is_game_over():
            return self._evaluate_terminal(board)

        # At depth 0, switch to quiescence search to stabilize the evaluation
        if depth <= 0:
            return self._quiescence(board, alpha, beta)

        # --- Transposition Table Lookup ---
        zobrist_key = chess.polyglot.zobrist_hash(board) # <-- CORRECTED
        tt_entry = self.transposition_table.get(zobrist_key)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_EXACT:
                return tt_entry.score
            elif tt_entry.flag == TT_LOWERBOUND:
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TT_UPPERBOUND:
                beta = min(beta, tt_entry.score)
            if alpha >= beta:
                return tt_entry.score

        best_score = -INF
        best_move = None
        tt_flag = TT_UPPERBOUND

        # --- Move Generation and Ordering ---
        moves = self._order_moves(board, tt_entry.best_move if tt_entry else None, ply)

        for move in moves:
            board.push(move)
            score = -self._alphabeta(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            
            if best_score > alpha:
                alpha = best_score
                tt_flag = TT_EXACT # We have found a new best move
            
            # --- Alpha-Beta Pruning ---
            if alpha >= beta:
                if not board.is_capture(move):
                    self._store_killer_move(move, ply)
                self.transposition_table[zobrist_key] = TTEntry(depth, beta, TT_LOWERBOUND, best_move)
                return beta # Fail-high

        # --- Transposition Table Store ---
        self.transposition_table[zobrist_key] = TTEntry(depth, best_score, tt_flag, best_move)
        return best_score

    def _quiescence(self, board: chess.Board, alpha: float, beta: float) -> float:
        """A specialized search that only considers captures to ensure stability."""
        self.nodes_searched += 1
        stand_pat_score = self._evaluate(board)

        if stand_pat_score >= beta:
            return beta
        if alpha < stand_pat_score:
            alpha = stand_pat_score

        captures = [m for m in board.legal_moves if board.is_capture(m)]
        ordered_captures = sorted(captures, key=lambda m: self._mvv_lva_score(board, m), reverse=True)
        
        for move in ordered_captures:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    ## ------------------------ ##
    ## EVALUATION & HEURISTICS  ##
    ## ------------------------ ##

    def _evaluate(self, board: chess.Board) -> int:
        """Evaluates the board ALWAYS from White's perspective."""
        if board.is_game_over():
            return self._evaluate_terminal(board)

        score = 0
        for square, piece in board.piece_map().items():
            piece_val = self.PIECE_VALUES[piece.piece_type]
            pst_val = self.PIECE_SQUARE_TABLES[piece.piece_type][square if piece.color == chess.WHITE else 63 - square]
            score += (piece_val + pst_val) * (1 if piece.color == chess.WHITE else -1)

        # Return score relative to the current player to move
        return score if board.turn == chess.WHITE else -score

    def _evaluate_terminal(self, board: chess.Board) -> int:
        """Returns a large score for checkmates or 0 for draws."""
        outcome = board.outcome()
        if outcome is None or outcome.winner is None: return 0
        
        # Mate score is relative to whose turn it is
        score = MATE_SCORE - board.fullmove_number
        return score if outcome.winner == board.turn else -score

    def _order_moves(self, board: chess.Board, tt_best_move: Optional[chess.Move], ply: int) -> list:
        """Orders legal moves to improve alpha-beta pruning efficiency."""
        move_scores = {}
        for move in board.legal_moves:
            score = 0
            if move == tt_best_move:
                score = 10000
            elif board.is_capture(move):
                score = self._mvv_lva_score(board, move) + 1000
            elif move in self.killer_moves[ply]:
                score = 500
            move_scores[move] = score
        return sorted(move_scores, key=move_scores.get, reverse=True)

    def _mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        """Scores captures using Most Valuable Victim - Least Valuable Aggressor."""
        victim_type = board.piece_type_at(move.to_square) or chess.PAWN # En passant
        attacker_type = board.piece_type_at(move.from_square)
        return self.PIECE_VALUES[victim_type] - self.PIECE_VALUES[attacker_type]

    def _store_killer_move(self, move: chess.Move, ply: int):
        if move != self.killer_moves[ply][0]:
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = move

    ## ----------------- ##
    ## UTILITY FUNCTIONS ##
    ## ----------------- ##

    def _calculate_time_budget(self, board: chess.Board, time_limit: Limit):
        """Calculates and sets the stop time for the current search."""
        if isinstance(time_limit.time, (int, float)):
            my_time = time_limit.time
            my_inc = 0
        elif board.turn == chess.WHITE:
            my_time = time_limit.white_clock
            my_inc = time_limit.white_inc
        else:
            my_time = time_limit.black_clock
            my_inc = time_limit.black_inc
        
        my_time = float(my_time or 5) # Default to 5 seconds if time is None
        my_inc = float(my_inc or 0)
        
        # A common strategy: use 1/40th of remaining time + 80% of increment
        time_budget = my_time / 40 + my_inc * 0.8
        time_budget = min(time_budget, my_time * 0.5) # Don't use more than half our time

        self.start_time = time.monotonic()
        self.stop_time = self.start_time + max(0.1, time_budget)

    def _init_evaluation_tables(self):
        """Initializes material values and piece-square tables."""
        self.PIECE_VALUES = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 950, chess.KING: 20000
        }
        # Piece-Square Tables (from White's perspective)
        pawn_pst=[0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0]
        knight_pst=[-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50]
        bishop_pst=[-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20]
        rook_pst=[0,0,0,0,0,0,0,0,5,10,10,10,10,10,10,5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0]
        queen_pst=[-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20]
        king_pst=[-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,-10,-20,-20,-20,-20,-20,-20,-10,20,20,0,0,0,0,20,20,20,30,10,0,0,10,30,20]
        self.PIECE_SQUARE_TABLES={chess.PAWN:pawn_pst,chess.KNIGHT:knight_pst,chess.BISHOP:bishop_pst,chess.ROOK:rook_pst,chess.QUEEN:queen_pst,chess.KING:king_pst}