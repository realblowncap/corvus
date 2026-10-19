#!/usr/bin/env python3
"""Hunt for "queen takes a cheaper piece and loses material" blunders.

Plays the engine against itself from `tools/openings.epd` (the way the GUI
does), judges every move with Stockfish, and prints every move that loses
>= BLUNDER cp *and* is a queen capture of a lower-valued piece.  That is the
exact shape of the reported "traded my queen for a bishop" mistake.

Usage:
    python3 tools/qblunder.py <engine> [--depth 10] [--judge-depth 16]
                               [--openings 25] [--plies 20]
"""
import argparse
import os
import subprocess
import sys

import chess
import chess.engine

BLUNDER = 300
MATE_SCORE = 100_000
HERE = os.path.dirname(os.path.abspath(__file__))


def load_positions(path, n):
    pos = []
    with open(path) as f:
        for line in f:
            fields = line.split(";")[0].split()
            if len(fields) >= 4:
                pos.append(" ".join(fields[:6]))
            if len(pos) >= n:
                break
    return pos


class Engine:
    def __init__(self, cmd, depth):
        self.depth = depth
        self.proc = subprocess.Popen([cmd], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci")
        self._read_until("uciok")
        self._send("setoption name Hash value 256")

    def _send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, prefix):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.strip().startswith(prefix):
                return line.strip()

    def bestmove(self, board):
        self._send("ucinewgame")
        self._send("position fen " + board.fen())
        self._send(f"go depth {self.depth}")
        return self._read_until("bestmove").split()[1]

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--judge-depth", type=int, default=16)
    ap.add_argument("--openings", type=int, default=25)
    ap.add_argument("--plies", type=int, default=20)
    ap.add_argument("--stockfish", default="stockfish")
    args = ap.parse_args()

    positions = load_positions(os.path.join(HERE, "openings.epd"), args.openings)
    eng = Engine(args.engine, args.depth)
    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    sf.configure({"Hash": 256})

    losses = []
    qhits = 0
    try:
        for fen in positions:
            board = chess.Board(fen)
            for _ in range(args.plies):
                if board.is_game_over() or board.can_claim_draw():
                    break
                best = sf.analyse(board, chess.engine.Limit(depth=args.judge_depth))
                best_sc = best["score"].pov(board.turn).score(mate_score=MATE_SCORE)
                best_mv = best["pv"][0] if best["pv"] else None
                mv = eng.bestmove(board)
                try:
                    move = chess.Move.from_uci(mv)
                except Exception:
                    break
                if move not in board.legal_moves:
                    print(f"ILLEGAL {mv} in {board.fen()}")
                    break
                mover = board.turn
                cap = board.piece_at(move.to_square)
                piece = board.piece_at(move.from_square)
                board.push(move)
                aft = sf.analyse(board, chess.engine.Limit(depth=args.judge_depth))
                played_sc = aft["score"].pov(mover).score(mate_score=MATE_SCORE)
                board.pop()
                loss = best_sc - played_sc
                losses.append(loss)
                if loss >= BLUNDER:
                    tag = ""
                    is_qcap = (piece is not None and piece.piece_type == chess.QUEEN
                               and cap is not None and cap.piece_type != chess.QUEEN)
                    if is_qcap:
                        qhits += 1
                        tag = "  <== QUEEN-FOR-PIECE"
                    best_s = best_mv.uci() if best_mv else "?"
                    print(f"blunder loss {loss:5d} (best {best_sc:5d}->{played_sc:5d}) "
                          f"played {mv} (best {best_s}) | {board.fen()}{tag}")
                board.push(move)
    finally:
        eng.close()
        sf.quit()

    n = len(losses) or 1
    blunders = sum(1 for x in losses if x >= BLUNDER)
    print(f"\n=== {args.engine} depth {args.depth} judge d{args.judge_depth} "
          f"n={len(losses)} ===")
    print(f"avg_loss={sum(losses)/n:.1f}cp blunders={blunders} "
          f"({100*blunders/n:.1f}%) queen-for-piece blunders={qhits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
