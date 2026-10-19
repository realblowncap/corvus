#!/usr/bin/env python3
"""Mistake-rate meter for Corvus.

Plays the engine against itself from `tools/openings.epd` at a fixed depth
(the way the GUI plays it) and judges every single move with Stockfish:
for each position, the loss is

    score(best line, mover's POV) - score(after played move, mover's POV)

averaged over Stockfish's own search at a fixed judge depth.  Moves with
loss >= MISTAKE cp count as mistakes, >= BLUNDER cp as blunders.

Usage:
    python3 tools/mistakes.py <engine> [--depth 10] [--judge-depth 14]
                              [--openings 25] [--plies 16] [--threads 0]
                              [--label name]

`--threads 0` means "don't setoption Threads" (engine default = all cores).
Set a label so runs are distinguishable in the summary table.
"""
import argparse
import subprocess
import sys

import chess
import chess.engine

MISTAKE = 100   # cp loss that counts as a "mistake"
BLUNDER = 300   # cp loss that counts as a "blunder"
MATE_SCORE = 100_000


def load_positions(path, n):
    pos = []
    with open(path) as f:
        for line in f:
            fields = line.split(";")[0].split()
            if len(fields) < 4:
                continue
            pos.append(" ".join(fields[:6]))
            if len(pos) >= n:
                break
    return pos


def cp(score):
    return score.score(mate_score=MATE_SCORE)


class Engine:
    def __init__(self, cmd, threads, persist_tt=False):
        self.persist_tt = persist_tt
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci")
        self._read_until("uciok")
        if threads:
            self._send(f"setoption name Threads value {threads}")
        self._send("setoption name Hash value 256")

    def _send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, prefix):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            line = line.strip()
            if line.startswith(prefix):
                return line

    def bestmove(self, board, depth, time_limit):
        # Default: ucinewgame before every search so builds with a persistent
        # TT behave like builds that rebuild it per `go` — the A/B comparison
        # then measures search/eval changes only, not TT persistence.  With
        # --persist-tt the clear happens once per game (sent by the caller),
        # matching real GUI play.
        if not self.persist_tt:
            self._send("ucinewgame")
        self._send("position fen " + board.fen())
        self._send(f"go depth {depth}")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.startswith("bestmove"):
                return line.split()[1]

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def judge_move(sf, board, move, judge_depth):
    """Return (loss_cp, best_cp, played_cp) from the mover's perspective."""
    mover = board.turn
    best_info = sf.analyse(board, chess.engine.Limit(depth=judge_depth))
    best = cp(best_info["score"].pov(mover))
    board.push(move)
    played_info = sf.analyse(board, chess.engine.Limit(depth=judge_depth))
    played = cp(played_info["score"].pov(mover))
    board.pop()
    return best - played, best, played


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--judge-depth", type=int, default=14)
    ap.add_argument("--openings", type=int, default=25)
    ap.add_argument("--plies", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--persist-tt", action="store_true",
                    help="share the TT across moves within a game (real-play semantics)")
    ap.add_argument("--label", default="")
    ap.add_argument("--stockfish", default="stockfish")
    args = ap.parse_args()

    positions = load_positions("tools/openings.epd", args.openings)
    eng = Engine(args.engine, args.threads, args.persist_tt)
    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    sf.configure({"Hash": 256})

    losses, mistakes, blunders, judged = [], 0, 0, 0
    games = 0
    try:
        for fen in positions:
            board = chess.Board(fen)
            eng._send("ucinewgame")
            for _ in range(args.plies):
                if board.is_game_over() or board.can_claim_draw():
                    break
                mv_uci = eng.bestmove(board, args.depth, 30)
                try:
                    move = chess.Move.from_uci(mv_uci)
                except Exception:
                    break
                if move not in board.legal_moves:
                    print(f"  ILLEGAL {mv_uci} in {board.fen()}", file=sys.stderr)
                    break
                loss, best, played = judge_move(sf, board, move, args.judge_depth)
                judged += 1
                losses.append(loss)
                if loss >= MISTAKE:
                    mistakes += 1
                if loss >= BLUNDER:
                    blunders += 1
                if loss >= BLUNDER:
                    print(f"  blunder {board.fen()} move {move} loss {loss} "
                          f"(best {best} -> {played})")
                board.push(move)
            games += 1
    finally:
        eng.close()
        sf.quit()

    n = len(losses) or 1
    avg = sum(losses) / n
    worst = sorted(zip(losses, positions), reverse=True)[:3]
    print(f"\n=== {args.label or args.engine} | threads={args.threads or 'default'} "
          f"| depth {args.depth} | judge d{args.judge_depth} ===")
    print(f"games={games} decisions={judged} avg_loss={avg:.1f}cp "
          f"mistakes(>={MISTAKE})={mistakes} ({100*mistakes/n:.1f}%) "
          f"blunders(>={BLUNDER})={blunders} ({100*blunders/n:.1f}%)")


if __name__ == "__main__":
    main()
