#!/usr/bin/env python3
"""Paired position benchmark for Corvus.

`mistakes.py` plays self-play games, so two different engines quickly reach
*different* positions and their mistake counts are independent samples: with
only ~400 decisions the binomial noise (sigma ~ 4-5 mistakes) hides any real
1% improvement.

This tool instead evaluates every engine on the *same* fixed list of
positions:

    gen   : generate the position list once (self-play of a reference build,
            plus optionally positions harvested from existing logs) and cache
            Stockfish's best score for each FEN.
    eval  : search each FEN with the engine at a fixed depth, judge both the
            best (cached) and the played move with Stockfish, and report
            mistakes/blunders/average loss.  With --ref it also prints a
            *paired* comparison against a previously saved loss vector
            (mean delta, paired t, and decisive flips).

Because every engine sees the same FENs and the cached best score is shared,
position-to-position difficulty cancels out of the comparison.

Position file format (one per line):
    <fen> | <best_cp_from_mover_pov> | <origin>

Loss vector file format: one integer cp loss per line, same order as positions.
"""
import argparse
import os
import random
import statistics
import subprocess
import sys

import chess
import chess.engine

MISTAKE = 100
BLUNDER = 300
MATE_SCORE = 100_000
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SET = os.path.join(HERE, "positions_paired.txt")


def cp(score):
    return score.score(mate_score=MATE_SCORE)


def load_set(path):
    positions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            positions.append((parts[0], int(parts[1]), parts[2] if len(parts) > 2 else ""))
    return positions


def positions_from_logs(paths):
    """Harvest FENs the engine already got wrong in earlier mistake runs."""
    out = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("blunder ") or line.startswith("mistake "):
                    # "blunder <fen fields> move <uci> loss ..."
                    head = line.split(" move ")[0].split(None, 1)
                    if len(head) == 2:
                        out.add(head[1].strip())
    return sorted(out)


def best_score(sf, board, judge_depth):
    return cp(sf.analyse(board, chess.engine.Limit(depth=judge_depth))["score"].pov(board.turn))


def gen(args):
    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    sf.configure({"Hash": 256})
    fens = []

    if args.openings:
        positions = []
        with open(os.path.join(HERE, "openings.epd")) as f:
            for line in f:
                fields = line.split(";")[0].split()
                if len(fields) >= 4:
                    positions.append(" ".join(fields[:6]))
                if len(positions) >= args.openings:
                    break
        rnd = random.Random(args.seed)
        eng = subprocess.Popen([args.engine], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True, bufsize=1)

        def send(line):
            eng.stdin.write(line + "\n")
            eng.stdin.flush()

        def read_until(prefix):
            while True:
                line = eng.stdout.readline()
                if not line:
                    raise RuntimeError("engine died")
                if line.strip().startswith(prefix):
                    return line.strip()
        send("uci")
        read_until("uciok")
        send("setoption name Hash value 256")
        for fen in positions:
            board = chess.Board(fen)
            send("ucinewgame")
            for _ in range(args.plies):
                if board.is_game_over() or board.can_claim_draw():
                    break
                send("position fen " + board.fen())
                send(f"go depth {args.gen_depth}")
                mv = read_until("bestmove").split()[1]
                try:
                    move = chess.Move.from_uci(mv)
                except Exception:
                    break
                if move not in board.legal_moves:
                    break
                board.push(move)
                if board.is_game_over():
                    break
                # Prefer a spread: keep two random positions per game plus any
                # position after the first 8 plies.
                fens.append(board.fen())
            rnd.shuffle(fens[-args.plies:])
        send("quit")

    for fen in positions_from_logs(args.logs):
        fens.append(fen)

    seen, uniq = set(), []
    for fen in fens:
        key = " ".join(fen.split()[:4])
        if key not in seen:
            seen.add(key)
            uniq.append(fen)

    if args.limit:
        uniq = uniq[: args.limit]

    with open(args.out, "w") as f:
        for fen in uniq:
            board = chess.Board(fen)
            if board.is_game_over() or board.can_claim_draw():
                continue
            b = best_score(sf, board, args.judge_depth)
            f.write(f"{fen} | {b} | gen\n")
    sf.quit()
    print(f"wrote {args.out} ({len(uniq)} positions)")


class Engine:
    def __init__(self, cmd, threads):
        self.proc = subprocess.Popen([cmd], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("uci")
        self.read_until("uciok")
        if threads:
            self.send(f"setoption name Threads value {threads}")
        self.send("setoption name Hash value 256")

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefix):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.strip().startswith(prefix):
                return line.strip()

    def bestmove(self, board, depth):
        self.send("ucinewgame")
        self.send("position fen " + board.fen())
        self.send(f"go depth {depth}")
        return self.read_until("bestmove").split()[1]

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def eval_run(args):
    positions = load_set(args.set)
    eng = Engine(args.engine, args.threads)
    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    sf.configure({"Hash": 256})

    losses, names = [], []
    mistakes = blunders = 0
    try:
        for i, (fen, best, _origin) in enumerate(positions):
            board = chess.Board(fen)
            mv = eng.bestmove(board, args.depth)
            try:
                move = chess.Move.from_uci(mv)
            except Exception:
                continue
            if move not in board.legal_moves:
                print(f"  ILLEGAL {mv} in {fen}", file=sys.stderr)
                continue
            # `played` must be scored from the *mover's* POV: after pushing the
            # move it is the opponent's turn, so capture the colour first and
            # pass the score through .pov(mover).  (Getting this wrong makes the
            # loss a meaningless mix of two points of view.)
            mover = board.turn
            board.push(move)
            played = cp(sf.analyse(board, chess.engine.Limit(depth=args.judge_depth))["score"].pov(mover))
            board.pop()
            loss = best - played
            losses.append(loss)
            names.append(fen)
            if loss >= MISTAKE:
                mistakes += 1
            if loss >= BLUNDER:
                blunders += 1
                print(f"  blunder {fen} move {move} loss {loss} (best {best} -> {played})")
    finally:
        eng.close()
        sf.quit()

    n = len(losses) or 1
    avg = sum(losses) / n
    med = statistics.median(losses) if losses else 0
    print(f"\n=== {args.label or args.engine} | threads={args.threads or 'default'} "
          f"| depth {args.depth} | judge d{args.judge_depth} | n={len(losses)} ===")
    print(f"avg_loss={avg:.1f}cp median={med:.1f}cp "
          f"mistakes(>={MISTAKE})={mistakes} ({100*mistakes/n:.1f}%) "
          f"blunders(>={BLUNDER})={blunders} ({100*blunders/n:.1f}%)")

    if args.save:
        with open(args.save, "w") as f:
            for x in losses:
                f.write(f"{x}\n")
        print(f"saved loss vector -> {args.save}")

    if args.ref and os.path.exists(args.ref):
        ref = [int(x) for x in open(args.ref) if x.strip()]
        m = min(len(ref), len(losses))
        if m != len(losses):
            print(f"note: ref has {len(ref)} entries, comparing first {m}")
        diffs = [losses[i] - ref[i] for i in range(m)]
        mean_d = statistics.mean(diffs)
        sd_d = statistics.stdev(diffs) if m > 1 else 0.0
        t = mean_d / (sd_d / m ** 0.5) if sd_d > 0 else 0.0
        better = sum(1 for d in diffs if d <= -MISTAKE)
        worse = sum(1 for d in diffs if d >= MISTAKE)
        print(f"paired vs {os.path.basename(args.ref)} (n={m}): "
              f"mean_delta={mean_d:+.1f}cp sd={sd_d:.1f} t={t:+.2f} "
              f"decisive_better(<=-{MISTAKE})={better} decisive_worse(>={MISTAKE})={worse}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--engine", required=True)
    g.add_argument("--out", default=DEFAULT_SET)
    g.add_argument("--openings", type=int, default=25)
    g.add_argument("--plies", type=int, default=12)
    g.add_argument("--gen-depth", type=int, default=8)
    g.add_argument("--limit", type=int, default=400)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--judge-depth", type=int, default=14)
    g.add_argument("--stockfish", default="stockfish")
    g.add_argument("--logs", nargs="*", default=[])

    e = sub.add_parser("eval")
    e.add_argument("--engine", required=True)
    e.add_argument("--set", default=DEFAULT_SET)
    e.add_argument("--depth", type=int, default=10)
    e.add_argument("--judge-depth", type=int, default=14)
    e.add_argument("--threads", type=int, default=0)
    e.add_argument("--label", default="")
    e.add_argument("--save", default="")
    e.add_argument("--ref", default="")
    e.add_argument("--stockfish", default="stockfish")

    args = ap.parse_args()
    if args.cmd == "gen":
        gen(args)
    else:
        sys.exit(eval_run(args))


if __name__ == "__main__":
    main()
