#!/usr/bin/env python3
"""Node-budget self-play matcher for tuning Corvus.

Usage:
    python3 tools/match.py <old> <new> [rounds=40] [nodes-per-move=300000] [log=path]
"""
import sys, math, time, datetime
import chess, chess.engine

OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # startpos
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",  # italian
    "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5",  # ruy
    "r1bqkbnr/pppp1ppp/2n5/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4",  # scotch
    "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",  # french
    "rn1qkbnr/pp2pppp/2p3b1/8/3P4/6N1/PPP2PPP/R1BQKBNR w KQkq - 3 6",  # caro-kann
    "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",  # dragon
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",  # najdorf
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",  # qgd
    "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N1P3/PP3PPP/R1BQKBNR b KQkq - 0 4",  # slav
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",  # nimzo
    "rnbqkb1r/p1pp1ppp/1p2pn2/8/2PP4/5NP1/PP2PP1P/RNBQKB1R b KQkq - 0 4",  # qid
    "rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",  # gruenfeld
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ - 3 6",  # kingindian
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/6P1/PP2PPBP/RNBQK1NR b KQkq - 1 4",  # catalan
    "rnbqkb1r/3ppppp/p4n2/1PpP4/8/8/PP2PPPP/RNBQKBNR w KQkq - 0 5",  # benko
    "rnbqkb1r/pp1p1ppp/4pn2/2pP4/2P5/8/PP2PPPP/RNBQKBNR w KQkq - 0 4",  # benoni
    "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/4P3/PPP2PPP/RN1QKBNR b KQkq - 0 3",  # london
    "rnbqkb1r/pppp1ppp/5n2/4p3/2P5/2N3P1/PP1PPP1P/R1BQKBNR b KQkq - 0 3",  # english
    "rnbqkb1r/ppp1pppp/5n2/3p4/8/5NP1/PPPPPPBP/RNBQK2R b KQkq - 2 3",  # reti
    "rnbqkb1r/ppp1pp1p/3p1np1/8/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4",  # pirc
    "rnb1kbnr/ppp1pppp/8/q7/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 4",  # scandinavian
    "rnbqkb1r/pppp1ppp/4pn2/6B1/3P4/5N2/PPP1PPPP/RN1QKB1R b KQkq - 1 3",  # torre
    "rnbqkb1r/ppppp1pp/5n2/5p2/3P4/6P1/PPP1PPBP/RNBQK1NR b KQkq - 2 3",  # dutch
    "rnbqkbnr/ppp1pppp/8/3p4/5P2/5N2/PPPPP1PP/RNBQKB1R b KQkq - 1 2",  # bird
]


def elo(s):
    if 0 < s < 1:
        return 400 * math.log10(s / (1 - s))
    return float("inf") if s >= 1 else float("-inf")


def play_game(path, limit, tc, opening):
    """Return ('1-0'|'0-1'|'1/2-1/2', crasher) where crasher is 'w'/'b'/None."""
    eng = chess.engine.SimpleEngine.popen_uci(path)
    board = chess.Board(opening)
    turn = 'w'
    crasher = None
    try:
        for _ in range(600):
            if board.is_game_over():
                break
            if tc and tc > 0:
                result = eng.play(board, chess.engine.Limit(time=tc))
            else:
                result = eng.play(board, chess.engine.Limit(nodes=limit))
            if result.move is None:
                break
            board.push(result.move)
            turn = 'b' if turn == 'w' else 'w'
    except chess.engine.EngineTerminatedError:
        crasher = turn
    finally:
        eng.quit()
    if crasher is not None:
        return ("0-1" if crasher == 'w' else "1-0"), crasher
    if board.is_game_over():
        return board.result(), None
    return "1/2-1/2", None


def main():
    old, new = sys.argv[1], sys.argv[2]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    nodes = 300000
    tc = 0.0
    if len(sys.argv) > 4:
        try:
            # try int first (nodes), else float (time)
            s = sys.argv[4]
            if '.' in s:
                tc = float(s)
            else:
                # could be "5" for 5 seconds or 300000 nodes; treat small ints as seconds for bullet
                val = int(s)
                if val < 60:  # seconds
                    tc = float(val)
                else:
                    nodes = val
                    tc = 0.0
        except Exception:
            pass
    logpath = sys.argv[5] if len(sys.argv) > 5 else None
    logpath = sys.argv[5] if len(sys.argv) > 5 else None
    res = {"Old": 0, "New": 0, "Draw": 0}
    t0 = time.time()
    who_breaks = {"Old": 0, "New": 0}
    for i in range(rounds):
        op = OPENINGS[i % len(OPENINGS)]
        white, black = (old, new) if i % 2 == 0 else (new, old)
        rw, cw = play_game(white, nodes, tc, op)
        if rw == "1-0":
            res["Old" if white == old else "New"] += 1
        elif rw == "0-1":
            res["Old" if black == old else "New"] += 1
        else:
            res["Draw"] += 1
        if cw is not None:
            eng = "Old" if white == old else "New"
            who_breaks[eng] += 1
        print(f"  game {i+1}/{rounds}: {rw}", flush=True)
    o, n, d = res["Old"], res["New"], res["Draw"]
    tot = max(o + n + d, 1)
    S = (o + 0.5 * d) / tot
    z = 1.96
    h = z * math.sqrt((S * (1 - S)) / tot + (z * z) / (4 * tot * tot))
    dd = 1 + z * z / tot
    lo = (S + z * z / (2 * tot) - h) / dd
    hi = (S + z * z / (2 * tot) + h) / dd
    print(f"\n  games={tot}  Old={o} New={n} Draws={d}  ({time.time()-t0:.0f}s)  crashes O/N={who_breaks['Old']}/{who_breaks['New']}")
    print(f"  Old score = {S:.3f}  (95% CI {lo:.3f}..{hi:.3f})")
    print(f"  Elo diff (Old - New) = {elo(S):+.0f}  CI [{elo(lo):+.0f}, {elo(hi):+.0f}]")
    if logpath:
        with open(logpath, "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()},{old},{new},nodes={nodes},tc={tc},"
                    f"{rounds},{tot},{o},{n},{d},{S:.3f},{elo(S):+.0f}\n")


if __name__ == "__main__":
    main()