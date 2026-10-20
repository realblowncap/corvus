// Bit-exact forward-pass test: evaluates 10 positions with the SFNNv16 network
// and compares against Stockfish 19 (sf_19 tag, built ARCH=x86-64 scalar, which
// compiles the exact scalar NNUE reference paths this port reproduces).
//
// Reference values are Stockfish's `eval` command output:
//   "NNUE evaluation N (side to move, internal units)"
// which is exactly `psqt/16 + positional/16` from `Network::evaluate`.
mod common;

use chess_engine::board::Board;
use chess_engine::nnue_forward;

#[test]
fn forward_pass_matches_stockfish_19() {
    let Some(net) = common::load_net() else {
        return;
    };

    // (FEN, Stockfish 19 internal-units eval, side to move)
    let cases: &[(&str, i32)] = &[
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", -1),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 4", 71),
        ("rnbqkb1r/pppppppp/5n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3", 199),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 157),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", -560),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1", -80),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 549),
        ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 56),
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", 197),
        ("8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1", 558),
    ];

    let mut ok = true;
    for (fen, expected) in cases {
        let board = Board::from_fen(fen).expect("valid fen");
        let got = nnue_forward::evaluate(&net.parsed, &board);
        let status = if got == *expected { "OK " } else { "MISMATCH" };
        if got != *expected {
            ok = false;
        }
        println!("{status}  expected {expected:>5}  got {got:>5}  |  {fen}");
    }

    assert!(ok, "forward pass must match Stockfish 19 bit-exactly on all 10 positions");
}
