use chess_engine::board::Board;
use chess_engine::eval::evaluate;
use chess_engine::search::{SearchLimits, Searcher};

#[test]
fn knight_up_endgame_scores_positive() {
    let b = Board::from_fen("8/4k3/8/8/8/8/8/K6N b - - 0 1").unwrap();
    assert!(evaluate(&b) < -200, "black down a knight must be clearly negative");
}

#[test]
fn fork_is_converted() {
    // Knight d5 forks black's king (g8) and rook (c6). The winning line is
    // Ne7+ (forcing the king away) then Nxc6 with the knight escaping safely.
    let board = Board::from_fen("6k1/8/2r5/3N4/8/8/8/K7 w - - 0 1").unwrap();
    let mut s = Searcher::new(board);
    let res = s.think(SearchLimits {
        depth: 10,
        ..SearchLimits::default()
    });
    let bm = res.best.expect("position must have a legal move");
    assert_eq!(bm.uci(), "d5e7", "expected the fork Ne7+, got {}", bm.uci());
    assert!(
        res.score >= 300,
        "fork conversion must clearly win the rook, got {}",
        res.score
    );
    let pv = s.pv_line(4).iter().map(|m| m.uci()).collect::<Vec<String>>().join(" ");
    assert!(pv.starts_with("d5e7"), "PV must start with the fork: {}", pv);
}