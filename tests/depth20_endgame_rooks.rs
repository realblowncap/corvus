use chess_engine::board::Board;
use chess_engine::search::{SearchLimits, Searcher};

#[test]
fn depth20_endgame_rook_only() {
    // Simple rook endgame: White rook and king vs Black king
    let fen = "7k/8/8/8/8/8/8/K6R w - 0 1";
    let b = Board::from_fen(fen).unwrap();
    let mut s = Searcher::new(b);
    let r = s.think(SearchLimits { depth: 20, movetime: 1500, ..SearchLimits::default() });
    assert!(r.nodes > 0, "depth-20 rook endgame should generate nodes");
    assert!(r.best.is_some(), "depth-20 rook endgame should produce a best move");
}

#[test]
fn depth20_endgame_rooks_with_pawn() {
    // Rook endgame with an extra pawn on the board
    let fen = "7k/8/8/8/8/8/7P/K6R w - 0 1";
    let b = Board::from_fen(fen).unwrap();
    let mut s = Searcher::new(b);
    let r = s.think(SearchLimits { depth: 20, movetime: 1500, ..SearchLimits::default() });
    assert!(r.nodes > 0, "depth-20 rook+pawn endgame should generate nodes");
    assert!(r.best.is_some(), "depth-20 rook+pawn endgame should produce a best move");
}
