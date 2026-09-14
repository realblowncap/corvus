use chess_engine::board::Board;
use chess_engine::search::{SearchLimits, Searcher};

fn main() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    
    // Test with different depth limits
    for depth in [1, 2, 3] {
        let b = Board::from_fen(fen).unwrap();
        let mut s = Searcher::new(b);
        s.quiet = true;
        let r = s.think(SearchLimits { depth: depth, nodes: 100000, ..SearchLimits::default() });
        println!("depth={} search: depth={} score={} best={} nodes={}", depth, r.depth, r.score, r.best.map(|m| m.uci()).unwrap_or_default(), r.nodes);
    }
}