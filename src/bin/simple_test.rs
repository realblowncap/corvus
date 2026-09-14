use chess_engine::board::Board;
use chess_engine::search::{SearchLimits, Searcher};

fn main() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let b = Board::from_fen(fen).unwrap();
    println!("Board: {}", b.to_fen());
    println!("In check: {}", b.in_check(b.side));
    println!("Side to move: {}", b.side);
    
    let mut s = Searcher::new(b);
    s.quiet = true;
    
    // Test with minimal depth
    let r = s.think(SearchLimits { depth: 1, nodes: 1000, ..SearchLimits::default() });
    println!("Result: depth={} score={} best={} nodes={}", r.depth, r.score, r.best.map(|m| m.uci()).unwrap_or_default(), r.nodes);
}