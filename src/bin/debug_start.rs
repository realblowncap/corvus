use chess_engine::board::Board;
use chess_engine::movegen::{generate_legal, MoveList};
use chess_engine::search::{SearchLimits, Searcher};

fn main() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let b = Board::from_fen(fen).unwrap();
    let mut s = Searcher::new(b.clone());
    s.quiet = false; // Enable debug output
    
    println!("=== Starting Position Investigation ===");
    println!("Board: {}", b.to_fen());
    println!("In check: {}", b.in_check(b.side));
    println!("Side to move: {}", b.side);
    
    // Test root moves generation
    let mut root_moves = MoveList::new();
    generate_legal(&mut s.board, &mut root_moves);
    println!("Root moves generated: {}", root_moves.len);
    for (i, m) in root_moves.iter().take(5).enumerate() {
        println!("  {}: {}", i + 1, m.uci());
    }
    
    // Test search_root directly
    println!("\n=== Testing search_root directly ===");
    let (mv, sc) = s.search_root(1, -10000, 10000);
    println!("search_root result: move={} score={}", mv.map(|m| m.uci()).unwrap_or_default(), sc);
    println!("Nodes searched: {}", s.nodes);
    println!("Stopped: {}", s.stopped);
    
    // Test think with very small limits
    println!("\n=== Testing think with small limits ===");
    s = Searcher::new(b.clone());
    let r = s.think(SearchLimits { depth: 20, nodes: 10000, ..SearchLimits::default() });
    println!("think result: depth={} score={} best={} nodes={}", r.depth, r.score, r.best.map(|m| m.uci()).unwrap_or_default(), r.nodes);
}