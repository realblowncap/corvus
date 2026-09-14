use chess_engine::board::Board;
use chess_engine::search::{SearchLimits, Searcher};

fn main() {
    let test_positions = [
        ("starting_pos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("midgame_pos", "2rqr1k1/4bppp/5n2/np1B3b/2p1P3/4BN1P/1P1NQPP1/R3R1K1 w - - 0 1"),
        ("endgame_pos", "8/8/8/8/8/8/6PP/6Kk w - - 0 1"),
    ];
    
    for (name, fen) in test_positions.iter() {
        println!("=== Testing {} ===", name);
        println!("FEN: {}", fen);
        
        // Test with increasing node limits
        for nodes in [1_000_000, 4_000_000, 16_000_000] {
            let b = Board::from_fen(fen).unwrap();
            let mut s = Searcher::new(b);
            s.quiet = true;
            let t = std::time::Instant::now();
            let r = s.think(SearchLimits { depth: 30, nodes: nodes, ..SearchLimits::default() });
            let nps = (r.nodes as f64 / t.elapsed().as_secs_f64()) as u64;
            println!("  nodes={} -> depth={} ({} plies) score={} best={} nps={} time={:.2}s", 
                     nodes, r.depth, r.depth * 2, r.score, r.best.map(|m| m.uci()).unwrap_or_default(), nps, t.elapsed().as_secs_f32());
        }
        println!();
    }
}