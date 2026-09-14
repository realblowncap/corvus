use chess_engine::board::Board;
use chess_engine::search::{SearchLimits, Searcher};

fn main() {
    // Simple CLI: think at a given depth on the starting position
    // Usage: cargo run --bin deep_think [depth]
    let args: Vec<String> = std::env::args().collect();
    let depth: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let mut s = Searcher::new(Board::startpos());
    let r = s.think(SearchLimits { depth, ..SearchLimits::default() });
    println!("think depth {}: best {:?} score {} nodes {}", depth, r.best.map(|m| m.uci()).unwrap_or("(none)".to_string()), r.score, r.nodes);
    let pv = s.pv_line(depth as usize);
    let pv_str: Vec<String> = pv.iter().map(|m| m.uci()).collect();
    println!("pv: {}", pv_str.join(" "));
}
