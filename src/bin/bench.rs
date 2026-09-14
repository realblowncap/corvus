use std::time::{Instant};

use chess_engine::board::Board;
use chess_engine::search::{SearchLimits, Searcher};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Usage: bench [depth] [movetime_ms]
    // Default: depth=20, movetime=0 (no movetime limit)
    let depth: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let movetime: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    let board = Board::startpos();
    let mut s = Searcher::new(board);
    let limits = SearchLimits {
        depth,
        movetime,
        nodes: 0,
        wtime: 0,
        btime: 0,
        winc: 0,
        binc: 0,
        infinite: false,
    };

    let t0 = Instant::now();
    let res = s.think(limits);
    let elapsed = t0.elapsed().as_millis();

    println!("bench depth={} movetime={} best={:?} score={} nodes={} time_ms={}", depth, movetime, res.best.map(|m| m.uci()).unwrap_or("(none)".to_string()), res.score, res.nodes, elapsed);
    let pv = s.pv_line(depth as usize);
    let pv_str: Vec<String> = pv.iter().map(|m| m.uci()).collect();
    println!("pv: {}", pv_str.join(" "));
}
