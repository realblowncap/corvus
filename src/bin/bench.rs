use std::time::{Instant};

use chess_engine::board::Board;
use chess_engine::search::{SearchLimits, Searcher};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Usage: bench [depth] [movetime_ms] [threads]
    // Default: depth=20, movetime=0 (no movetime limit), threads=1 (serial)
    let depth: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let movetime: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let board = Board::startpos();
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
    let res = if threads > 1 {
        Searcher::think_parallel(
            board.clone(),
            limits,
            threads,
            true,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            0,
            false,
            std::sync::Arc::new(chess_engine::search::TT::new(24)),
        )
    } else {
        let mut s = Searcher::new(board.clone());
        s.think(limits)
    };
    let elapsed = t0.elapsed().as_millis();

    println!("bench depth={} movetime={} threads={} best={:?} score={} nodes={} time_ms={}", depth, movetime, threads, res.best.map(|m| m.uci()).unwrap_or("(none)".to_string()), res.score, res.nodes, elapsed);
    println!("done_depth={}", res.depth);
    let mut s = Searcher::new(board);
    let pv = s.pv_line(depth as usize);
    let pv_str: Vec<String> = pv.iter().map(|m| m.uci()).collect();
    println!("pv: {}", pv_str.join(" "));
}
