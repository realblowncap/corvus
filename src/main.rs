// Corvus Chess Engine
// Copyright (C) 2026 Avi
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

use chess_engine::analyze;
use chess_engine::board::Board;
use chess_engine::perft;
use chess_engine::uci;

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 {
        match args[1].as_str() {
            "perft" => {
                let depth: u32 = args
                    .get(2)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5);
                let mut b = Board::startpos();
                let n = perft::perft(&mut b, depth);
                println!("perft depth {}: {}", depth, n);
                return;
            }
            "divide" => {
                let depth: u32 = args
                    .get(2)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2);
                let mut b = Board::startpos();
                perft::divide(&mut b, depth);
                return;
            }
            "perftfen" => {
                let depth: u32 = args.pop().and_then(|s| s.parse().ok()).unwrap_or(4);
                let fen = args[2..].join(" ");
                let mut b = match Board::from_fen(&fen) {
                    Ok(b) => b,
                    Err(e) => {
                        println!("bad fen: {}", e);
                        return;
                    }
                };
                let n = perft::perft(&mut b, depth);
                println!("perft depth {}: {}", depth, n);
                return;
            }
            "analyze" => {
                let path = args.get(2).cloned().unwrap_or_else(|| {
                    println!("usage: corvus analyze <file.pgn> [depth]");
                    std::process::exit(1);
                });
                let depth: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
                match analyze::analyze(&path, depth) {
                    Ok(()) => {}
                    Err(e) => {
                        println!("error: {}", e);
                        std::process::exit(1);
                    }
                }
                return;
            }
            _ => {}
        }
    }
    // Lightweight builtin bench/utility: allow running a depth-limited think from CLI
    // Example: cargo run -- bench 20
    // If invoked as a standalone binary (no args), fall back to UCI mode (uci::run).
    uci::run();
}