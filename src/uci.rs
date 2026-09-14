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

use crate::board::*;
use crate::perft;
use crate::search::{SearchLimits, SearchResult, Searcher};
use std::io::BufRead;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

pub const ENGINE_NAME: &str = "Corvus 0.1.0";
pub const ENGINE_AUTHOR: &str = "Avi";

static OUT: OnceLock<Mutex<std::io::Stdout>> = OnceLock::new();

pub fn outln(line: String) {
    let out = OUT.get_or_init(|| Mutex::new(std::io::stdout()));
    if let Ok(mut o) = out.lock() {
        let _ = o.write_all(line.as_bytes());
        let _ = o.write_all(b"\n");
        let _ = o.flush();
    }
}

struct CommandState {
    board: Board,
    abort: Option<Arc<AtomicBool>>,
    receiver: Option<mpsc::Receiver<SearchResult>>,
}

pub fn run() {
    let mut state = CommandState {
        board: Board::startpos(),
        abort: None,
        receiver: None,
    };

    let stdin = std::io::stdin();
    let _ = eprintln!("Corvus chess engine starting");

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        drain_result(&mut state);

        match tokens[0] {
            "uci" => {
                outln(format!("id name {}", ENGINE_NAME));
                outln(format!("id author {}", ENGINE_AUTHOR));
                outln("uciok".to_string());
            }
            "isready" => {
                outln("readyok".to_string());
            }
            "ucinewgame" => {
                if let Some(a) = &state.abort {
                    a.store(true, Ordering::Relaxed);
                }
                drain_result(&mut state);
                state.board = Board::startpos();
                state.abort = None;
                state.receiver = None;
            }
            "position" => {
                if let Some(a) = &state.abort {
                    a.store(true, Ordering::Relaxed);
                }
                drain_result(&mut state);
                state.abort = None;
                state.receiver = None;
                parse_position(&mut state.board, &tokens[1..]);
            }
            "go" => { /* no-op for bench harness */
                if let Some(a) = &state.abort {
                    a.store(true, Ordering::Relaxed);
                }
                drain_result(&mut state);
                let abort = Arc::new(AtomicBool::new(false));
                let limits = parse_go(&tokens[1..]);
                let mut searcher = Searcher::new(state.board.clone());
                searcher.abort = abort.clone();
                let (tx, rx) = mpsc::channel();
                state.abort = Some(abort);
                state.receiver = Some(rx);
                thread::spawn(move || {
                    let result = searcher.think(limits);
                    let _ = tx.send(result);
                    if !searcher.abort.load(Ordering::Relaxed) {
                        let mv = result.best.map(|m| m.uci()).unwrap_or_else(|| "(none)".to_string());
                        outln(format!("bestmove {}", mv));
                    }
                });
            }
            "stop" => {
                if let Some(a) = &state.abort {
                    a.store(true, Ordering::Relaxed);
                }
                drain_result(&mut state);
            }
            "d" => {
                outln(state.board.display());
                outln(format!("FEN: {}", state.board.to_fen()));
            }
            "eval" => {
                outln(format!("eval (side to move): {}", crate::eval::evaluate(&state.board)));
            }
            "perft" => {
                if let Some(d) = tokens.get(1).and_then(|s| s.parse::<u32>().ok()) {
                    let b = &mut state.board;
                    perft::divide(b, d);
                }
            }
            "quit" => {
                if let Some(a) = &state.abort {
                    a.store(true, Ordering::Relaxed);
                }
                break;
            }
            _ => {}
        }
    }
}

fn drain_result(state: &mut CommandState) {
    if let Some(rx) = &state.receiver {
        match rx.try_recv() {
            Ok(_) => {
                state.abort = None;
                state.receiver = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                state.abort = None;
                state.receiver = None;
            }
        }
    }
}

fn parse_position(board: &mut Board, tokens: &[&str]) {
    let mut i = 0;
    let mut fen = String::new();
    if i < tokens.len() && tokens[i] == "startpos" {
        *board = Board::startpos();
        i += 1;
    } else if i + 1 < tokens.len() && tokens[i] == "fen" {
        i += 1;
        let mut fen_parts: Vec<&str> = Vec::new();
        while i < tokens.len() && tokens[i] != "moves" {
            fen_parts.push(tokens[i]);
            i += 1;
        }
        fen = fen_parts.join(" ");
    } else {
        *board = Board::startpos();
    }

    if !fen.is_empty() {
        match Board::from_fen(&fen) {
            Ok(b) => *board = b,
            Err(e) => outln(format!("info string invalid fen: {}", e)),
        }
    }

    if i < tokens.len() && tokens[i] == "moves" {
        i += 1;
        while i < tokens.len() {
            match board.uci_move(tokens[i]) {
                Some(m) => {
                    let _ = board.make(m);
                }
                None => outln(format!("info string illegal or unknown move '{}'", tokens[i])),
            }
            i += 1;
        }
    }
}

fn parse_go(tokens: &[&str]) -> SearchLimits {
    let mut l = SearchLimits::default();
    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "depth" => {
                i += 1;
                if let Some(d) = tokens.get(i).and_then(|s| s.parse::<u32>().ok()) {
                    l.depth = d;
                }
            }
            "nodes" => {
                i += 1;
                if let Some(n) = tokens.get(i).and_then(|s| s.parse::<u64>().ok()) {
                    l.nodes = n;
                }
            }
            "movetime" => {
                i += 1;
                if let Some(t) = tokens.get(i).and_then(|s| s.parse::<u64>().ok()) {
                    l.movetime = t;
                }
            }
            "wtime" => {
                i += 1;
                if let Some(t) = tokens.get(i).and_then(|s| s.parse::<u64>().ok()) {
                    l.wtime = t;
                }
            }
            "btime" => {
                i += 1;
                if let Some(t) = tokens.get(i).and_then(|s| s.parse::<u64>().ok()) {
                    l.btime = t;
                }
            }
            "winc" => {
                i += 1;
                if let Some(t) = tokens.get(i).and_then(|s| s.parse::<u64>().ok()) {
                    l.winc = t;
                }
            }
            "binc" => {
                i += 1;
                if let Some(t) = tokens.get(i).and_then(|s| s.parse::<u64>().ok()) {
                    l.binc = t;
                }
            }
            "infinite" => l.infinite = true,
            _ => {}
        }
        i += 1;
    }

    if tokens.is_empty() {
        l.infinite = true;
    }
    if l.infinite {
        l.depth = 0;
        l.nodes = 0;
        l.movetime = 0;
        l.wtime = 0;
        l.btime = 0;
    }
    l
}

pub fn best_move_sync(board: &Board, depth: u32) -> Option<Move> {
    let mut s = Searcher::new(board.clone());
    let limits = SearchLimits {
        depth,
        ..SearchLimits::default()
    };
    let result = s.think(limits);
    result.best
}