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
use crate::search::{SearchLimits, SearchResult, Searcher, TT};
use std::io::BufRead;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

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
    threads: usize,
    /// Whether time-controlled game moves may be played from the internal book.
    own_book: bool,
    /// Persistent transposition table shared by every `go` in the session;
    /// cleared on `ucinewgame` and resized by the `Hash` option.
    tt: Arc<TT>,
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).max(1)
}

pub fn run() {
    let mut state = CommandState {
        board: Board::startpos(),
        abort: None,
        receiver: None,
        threads: default_threads(),
        own_book: true,
        tt: Arc::new(TT::new(24)),
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
                outln(format!(
                    "option name Threads type spin default {} min 1 max 512",
                    default_threads()
                ));
                outln("option name Hash type spin default 256 min 1 max 4096".to_string());
                outln("option name OwnBook type check default true".to_string());
                outln("uciok".to_string());
            }
            "setoption" => {
                if let Some(vpos) = tokens.iter().position(|&t| t == "value") {
                    let name = tokens[1..vpos].join(" ");
                    let value = tokens.get(vpos + 1).map(|s| s.to_string()).unwrap_or_default();
                    if name.eq_ignore_ascii_case("Threads") {
                        if let Ok(v) = value.parse::<usize>() {
                            state.threads = v.clamp(1, 512);
                        }
                    } else if name.eq_ignore_ascii_case("OwnBook") {
                        state.own_book = !value.eq_ignore_ascii_case("false");
                    } else if name.eq_ignore_ascii_case("Hash") {
                        if let Ok(mb) = value.parse::<usize>() {
                            let mb = mb.clamp(1, 4096);
                            // TTEntry = 16 bytes, so MB * 65536 entries;
                            // TT::new takes the power-of-two bucket count.
                            let entries = mb * 65536;
                            let pow = (usize::BITS - 1 - entries.leading_zeros()) as u32;
                            state.tt = Arc::new(TT::new(pow));
                        }
                    }
                }
            }
            "isready" => {
                outln("readyok".to_string());
            }
            "ucinewgame" => {
                if let Some(a) = &state.abort {
                    a.store(true, Ordering::Relaxed);
                }
                drain_result(&mut state);
                state.tt.clear();
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
                let board = state.board.clone();
                let threads = state.threads;
                // Fresh decorrelator per move so openings vary across games
                // (and even within replay/analysis loops).
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0);
                let (tx, rx) = mpsc::channel();
                let state_tt = state.tt.clone();
                // The book only helps real game moves, never `go depth`/`nodes`/
                // `infinite` analysis requests.
                let use_book = state.own_book && (limits.wtime > 0 || limits.btime > 0);
                state.abort = Some(abort.clone());
                state.receiver = Some(rx);
                thread::spawn(move || {
                    let result = Searcher::think_parallel(board, limits, threads, false, abort.clone(), seed, use_book, state_tt);
                    let _ = tx.send(result);
                    // Always emit bestmove — including when the search was
                    // aborted by `stop`.  Suppressing it on abort left GUIs
                    // waiting forever after interrupting a search (UCI
                    // requires a bestmove answer to both `go` and `stop`).
                    let mv = result.best.map(|m| m.uci()).unwrap_or_else(|| "(none)".to_string());
                    outln(format!("bestmove {}", mv));
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