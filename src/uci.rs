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
    // UCI options/state
    ponder: bool,
    multi_pv: usize,
    skill_level: i32,
    move_overhead: u64,
    nodestime: u64,
    uci_chess960: bool,
    uci_limit_strength: bool,
    uci_elo: i32,
    uci_show_wdl: bool,
    syzygy_path: String,
    syzygy_probe_depth: i32,
    syzygy_50_move_rule: bool,
    syzygy_probe_limit: i32,
    eval_file: String,
    hash_mb: usize,
    clear_hash_pending: bool,
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
        ponder: false,
        multi_pv: 1,
        skill_level: 20,
        move_overhead: 10,
        nodestime: 0,
        uci_chess960: false,
        uci_limit_strength: false,
        uci_elo: 1320,
        uci_show_wdl: false,
        syzygy_path: String::new(),
        syzygy_probe_depth: 1,
        syzygy_50_move_rule: true,
        syzygy_probe_limit: 7,
        eval_file: String::new(),
        hash_mb: 256,
        clear_hash_pending: false,
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
                outln("option name Clear Hash type button".to_string());
                outln("option name Ponder type check default false".to_string());
                outln("option name MultiPV type spin default 1 min 1 max 256".to_string());
                outln("option name Skill Level type spin default 20 min 0 max 20".to_string());
                outln("option name Move Overhead type spin default 10 min 0 max 5000".to_string());
                outln("option name nodestime type spin default 0 min 0 max 10000".to_string());
                outln("option name UCI_Chess960 type check default false".to_string());
                outln("option name UCI_LimitStrength type check default false".to_string());
                outln("option name UCI_Elo type spin default 1320 min 1320 max 3190".to_string());
                outln("option name UCI_ShowWDL type check default false".to_string());
                outln("option name SyzygyPath type string default <empty>".to_string());
                outln("option name SyzygyProbeDepth type spin default 1 min 1 max 100".to_string());
                outln("option name Syzygy50MoveRule type check default true".to_string());
                outln("option name SyzygyProbeLimit type spin default 7 min 0 max 7".to_string());
                outln(format!(
                    "option name EvalFile type string default {}",
                    crate::eval::DEFAULT_NNUE_FILE
                ));
                outln("option name OwnBook type check default true".to_string());
                // Resolve the NNUE network before the GUI starts timing us, so
                // the first move of a game isn't charged the file-read time.
                crate::eval::warm_network();
                // Auto-load tablebases from the conventional locations so
                // Syzygy works with zero GUI setup.  An explicit `SyzygyPath`
                // (or CORVUS_SYZYGY_PATH) still takes priority.
                if state.syzygy_path.trim().is_empty() {
                    if let Some(dir) = crate::syzygy::discover_default() {
                        let largest = crate::syzygy::max_pieces();
                        state.syzygy_path = dir.clone();
                        eprintln!("info string Syzygy tablebases up to {largest} pieces from {dir}");
                    }
                }
                outln("uciok".to_string());
            }
            "setoption" => {
                if let Some(vpos) = tokens.iter().position(|&t| t == "value") {
                    // `setoption name <NAME> value <VALUE>`: skip the literal
                    // `name` keyword, otherwise every lookup below would compare
                    // against "name Hash" and never match.
                    let name_start = if tokens.get(1) == Some(&"name") { 2 } else { 1 };
                    let name = if name_start <= vpos {
                        tokens[name_start..vpos].join(" ")
                    } else {
                        String::new()
                    };
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
                            state.hash_mb = mb;
                        }
                    } else if name.eq_ignore_ascii_case("Clear Hash") {
                        state.tt.clear();
                        state.clear_hash_pending = true;
                    } else if name.eq_ignore_ascii_case("Ponder") {
                        state.ponder = !value.eq_ignore_ascii_case("false");
                    } else if name.eq_ignore_ascii_case("MultiPV") {
                        if let Ok(v) = value.parse::<usize>() {
                            state.multi_pv = v.clamp(1, 256);
                        }
                    } else if name.eq_ignore_ascii_case("Skill Level") {
                        if let Ok(v) = value.parse::<i32>() {
                            state.skill_level = v.clamp(0, 20);
                        }
                    } else if name.eq_ignore_ascii_case("Move Overhead") {
                        if let Ok(v) = value.parse::<u64>() {
                            state.move_overhead = v.min(5000);
                        }
                    } else if name.eq_ignore_ascii_case("nodestime") {
                        if let Ok(v) = value.parse::<u64>() {
                            state.nodestime = v.min(10000);
                        }
                    } else if name.eq_ignore_ascii_case("UCI_Chess960") {
                        state.uci_chess960 = !value.eq_ignore_ascii_case("false");
                    } else if name.eq_ignore_ascii_case("UCI_LimitStrength") {
                        state.uci_limit_strength = !value.eq_ignore_ascii_case("false");
                    } else if name.eq_ignore_ascii_case("UCI_Elo") {
                        if let Ok(v) = value.parse::<i32>() {
                            state.uci_elo = v.clamp(1320, 3190);
                        }
                    } else if name.eq_ignore_ascii_case("UCI_ShowWDL") {
                        state.uci_show_wdl = !value.eq_ignore_ascii_case("false");
                    } else if name.eq_ignore_ascii_case("SyzygyPath") {
                        state.syzygy_path = value.clone();
                        // Load now rather than at the first probe, so a bad path
                        // is reported to the GUI instead of silently costing
                        // endgame strength.
                        if state.syzygy_path.trim().is_empty() {
                            // Empty disables probing for this session (Fathom
                            // cannot unload tables that are already mapped).
                            eprintln!("info string SyzygyProbe disabled (empty SyzygyPath)");
                        } else if let Some(already) = crate::syzygy::loaded_path() {
                            if already != state.syzygy_path {
                                // A startup auto-load already pinned Fathom to a
                                // different directory.  Say so instead of
                                // pretending the new path was applied.
                                eprintln!(
                                    "info string SyzygyPath '{0}' ignored: '{1}' already loaded \
                                     (tablebases cannot be reloaded in-process)",
                                    state.syzygy_path, already
                                );
                                state.syzygy_path = already;
                            } else {
                                let largest = crate::syzygy::max_pieces();
                                eprintln!("info string Syzygy tablebases up to {largest} pieces");
                            }
                        } else {
                            let largest = crate::syzygy::init(&state.syzygy_path);
                            if largest == 0 {
                                eprintln!(
                                    "info string SyzygyPath '{}' loaded no tablebases",
                                    state.syzygy_path
                                );
                            } else {
                                eprintln!(
                                    "info string Syzygy tablebases up to {largest} pieces from {}",
                                    state.syzygy_path
                                );
                            }
                        }
                    } else if name.eq_ignore_ascii_case("SyzygyProbeDepth") {
                        if let Ok(v) = value.parse::<i32>() {
                            state.syzygy_probe_depth = v.clamp(1, 100);
                        }
                    } else if name.eq_ignore_ascii_case("Syzygy50MoveRule") {
                        state.syzygy_50_move_rule = !value.eq_ignore_ascii_case("false");
                    } else if name.eq_ignore_ascii_case("SyzygyProbeLimit") {
                        if let Ok(v) = value.parse::<i32>() {
                            state.syzygy_probe_limit = v.clamp(0, 7);
                        }
                    } else if name.eq_ignore_ascii_case("EvalFile") {
                        state.eval_file = value.clone();
                        crate::eval::set_eval_file(&value);
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
                let mut limits = parse_go(&tokens[1..]);
                limits.move_overhead = state.move_overhead;
                limits.syzygy_path = state.syzygy_path.clone();
                limits.syzygy_probe_depth = state.syzygy_probe_depth;
                limits.syzygy_probe_limit = state.syzygy_probe_limit.max(0) as u32;
                limits.syzygy_50_move_rule = state.syzygy_50_move_rule;
                let board = state.board.clone();
                let threads = state.threads;
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0);
                let (tx, rx) = mpsc::channel();
                let state_tt = state.tt.clone();
                let use_book = state.own_book && (limits.wtime > 0 || limits.btime > 0);
                limits.ponder = state.ponder || limits.ponder;
                state.abort = Some(abort.clone());
                state.receiver = Some(rx);
                let multi_pv = state.multi_pv;
                let skill_level = state.skill_level;
                let limit_strength = state.uci_limit_strength;
                let uci_elo = state.uci_elo;
                let show_wdl = state.uci_show_wdl;
                thread::spawn(move || {
                    // TODO: wire to searchers properly
                    let result = Searcher::think_parallel(board, limits.clone(), threads, false, abort.clone(), seed, use_book, state_tt.clone());
                    let _ = tx.send(result);
                    let hashfull = state_tt.hashfull();
                    let _ = (multi_pv, skill_level, limit_strength, uci_elo, show_wdl, hashfull);
                    let mv = result.best.map(|m| m.uci()).unwrap_or_else(|| "(none)".to_string());
                    if limits.ponder && result.best.is_some() {
                        outln(format!("bestmove {} ponder", mv));
                    } else {
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
            "ponderhit" => {
                // Accept ponderhit and continue/complete as normal
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
            "movestogo" => {
                i += 1;
                if let Some(t) = tokens.get(i).and_then(|s| s.parse::<u64>().ok()) {
                    l.movestogo = t;
                }
            }
            "mate" => {
                i += 1;
                if let Some(t) = tokens.get(i).and_then(|s| s.parse::<u32>().ok()) {
                    l.mate = t;
                }
            }
            "ponder" => l.ponder = true,
            "infinite" => l.infinite = true,
            "searchmoves" => {
                i += 1;
                while i < tokens.len() {
                    match tokens[i] {
                        "wtime" | "btime" | "winc" | "binc" | "movestogo" | "depth"
                        | "nodes" | "movetime" | "mate" | "ponder" | "infinite" => break,
                        _ => {
                            // Just record presence for now; filtering can be added later
                            i += 1;
                            continue;
                        }
                    }
                }
                i -= 1;
            }
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