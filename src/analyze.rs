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

use crate::board::{Board, Move};
use crate::eval::MATE;
use crate::san::san_to_move;
use crate::search::{SearchLimits, Searcher};
use std::fs;

const WHITE: usize = crate::bitboard::WHITE;

struct Row {
    san: String,
    mover: usize,
    played: Move,
    alt: Option<Move>,
    before: i32,
    after: i32,
}

fn pretty(s: i32) -> String {
    if s > MATE - 200 {
        format!("+M{}", (MATE - s + 1) / 2)
    } else if s < -(MATE - 200) {
        format!("-M{}", (MATE + s + 1) / 2)
    } else {
        format!("{:+.2}", s as f64 / 100.0)
    }
}

fn swing(before: i32, after: i32, mover: usize) -> i32 {
    if mover == WHITE {
        after - before
    } else {
        before - after
    }
}

fn tag(before: i32, after: i32, mover: usize) -> &'static str {
    if swing(before, after, mover) <= -300 {
        "??"
    } else if swing(before, after, mover) <= -150 {
        "?"
    } else if swing(before, after, mover) <= -80 {
        "?!"
    } else if swing(before, after, mover) >= 120 {
        "!"
    } else {
        ""
    }
}

fn search_pos(board: &Board, depth: u32) -> (i32, Option<Move>) {
    let mut s = Searcher::new(board.clone());
    s.quiet = true;
    let res = s.think(SearchLimits {
        depth,
        ..SearchLimits::default()
    });
    (res.score, res.best)
}

fn extract_start_move(text: &str) -> (Option<String>, Vec<String>) {
    let mut fen: Option<String> = None;
    let mut moves: Vec<String> = Vec::new();
    let mut collected = String::new();

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            if let Some(rest) = line.strip_prefix("[FEN") {
                if let Some(start) = rest.find('"') {
                    if let Some(end) = rest[start + 1..].find('"') {
                        fen = Some(rest[start + 1..start + 1 + end].to_string());
                    }
                }
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        collected.push_str(line);
        collected.push(' ');
    }

    let mut clean = String::new();
    for c in collected.chars() {
        if c == '{' || c == '}' {
            clean.push(' ');
        } else {
            clean.push(c);
        }
    }

    for tok in clean.split_whitespace() {
        if matches!(tok, "1-0" | "0-1" | "1/2-1/2" | "*") {
            break;
        }
        if tok.ends_with('.') {
            continue;
        }
        let san = if tok.contains('.') {
            let idx = tok.find('.').unwrap();
            let rest = &tok[idx + 1..];
            if rest.is_empty() {
                continue;
            }
            rest
        } else {
            tok
        };
        if san.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        moves.push(san.to_string());
    }
    (fen, moves)
}

pub fn analyze(path: &str, depth: u32) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let (fen, sans) = extract_start_move(&text);

    let mut board = match &fen {
        Some(f) => Board::from_fen(f).map_err(|e| format!("bad FEN: {}", e))?,
        None => Board::startpos(),
    };

    let mut rows: Vec<Row> = Vec::new();
    let mut prev: Option<(i32, Option<Move>, usize, Move)> = None;
    let mut k = 0usize;

    for san in &sans {
        let mover = board.side;
        let (before, best) = search_pos(&board, depth);
        let played = san_to_move(&mut board, san)
            .ok_or_else(|| format!("could not parse SAN move '{}'", san))?;
        board.make(played);

        if k > 0 {
            let (pb, pbm, p_mover, p_played) = prev.unwrap();
            rows.push(Row {
                san: sans[k - 1].clone(),
                mover: p_mover,
                played: p_played,
                alt: pbm,
                before: pb,
                after: before,
            });
        }
        prev = Some((before, best, mover, played));
        k += 1;
    }

    if let Some((lb, lm, l_mover, l_played)) = prev {
        let (last_after, _) = search_pos(&board, depth);
        rows.push(Row {
            san: sans[k - 1].clone(),
            mover: l_mover,
            played: l_played,
            alt: lm,
            before: lb,
            after: last_after,
        });
    }

    println!("Corvus analysis: {} plies at depth {}", sans.len(), depth);
    match &fen {
        Some(f) => println!("Start FEN: {}", f),
        None => println!("Start FEN: startpos"),
    }
    println!("{}", "-".repeat(60));
    println!(" ply  Move            Eval after   Swing   Note");
    println!("{}", "-".repeat(60));

    for (i, r) in rows.iter().enumerate() {
        let disp = if r.mover == WHITE {
            format!("{:>8}", r.san)
        } else {
            format!("…{:>7}", r.san)
        };
        let sw = swing(r.before, r.after, r.mover);
        println!(
            "{:>4}  {:<14}  {:>10}  {:>6}  {}",
            i + 1,
            disp,
            pretty(r.after),
            format!("{:+.0}", sw as f64),
            tag(r.before, r.after, r.mover)
        );
    }
    println!("{}", "-".repeat(60));

    let mut tagged: Vec<(usize, &Row, &str)> = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let t = tag(r.before, r.after, r.mover);
        if t == "?" || t == "??" {
            tagged.push((i + 1, r, t));
        }
    }
    if tagged.is_empty() {
        println!("No clear mistakes found.");
        return Ok(());
    }

    tagged.sort_by_key(|(_, r, _)| -swing(r.before, r.after, r.mover));
    println!("\nWorst moves (alternatives are engine's choice at that position):");
    for (ply, r, t) in tagged.iter().take(10) {
        let alt = match r.alt {
            Some(m) if m != r.played => format!("   better: {}", m.uci()),
            _ => String::new(),
        };
        let drop = swing(r.before, r.after, r.mover);
        println!(
            "  ply {:<3} {} {}   eval change {}{}",
            ply,
            r.san,
            t,
            format!("{:+.0}", drop as f64),
            alt
        );
    }
    Ok(())
}
