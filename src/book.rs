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
use crate::movegen::{generate_legal, MoveList};
use std::collections::HashMap;
use std::sync::OnceLock;

struct Pick {
    m: Move,
    w: u32,
}

pub struct Book {
    map: HashMap<u64, Vec<Pick>>,
}

fn book_lines() -> &'static [&'static str] {
    &[
        // e4 e5: Ruy Lopez (closed and open variants)
        "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8",
        "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 e8g8 c2c3 d7d5",
        "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8c5 c2c3 d7d6 d2d4 c5b6 a4b3 e8g8",
        // e4 e5: Italian / Scotch / Petrov
        "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d3 d7d6 d3d4 c5b6 c4b3 e8g8",
        "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5 c2c3 d7d6 d3d4 c5b6 c4b3 e8g8",
        "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6 d4c6 b7c6 e4e5 f6d5 c2c4 d5b6",
        "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4 d2d4 d6d5",
        // Sicilian (Najdorf, Dragon, Taimanov, Classical, Sveshnikov, Alapin)
        "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4f3 f8e7 f2f4",
        "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6 c1e3 f8g7 f2f3 e8g8 d1d2",
        "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 g8f6 b1c3 d7d6 f1e2 a7a6 c1e3 f8e7 d1d2 e8g8 e1c1 b8c6",
        "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 d7d6 c1g5 e7e6 d1d2 f8e7 e1c1 e8g8",
        "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5 d4b5 d7d6 c1g5 a7a6 b5a3 b7b5",
        "e2e4 c7c5 g1f3 g8f6 e4e5 f6d5 b1c3 d5c3 d2c3 d7d6 e5d6 f8d6",
        // Caro-Kann, French, Scandi
        "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6 h2h4 h7h6 g1f3 g8f6",
        "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3 e7e6 f1e2 c6c5",
        "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 e4e5 f6d7 f2f4 c7c5 g1f3 b8c6 c1e3 c5d4 f3d4",
        "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 c7c5 a2a3 b4c3 b2c3 d8c7 g1f3 b8c6 f1d3",
        "e2e4 d7d5 e4d5 d8d5 b1c3 d5e6",
        // KID / Pirc / Alekhine
        "e2e4 g7g6 d2d4 f8g7 b1c3 g8f6 g1f3 d7d6 f1e2 e8g8 e1g1",
        "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 g1f3 f8g7 f1e2 e8g8 e1g1",
        "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6 c2c4 d5b6 g1f3 g7g6",
        // d4 d5: QGD, Slav, Catalan, London
        "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3 b8d7 f1d3",
        "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4 a2a4 c8f5 e2e3 e7e6 f1c4",
        "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 e7e6 c1g5 h7h6 g5h4 d5c4 e2e4 g7g5 h4g3",
        "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6 e2e3 e7e6 f1c4 c7c5 e1g1",
        "d2d4 d7d5 g1f3 g8f6 c1f4 e7e6 e2e3 f8d6 f4g3",
        // d4 Nf6: Nimzo, QID, Gruenfeld, Benko, Benoni, KID, Torre
        "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8 f1d3 d7d5",
        "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3 c8a6 b2b3 f8b4 c1d2 b4d2",
        "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4 d5c3 b2c3 f8g7 f1c4 e8g8 g1e2 c7c5",
        "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 g1f3 d7d6 g2g3 e8g8 f1g2 b8d7 e1g1",
        "d2d4 g8f6 c2c4 c7c5 d4d5 b7b5 c4b5 a7a6 b5a6 c8a6 b1c3 d7d6 e2e4 a6f1",
        "d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7 g1f3 e8g8 e1g1 b8d7",
        "d2d4 e7e6 c2c4 b7b6 e2e3 c8b7 g1f3 g8f6 b1c3 f8b4 f1d3",
        // English / Reti
        "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5 c4d5 f6d5 f1g2 d5c3 b2c3 b8c6",
        "c2c4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7 g1f3 e8g8",
        "g1f3 d7d5 g2g3 g8f6 f1g2 g7g6 e1g1 f8g7 d2d4 e8g8 c2c4",
    ]
}

static BOOK: OnceLock<Book> = OnceLock::new();

/// `seed` decorrelates book picks across games so the same position does not
/// always produce the same move. Pass 0 for deterministic bench/analysis runs.
pub fn probe(board: &mut Board, legal: &[Move], seed: u64) -> Option<Move> {
    let book = BOOK.get_or_init(Book::load);
    book.lookup(board.key, legal, seed)
}

impl Book {
    pub fn load() -> Book {
        let mut map: HashMap<u64, Vec<Pick>> = HashMap::new();
        for line in book_lines() {
            let mut b = Board::startpos();
            for token in line.split_whitespace() {
                let Some(m) = b.uci_move(token) else {
                    break;
                };
                let mut legal = MoveList::new();
                generate_legal(&mut b, &mut legal);
                if !legal.contains(&m) {
                    break;
                }
                let key = b.key;
                map.entry(key).or_default().push(Pick { m, w: 1 });
                b.make(m);
            }
        }
        Book { map }
    }

    pub fn lookup(&self, key: u64, legal: &[Move], seed: u64) -> Option<Move> {
        let entries = self.map.get(&key)?;
        let mut x = key ^ seed ^ 0x9E3779B97F4A7C15u64;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        let total: u32 = entries.iter().map(|p| p.w).sum();
        let mut v = (x % total as u64) as u32;
        for p in entries {
            if legal.contains(&p.m) {
                if v < p.w {
                    return Some(p.m);
                }
                v -= p.w;
            }
        }
        for p in entries {
            if legal.contains(&p.m) {
                return Some(p.m);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn book_loads_and_probes_startpos() {
        let book = Book::load();
        assert!(book.map.len() > 250, "book too small: {}", book.map.len());
        let mut b = Board::startpos();
        let mut legal = MoveList::new();
        generate_legal(&mut b, &mut legal);
        let m = book.lookup(b.key, legal.as_slice(), 0);
        assert!(m.is_some());
        let mv = m.unwrap();
        assert!(legal.contains(&mv));
    }

    #[test]
    fn book_seeds_pick_varied_first_moves() {
        let book = Book::load();
        let mut b = Board::startpos();
        let mut legal = MoveList::new();
        generate_legal(&mut b, &mut legal);
        let mut seen: Vec<Move> = Vec::new();
        for seed in 0..96u64 {
            let m = book.lookup(b.key, legal.as_slice(), seed).unwrap();
            if !seen.contains(&m) {
                seen.push(m);
            }
        }
        // Several distinct openings must be reachable purely by seeding.
        assert!(seen.len() >= 3, "only {} distinct start moves from book", seen.len());
    }

    #[test]
    fn book_probes_sicilian_position() {
        let book = Book::load();
        let mut b = Board::startpos();
        for t in ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"] {
            let mv = b.uci_move(t).unwrap();
            let mut legal = MoveList::new();
            generate_legal(&mut b, &mut legal);
            assert!(legal.contains(&mv));
            b.make(mv);
        }
        let mut legal = MoveList::new();
        generate_legal(&mut b, &mut legal);
        let m = book.lookup(b.key, legal.as_slice(), 0);
        assert!(m.is_some(), "expected a book move in the Open Sicilian");
        assert!(legal.contains(&m.unwrap()));
    }
}