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
use crate::bitboard::*;
use crate::movegen::{generate_legal, MoveList};

fn piece_from_char(c: char) -> Option<usize> {
    match c {
        'K' => Some(KING),
        'Q' => Some(QUEEN),
        'R' => Some(ROOK),
        'B' => Some(BISHOP),
        'N' => Some(KNIGHT),
        _ => None,
    }
}

fn file_idx(c: char) -> Option<usize> {
    match c {
        'a'..='h' => Some((c as u8 - b'a') as usize),
        _ => None,
    }
}

pub fn san_to_move(board: &mut Board, san: &str) -> Option<Move> {
    let mut s = san;
    while s.ends_with('!') || s.ends_with('?') {
        s = &s[..s.len() - 1];
    }
    if s.ends_with('+') || s.ends_with('#') {
        s = &s[..s.len() - 1];
    }

    let mut legal = MoveList::new();
    generate_legal(board, &mut legal);
    let slice = legal.as_slice();

    let lower = s.to_ascii_lowercase();
    if lower == "o-o" || lower == "0-0" {
        return castle_move(slice, true);
    }
    if lower == "o-o-o" || lower == "0-0-0" {
        return castle_move(slice, false);
    }

    let mut body = s;
    let mut promo = NO_PIECE;
    if let Some(eq) = s.rfind('=') {
        let Some(pc) = s[eq + 1..].chars().next() else {
            return None;
        };
        promo = piece_from_char(pc)?;
        body = &s[..eq];
    }

    let bytes = body.as_bytes();
    let n = bytes.len();
    if n < 2 {
        return None;
    }
    let dfile = file_idx(bytes[n - 2] as char)?;
    let drank = bytes[n - 1];
    if !(b'1'..=b'8').contains(&drank) {
        return None;
    }
    let to = ((drank - b'1') as usize) * 8 + dfile;

    let (piece, dis) = match bytes[0] as char {
        'K' => (KING, &body[1..n - 2]),
        'Q' => (QUEEN, &body[1..n - 2]),
        'R' => (ROOK, &body[1..n - 2]),
        'B' => (BISHOP, &body[1..n - 2]),
        'N' => (KNIGHT, &body[1..n - 2]),
        _ => (PAWN, &body[..n - 2]),
    };

    let mut ff = None;
    let mut rr = None;
    for c in dis.chars() {
        if c == 'x' {
            continue;
        }
        if let Some(f) = file_idx(c) {
            ff = Some(f);
        } else if ('1'..='8').contains(&c) {
            rr = Some((c as u8 - b'1') as usize);
        } else {
            return None;
        }
    }

    let mut found: Option<Move> = None;
    for &m in legal.iter() {
        if m.piece != piece || m.to != to || m.promo != promo {
            continue;
        }
        if let Some(f) = ff {
            if file_of(m.from) != f {
                continue;
            }
        }
        if let Some(r) = rr {
            if rank_of(m.from) != r {
                continue;
            }
        }
        if found.is_some() {
            return None;
        }
        found = Some(m);
    }
    found
}

fn castle_move(legal: &[Move], kingside: bool) -> Option<Move> {
    let mut found: Option<Move> = None;
    for m in legal {
        if m.flags & M_CASTLE == 0 {
            continue;
        }
        let df = file_of(m.to);
        if (kingside && df == 6) || (!kingside && df == 2) {
            if found.is_some() {
                return None;
            }
            found = Some(*m);
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(b: &mut Board, san: &str, expected: &str) {
        let mv = san_to_move(b, san).unwrap_or_else(|| panic!("no move for {}", san));
        assert_eq!(mv.uci(), expected, "san {}", san);
        b.make(mv);
    }

    #[test]
    fn san_basic_moves() {
        let mut b = Board::startpos();
        m(&mut b, "e4", "e2e4");
        m(&mut b, "c5", "c7c5");
        m(&mut b, "Nf3", "g1f3");
        m(&mut b, "d6", "d7d6");
        m(&mut b, "d4", "d2d4");
        m(&mut b, "cxd4", "c5d4");
        m(&mut b, "Nxd4", "f3d4");
        m(&mut b, "Nf6", "g8f6");
        m(&mut b, "Nc3", "b1c3");
    }

    #[test]
    fn san_pawn_capture() {
        let mut b = Board::startpos();
        m(&mut b, "e4", "e2e4");
        m(&mut b, "e5", "e7e5");
        m(&mut b, "d4", "d2d4");
        let mv = san_to_move(&mut b, "exd4").unwrap();
        assert_eq!(mv.uci(), "e5d4");
    }

    #[test]
    fn san_castles() {
        let mut b = Board::startpos();
        for u in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "b1c3", "f8c5"] {
            let mv = b.uci_move(u).unwrap();
            b.make(mv);
        }
        let mv = san_to_move(&mut b, "O-O").unwrap();
        assert_eq!(mv.uci(), "e1g1");
        b.make(mv);
        let mv = san_to_move(&mut b, "O-O").unwrap();
        assert_eq!(mv.uci(), "e8g8");
    }

    #[test]
    fn san_disambiguation() {
        let mut b = Board::startpos();
        for u in ["g1f3", "d7d5", "d2d4", "g8f6", "c2c4", "e7e6", "b2b3", "f8e7"] {
            let mv = b.uci_move(u).unwrap();
            b.make(mv);
        }
        assert_eq!(san_to_move(&mut b, "Nbd2").unwrap().uci(), "b1d2");
        assert_eq!(san_to_move(&mut b, "N1d2").unwrap().uci(), "b1d2");
        assert!(san_to_move(&mut b, "Nd2").is_none());
    }

    #[test]
    fn san_promotion() {
        let mut b = Board::from_fen("3r4/4P3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mv = san_to_move(&mut b, "exd8=Q+").unwrap();
        assert_eq!(mv.uci(), "e7d8q");
        let mut b = Board::from_fen("8/4P3/8/8/4k3/8/8/4K3 w - - 0 1").unwrap();
        let mv = san_to_move(&mut b, "e8=Q").unwrap();
        assert_eq!(mv.uci(), "e7e8q");
    }
}