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
use crate::movegen::*;

pub fn perft(b: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut list = MoveList::new();
    generate_legal(b, &mut list);
    if depth == 1 {
        return list.len as u64;
    }
    let mut total = 0u64;
    for &m in list.iter() {
        let _ = b.make(m);
        total += perft(b, depth - 1);
        b.unmake(m);
    }
    total
}

pub fn divide(b: &mut Board, depth: u32) {
    let mut list = MoveList::new();
    generate_legal(b, &mut list);
    let mut total = 0u64;
    for &m in list.iter() {
        let _ = b.make(m);
        let c = if depth == 1 { 1 } else { perft(b, depth - 1) };
        b.unmake(m);
        total += c;
        println!("{} : {}", m.uci(), c);
    }
    println!();
    println!("Total: {}", total);
}

#[cfg(test)]
mod tests {
    use super::*;

    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    const POS3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
    const POS4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
    const POS5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
    const POS6: &str = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

    #[test]
    fn perft_kiwipete() {
        let mut b = Board::from_fen(KIWIPETE).unwrap();
        assert_eq!(perft(&mut b, 1), 48);
        assert_eq!(perft(&mut b, 2), 2039);
        assert_eq!(perft(&mut b, 3), 97862);
    }

    #[test]
    fn perft_pos3() {
        let mut b = Board::from_fen(POS3).unwrap();
        assert_eq!(perft(&mut b, 1), 14);
        assert_eq!(perft(&mut b, 2), 191);
        assert_eq!(perft(&mut b, 3), 2812);
        assert_eq!(perft(&mut b, 4), 43238);
    }

    #[test]
    fn perft_pos4() {
        let mut b = Board::from_fen(POS4).unwrap();
        assert_eq!(perft(&mut b, 1), 6);
        assert_eq!(perft(&mut b, 2), 264);
        assert_eq!(perft(&mut b, 3), 9467);
    }

    #[test]
    fn perft_pos5() {
        let mut b = Board::from_fen(POS5).unwrap();
        assert_eq!(perft(&mut b, 1), 44);
        assert_eq!(perft(&mut b, 2), 1486);
        assert_eq!(perft(&mut b, 3), 62379);
    }

    #[test]
    fn perft_pos6() {
        let mut b = Board::from_fen(POS6).unwrap();
        assert_eq!(perft(&mut b, 1), 46);
        assert_eq!(perft(&mut b, 2), 2079);
        assert_eq!(perft(&mut b, 3), 89890);
    }
}