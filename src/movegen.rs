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

use crate::bitboard::*;
use crate::board::*;
use crate::eval::PIECE_VALUES;


pub fn generate_pseudo(b: &mut Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(64);
    let us = b.side;
    let them = us ^ 1;
    let own = b.occ[us];
    let enemy = b.occ[them];
    let all = b.all;

    let mut kn = b.pieces[us][KNIGHT];
    while kn != 0 {
        let sq = kn.trailing_zeros() as usize;
        kn &= kn - 1;
        let mut t = KNIGHT_ATTACKS[sq] & !own;
        while t != 0 {
            let to = t.trailing_zeros() as usize;
            t &= t - 1;
            moves.push(Move::normal(sq, to, KNIGHT, b.piece_on(to)));
        }
    }

    let mut bi = b.pieces[us][BISHOP];
    while bi != 0 {
        let sq = bi.trailing_zeros() as usize;
        bi &= bi - 1;
        let mut t = bishop_attacks(all, sq) & !own;
        while t != 0 {
            let to = t.trailing_zeros() as usize;
            t &= t - 1;
            moves.push(Move::normal(sq, to, BISHOP, b.piece_on(to)));
        }
    }

    let mut ro = b.pieces[us][ROOK];
    while ro != 0 {
        let sq = ro.trailing_zeros() as usize;
        ro &= ro - 1;
        let mut t = rook_attacks(all, sq) & !own;
        while t != 0 {
            let to = t.trailing_zeros() as usize;
            t &= t - 1;
            moves.push(Move::normal(sq, to, ROOK, b.piece_on(to)));
        }
    }

    let mut qu = b.pieces[us][QUEEN];
    while qu != 0 {
        let sq = qu.trailing_zeros() as usize;
        qu &= qu - 1;
        let mut t = queen_attacks(all, sq) & !own;
        while t != 0 {
            let to = t.trailing_zeros() as usize;
            t &= t - 1;
            moves.push(Move::normal(sq, to, QUEEN, b.piece_on(to)));
        }
    }

    let mut ki = b.pieces[us][KING];
    while ki != 0 {
        let sq = ki.trailing_zeros() as usize;
        ki &= ki - 1;
        let mut t = KING_ATTACKS[sq] & !own;
        while t != 0 {
            let to = t.trailing_zeros() as usize;
            t &= t - 1;
            moves.push(Move::normal(sq, to, KING, b.piece_on(to)));
        }
    }

    let mut pw = b.pieces[us][PAWN];
    while pw != 0 {
        let sq = pw.trailing_zeros() as usize;
        pw &= pw - 1;
        let r = rank_of(sq);

        if us == WHITE {
            let one = sq + 8;
            if one < 64 && all & bit(one) == 0 {
                if r == 6 {
                    for p in [KNIGHT, BISHOP, ROOK, QUEEN] {
                        moves.push(Move::promo(sq, one, p));
                    }
                } else {
                    moves.push(Move::normal(sq, one, PAWN, NO_PIECE));
                    if r == 1 && all & bit(sq + 16) == 0 {
                        moves.push(Move::normal(sq, sq + 16, PAWN, NO_PIECE));
                    }
                }
            }
        } else {
            let one = sq - 8;
            if all & bit(one) == 0 {
                if r == 1 {
                    for p in [KNIGHT, BISHOP, ROOK, QUEEN] {
                        moves.push(Move::promo(sq, one, p));
                    }
                } else {
                    moves.push(Move::normal(sq, one, PAWN, NO_PIECE));
                    if r == 6 && all & bit(sq - 16) == 0 {
                        moves.push(Move::normal(sq, sq - 16, PAWN, NO_PIECE));
                    }
                }
            }
        }

        let mut atk = PAWN_ATTACKS[us][sq] & enemy;
        while atk != 0 {
            let to = atk.trailing_zeros() as usize;
            atk &= atk - 1;
            if us == WHITE && rank_of(to) == 7 {
                for p in [KNIGHT, BISHOP, ROOK, QUEEN] {
                    moves.push(Move {
                        from: sq,
                        to,
                        piece: PAWN,
                        captured: PAWN,
                        promo: p,
                        flags: M_PROMO,
                    });
                }
            } else if us == BLACK && rank_of(to) == 0 {
                for p in [KNIGHT, BISHOP, ROOK, QUEEN] {
                    moves.push(Move {
                        from: sq,
                        to,
                        piece: PAWN,
                        captured: PAWN,
                        promo: p,
                        flags: M_PROMO,
                    });
                }
            } else {
                moves.push(Move::normal(sq, to, PAWN, b.piece_on(to)));
            }
        }
    }

    if let Some(e) = b.ep {
        let attackers = PAWN_ATTACKS[us ^ 1][e] & b.pieces[us][PAWN];
        let mut a = attackers;
        while a != 0 {
            let sq = a.trailing_zeros() as usize;
            a &= a - 1;
            moves.push(Move::ep(sq, e));
        }
    }

    if !b.in_check(us) {
        if us == WHITE {
            if b.castle & WK != 0
                && all & bit(5) == 0
                && all & bit(6) == 0
                && b.pieces[us][ROOK] & bit(7) != 0
                && !b.attacked(them, 5)
            {
                moves.push(Move::castle(4, 6));
            }
            if b.castle & WQ != 0
                && all & bit(1) == 0
                && all & bit(2) == 0
                && all & bit(3) == 0
                && b.pieces[us][ROOK] & bit(0) != 0
                && !b.attacked(them, 3)
            {
                moves.push(Move::castle(4, 2));
            }
        } else {
            if b.castle & BK != 0
                && all & bit(61) == 0
                && all & bit(62) == 0
                && b.pieces[us][ROOK] & bit(63) != 0
                && !b.attacked(them, 61)
            {
                moves.push(Move::castle(60, 62));
            }
            if b.castle & BQ != 0
                && all & bit(57) == 0
                && all & bit(58) == 0
                && all & bit(59) == 0
                && b.pieces[us][ROOK] & bit(56) != 0
                && !b.attacked(them, 59)
            {
                moves.push(Move::castle(60, 58));
            }
        }
    }

    moves
}

pub fn generate_legal(b: &mut Board) -> Vec<Move> {
    let moves = generate_pseudo(b);
    moves
        .into_iter()
        .filter(|&m| {
            let legal = b.make(m);
            b.unmake(m);
            legal
        })
        .collect()
}

pub fn generate_captures(b: &mut Board) -> Vec<Move> {
    generate_pseudo(b)
        .into_iter()
        .filter(|m| m.captured != NO_PIECE || m.promo != NO_PIECE)
        .collect()
}

fn attackers_mask(b: &Board, occ: u64, sq: usize, color: usize) -> u64 {
    let mut m = PAWN_ATTACKS[color ^ 1][sq] & b.pieces[color][PAWN];
    m |= KNIGHT_ATTACKS[sq] & b.pieces[color][KNIGHT];
    m |= KING_ATTACKS[sq] & b.pieces[color][KING];
    let d = bishop_attacks(occ, sq);
    let o = rook_attacks(occ, sq);
    m |= d & (b.pieces[color][BISHOP] | b.pieces[color][QUEEN]);
    m |= o & (b.pieces[color][ROOK] | b.pieces[color][QUEEN]);
    m
}

fn see_rec(b: &Board, sq: usize, side: usize, occ: u64, on_sq: i32, can_pass: bool) -> i32 {
    let atk = attackers_mask(b, occ, sq, side);
    let mut att = None;
    for pc in PAWN..=QUEEN {
        let p = atk & b.pieces[side][pc] & occ;
        if p != 0 {
            att = Some((p.trailing_zeros() as usize, PIECE_VALUES[pc]));
            break;
        }
    }
    let (att_sq, att_val) = match att {
        Some(x) => x,
        None => return 0,
    };
    let occ2 = occ & !bit(att_sq);
    let resp = see_rec(b, sq, side ^ 1, occ2, att_val, true);
    let gain = on_sq - resp;
    if can_pass {
        gain.max(0)
    } else {
        gain
    }
}

pub fn see(b: &Board, sq: usize, stm: usize) -> i32 {
    if b.piece_on(sq) == NO_PIECE {
        return 0;
    }
    let occ = b.all & !bit(sq);
    see_rec(b, sq, stm, occ, PIECE_VALUES[b.piece_on(sq)], false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_move_count() {
        let mut b = Board::startpos();
        assert_eq!(generate_legal(&mut b).len(), 20);
        assert_eq!(crate::perft::perft(&mut b, 1), 20);
        assert_eq!(crate::perft::perft(&mut b, 2), 400);
    }

    #[test]
    fn queen_recapture_generated() {
        let mut b = Board::from_fen(
            "rnb1kbnr/pppp1ppp/4p3/8/3P1q2/6P1/PPP2P1P/RNBQKBNR w KQkq - 0 1",
        )
        .unwrap();
        let moves = generate_legal(&mut b);
        assert!(
            moves.iter().any(|m| m.from == 22 && m.to == 29 && m.piece == PAWN),
            "g3xf4 missing from legal moves"
        );
    }

    #[test]
    fn see_winning_recapture_is_positive() {
        let b = Board::from_fen("4k3/8/8/3p4/4n3/5P2/8/4K3 w - - 0 1").unwrap();
        let se = see(&b, 28, WHITE);
        assert!(se >= 200 && se <= 400, "pawn recaptures knight: got {}", se);
    }

    #[test]
    fn see_equal_trade_is_zero() {
        let b = Board::from_fen("4k3/8/5n2/8/4n3/2N5/8/4K3 w - - 0 1").unwrap();
        let se = see(&b, 28, WHITE);
        assert_eq!(se, 0, "knight for knight: got {}", se);
    }

    #[test]
    fn see_losing_capture_is_negative() {
        let b = Board::from_fen("4k3/8/8/3p4/4n3/8/8/4QK2 w - - 0 1").unwrap();
        let se = see(&b, 28, WHITE);
        assert!(se < 0, "queen takes defended knight: got {}", se);
    }
}