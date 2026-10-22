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

/// Maximum number of pseudo-legal moves in any position (218 documented max).
pub const MAX_MOVES: usize = 256;

/// Stack-allocated move list — zero heap allocation on the hot path.
/// Entries are `MaybeUninit`: `new()` is O(1) (no 256-element initialization),
/// and only the pushed prefix is ever read.
pub struct MoveList {
    moves: [core::mem::MaybeUninit<Move>; MAX_MOVES],
    pub len: usize,
}

impl MoveList {
    #[inline(always)]
    pub fn new() -> Self {
        MoveList {
            moves: [core::mem::MaybeUninit::uninit(); MAX_MOVES],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        self.moves[self.len] = core::mem::MaybeUninit::new(m);
        self.len += 1;
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        // SAFETY: only `0..len` entries are written (each via `push`, which
        // initializes before incrementing `len`); indices below `len` are valid.
        unsafe { core::slice::from_raw_parts(self.moves.as_ptr() as *const Move, self.len) }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn iter(&self) -> core::slice::Iter<'_, Move> {
        self.as_slice().iter()
    }

    pub fn contains(&self, m: &Move) -> bool {
        self.as_slice().iter().any(|x| x == m)
    }

    pub fn retain<F: FnMut(&Move) -> bool>(&mut self, mut f: F) {
        // Swap-remove in place; `s` is derived from a raw pointer so touching
        // `self.len` afterwards does not conflict with `s`'s exclusive borrow.
        let s: &mut [Move] =
            unsafe { core::slice::from_raw_parts_mut(self.moves.as_mut_ptr() as *mut Move, MAX_MOVES) };
        let mut i = 0;
        while i < self.len {
            if f(&s[i]) {
                i += 1;
            } else {
                s[i] = s[self.len - 1];
                self.len -= 1;
            }
        }
    }
}

/// A scored move for move ordering — also stack-allocated, O(1) init.
#[derive(Clone, Copy)]
pub struct ScoredMove {
    pub mov: Move,
    pub score: i32,
}

/// Stack-allocated scored move list for move ordering.
pub struct ScoredMoveList {
    entries: [core::mem::MaybeUninit<ScoredMove>; MAX_MOVES],
    pub len: usize,
}

impl ScoredMoveList {
    #[inline(always)]
    pub fn new() -> Self {
        ScoredMoveList {
            entries: [core::mem::MaybeUninit::uninit(); MAX_MOVES],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, mov: Move, score: i32) {
        self.entries[self.len] = core::mem::MaybeUninit::new(ScoredMove { mov, score });
        self.len += 1;
    }

    /// Iterator yielding `(Move, score)` pairs for the scored entries.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = (Move, i32)> + '_ {
        // SAFETY: same reasoning as MoveList::as_slice.
        let s: &[ScoredMove] =
            unsafe { core::slice::from_raw_parts(self.entries.as_ptr() as *const ScoredMove, self.len) };
        s.iter().map(|e| (e.mov, e.score))
    }

    /// Read entry `i` (must be `< len`). Lets the recursion read a move/score
    /// without holding a borrow of the list across a recursive search call,
    /// which would conflict with the arena being borrowed mutably.
    #[inline(always)]
    pub fn get(&self, i: usize) -> (Move, i32) {
        debug_assert!(i < self.len);
        // SAFETY: `i < len` implies the entry was initialized by `push`.
        let e = unsafe { &*(self.entries.as_ptr().add(i) as *const ScoredMove) };
        (e.mov, e.score)
    }

    /// First (highest-scoring) move, or a null move if the list is empty.
    #[inline(always)]
    pub fn first_mov(&self) -> Move {
        if self.len == 0 {
            return Move::normal(0, 0, 0, NO_PIECE);
        }
        // SAFETY: entries[0] is written when len > 0.
        unsafe { (*(self.entries.as_ptr() as *const ScoredMove)).mov }
    }

    /// Sort descending by score (unstable; fast for small N).
    pub fn sort_desc(&mut self) {
        // SAFETY: only the first `len` entries have been initialized.
        let s: &mut [ScoredMove] = unsafe {
            core::slice::from_raw_parts_mut(self.entries.as_mut_ptr() as *mut ScoredMove, self.len)
        };
        s.sort_unstable_by(|a, b| b.score.cmp(&a.score));
    }
}

/// Fast direct-check detection without make/unmake.  Handles direct attacks
/// from the piece arriving on `m.to`.  Rare discovered checks from EP are
/// conservatively treated as non-checks — safe for pruning guards at low depth.
#[inline(always)]
pub fn gives_check_fast(b: &Board, m: Move) -> bool {
    if m.flags == M_EP {
        // Removing two pawns can open a discovered check.  Return false —
        // the pruning margins protect against rare tactical misses.
        return false;
    }
    let them = b.side ^ 1;
    let king_sq = b.kingsq[them];
    let occ_no_from = b.all & !bit(m.from);
    let to_bit = bit(m.to);

    let piece = if m.promo != NO_PIECE { m.promo } else { m.piece };
    match piece {
        KNIGHT => KNIGHT_ATTACKS[m.to] & to_bit & bit(king_sq) != 0,
        BISHOP => bishop_attacks(occ_no_from, m.to) & bit(king_sq) != 0,
        ROOK => rook_attacks(occ_no_from, m.to) & bit(king_sq) != 0,
        QUEEN => queen_attacks(occ_no_from, m.to) & bit(king_sq) != 0,
        KING => false,
        _ => PAWN_ATTACKS[b.side][m.to] & bit(king_sq) != 0,
    }
}

/// Generate all pseudo-legal moves, **replacing** any previous contents of
/// `list` (callers may reuse a list; nothing is appended to).
pub fn generate_pseudo(b: &Board, list: &mut MoveList) {
    list.len = 0;
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
            list.push(Move::normal(sq, to, KNIGHT, b.piece_on(to)));
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
            list.push(Move::normal(sq, to, BISHOP, b.piece_on(to)));
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
            list.push(Move::normal(sq, to, ROOK, b.piece_on(to)));
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
            list.push(Move::normal(sq, to, QUEEN, b.piece_on(to)));
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
            list.push(Move::normal(sq, to, KING, b.piece_on(to)));
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
                        list.push(Move::promo(sq, one, p));
                    }
                } else {
                    list.push(Move::normal(sq, one, PAWN, NO_PIECE));
                    if r == 1 && all & bit(sq + 16) == 0 {
                        list.push(Move::normal(sq, sq + 16, PAWN, NO_PIECE));
                    }
                }
            }
        } else {
            let one = sq - 8;
            if all & bit(one) == 0 {
                if r == 1 {
                    for p in [KNIGHT, BISHOP, ROOK, QUEEN] {
                        list.push(Move::promo(sq, one, p));
                    }
                } else {
                    list.push(Move::normal(sq, one, PAWN, NO_PIECE));
                    if r == 6 && all & bit(sq - 16) == 0 {
                        list.push(Move::normal(sq, sq - 16, PAWN, NO_PIECE));
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
                    list.push(Move {
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
                    list.push(Move {
                        from: sq,
                        to,
                        piece: PAWN,
                        captured: PAWN,
                        promo: p,
                        flags: M_PROMO,
                    });
                }
            } else {
                list.push(Move::normal(sq, to, PAWN, b.piece_on(to)));
            }
        }
    }

    if let Some(e) = b.ep {
        let attackers = PAWN_ATTACKS[us ^ 1][e] & b.pieces[us][PAWN];
        let mut a = attackers;
        while a != 0 {
            let sq = a.trailing_zeros() as usize;
            a &= a - 1;
            list.push(Move::ep(sq, e));
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
                list.push(Move::castle(4, 6));
            }
            if b.castle & WQ != 0
                && all & bit(1) == 0
                && all & bit(2) == 0
                && all & bit(3) == 0
                && b.pieces[us][ROOK] & bit(0) != 0
                && !b.attacked(them, 3)
            {
                list.push(Move::castle(4, 2));
            }
        } else {
            if b.castle & BK != 0
                && all & bit(61) == 0
                && all & bit(62) == 0
                && b.pieces[us][ROOK] & bit(63) != 0
                && !b.attacked(them, 61)
            {
                list.push(Move::castle(60, 62));
            }
            if b.castle & BQ != 0
                && all & bit(57) == 0
                && all & bit(58) == 0
                && all & bit(59) == 0
                && b.pieces[us][ROOK] & bit(56) != 0
                && !b.attacked(them, 59)
            {
                list.push(Move::castle(60, 58));
            }
        }
    }
}

/// Generate only legal moves.  Uses swap-remove on the stack list — zero allocation.
pub fn generate_legal(b: &mut Board, list: &mut MoveList) {
    generate_pseudo(b, list);
    // Raw slice over the MaybeUninit backing so we can swap-remove in place.
    let s: &mut [Move] =
        unsafe { core::slice::from_raw_parts_mut(list.moves.as_mut_ptr() as *mut Move, MAX_MOVES) };
    let mut i = 0;
    while i < list.len {
        let m = s[i];
        if !b.make(m) {
            // `make` pushes an undo record even for illegal moves; restore it
            // so the board is reusable for the remaining candidates.
            b.unmake(m);
            // swap-remove illegal move
            list.len -= 1;
            s[i] = s[list.len];
        } else {
            b.unmake(m);
            i += 1;
        }
    }
}

/// Generate only captures and promotions.  Generates all pseudo, then retains.
pub fn generate_captures(b: &Board, list: &mut MoveList) {
    generate_pseudo(b, list);
    list.retain(|m| m.captured != NO_PIECE || m.promo != NO_PIECE);
}

/// All pieces of `color` that attack `sq` given the occupancy `occ`.
///
/// Sliding attackers come from the magic bitboard tables: two branchless
/// probes replace the eight per-ray first-blocker scans.
fn attackers_mask(b: &Board, occ: u64, sq: usize, color: usize) -> u64 {
    let mut m = PAWN_ATTACKS[color ^ 1][sq] & b.pieces[color][PAWN];
    m |= KNIGHT_ATTACKS[sq] & b.pieces[color][KNIGHT];
    m |= KING_ATTACKS[sq] & b.pieces[color][KING];
    // Intersecting with `occ` keeps the probe bit-exact with a first-blocker
    // ray scan: a slider only counts when it really stands on the ray.
    let rooks = (b.pieces[color][ROOK] | b.pieces[color][QUEEN]) & occ;
    if rooks != 0 {
        m |= rook_attacks(occ, sq) & rooks;
    }
    let bishops = (b.pieces[color][BISHOP] | b.pieces[color][QUEEN]) & occ;
    if bishops != 0 {
        m |= bishop_attacks(occ, sq) & bishops;
    }
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
        let mut list = MoveList::new();
        generate_legal(&mut b, &mut list);
        assert_eq!(list.len(), 20);
        assert_eq!(crate::perft::perft(&mut b, 1), 20);
        assert_eq!(crate::perft::perft(&mut b, 2), 400);
    }

    #[test]
    fn queen_recapture_generated() {
        let mut b = Board::from_fen(
            "rnb1kbnr/pppp1ppp/4p3/8/3P1q2/6P1/PPP2P1P/RNBQKBNR w KQkq - 0 1",
        )
        .unwrap();
        let mut list = MoveList::new();
        generate_legal(&mut b, &mut list);
        assert!(
            list.iter().any(|m| m.from == 22 && m.to == 29 && m.piece == PAWN),
            "g3xf4 missing from legal moves"
        );
    }

    /// Ray-walking reference for `attackers_mask`, kept as the oracle for the
    /// magic-bitboard implementation.
    fn attackers_mask_by_rays(b: &Board, occ: u64, sq: usize, color: usize) -> u64 {
        let mut m = PAWN_ATTACKS[color ^ 1][sq] & b.pieces[color][PAWN];
        m |= KNIGHT_ATTACKS[sq] & b.pieces[color][KNIGHT];
        m |= KING_ATTACKS[sq] & b.pieces[color][KING];
        let rooks = b.pieces[color][ROOK] | b.pieces[color][QUEEN];
        let bishops = b.pieces[color][BISHOP] | b.pieces[color][QUEEN];
        for d in 0..8usize {
            let blockers = RAY[sq][d] & occ;
            if blockers == 0 {
                continue;
            }
            let first = if DIRD[d] > 0 {
                blockers & blockers.wrapping_neg()
            } else {
                1u64 << (63 - blockers.leading_zeros())
            };
            let sliders = if d % 2 == 0 { rooks } else { bishops };
            m |= sliders & first;
        }
        m
    }

    /// SEE reference driven by the ray-walking attacker scan.
    fn see_by_rays(b: &Board, sq: usize, stm: usize) -> i32 {
        fn rec(b: &Board, sq: usize, side: usize, occ: u64, on_sq: i32, can_pass: bool) -> i32 {
            let atk = attackers_mask_by_rays(b, occ, sq, side);
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
            let resp = rec(b, sq, side ^ 1, occ2, att_val, true);
            let gain = on_sq - resp;
            if can_pass {
                gain.max(0)
            } else {
                gain
            }
        }
        if b.piece_on(sq) == NO_PIECE {
            return 0;
        }
        let occ = b.all & !bit(sq);
        rec(b, sq, stm, occ, PIECE_VALUES[b.piece_on(sq)], false)
    }

    #[test]
    fn see_matches_ray_reference() {
        let mut b = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..60 {
            for sq in 0..64 {
                for stm in 0..2 {
                    assert_eq!(
                        see_by_rays(&b, sq, stm),
                        see(&b, sq, stm),
                        "SEE disagrees at {sq} (colour {stm}) in {}",
                        b.to_fen()
                    );
                }
            }
            let mut moves = MoveList::new();
            generate_legal(&mut b, &mut moves);
            if moves.is_empty() {
                break;
            }
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let m = moves.as_slice()[(rng >> 33) as usize % moves.len];
            let _ = b.make(m);
        }
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
