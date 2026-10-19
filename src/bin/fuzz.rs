use chess_engine::board::*;
use chess_engine::movegen::{generate_legal, MoveList};

fn main() {
    let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
    let mut rng = |lo: usize| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 32) as usize) % lo
    };

    for game in 0..400 {
        let mut b = Board::startpos();
        let mut plies = 0;
        loop {
            let mut a = MoveList::new();
            generate_legal(&mut b, &mut a);
            let r = reference::ref_legal_pub(&mut b);

            let sa = a.iter().map(|m| m.uci()).collect::<Vec<String>>();
            let sr = r.iter().map(|m| m.uci()).collect::<Vec<String>>();
            let same_set = sa.len() == sr.len() && sa.iter().all(|m| sr.contains(m));

            if !same_set {
                println!(
                    "MISMATCH game={} plies={} fen={}",
                    game, plies, b.to_fen()
                );
                for m in a.iter() {
                    if !sr.contains(&m.uci()) {
                        println!("  only mygen: {}", m.uci());
                    }
                }
                for m in r.iter() {
                    if !sa.contains(&m.uci()) {
                        println!("  only ref:   {}", m.uci());
                    }
                }
                return;
            }

            if a.is_empty() {
                break;
            }
            let m = a.as_slice()[rng(a.len)];
            let _ = b.make(m);
            plies += 1;
            if plies >= 60 {
                break;
            }
        }
    }
    println!("no mismatches found");
}

mod reference {
    use super::*;
    use chess_engine::bitboard::*;

    pub fn ref_legal_pub(b: &mut Board) -> Vec<Move> {
        let pseudo = ref_pseudo(b);
        let mut legal = Vec::new();
        for m in pseudo {
            if b.make(m) {
                legal.push(m);
            }
            b.unmake(m);
        }
        legal
    }

    fn ref_pseudo(b: &Board) -> Vec<Move> {
        let mut moves = Vec::new();
        let us = b.side;
        for sq in 0..64 {
            let pt = b.piece_on(sq);
            if pt == NO_PIECE || b.color_on(sq).unwrap() != us {
                continue;
            }
            let r = rank_of(sq);
            let f = file_of(sq);
            if pt == PAWN {
                if us == WHITE {
                    if r < 7 && b.all & bit(sq + 8) == 0 {
                        let to = sq + 8;
                        if r == 6 {
                            for p in [KNIGHT, BISHOP, ROOK, QUEEN] {
                                moves.push(Move { from: sq, to, piece: PAWN, captured: NO_PIECE, promo: p, flags: M_PROMO });
                            }
                        } else {
                            moves.push(Move::normal(sq, to, PAWN, NO_PIECE));
                            if r == 1 && b.all & bit(sq + 16) == 0 {
                                moves.push(Move::normal(sq, sq + 16, PAWN, NO_PIECE));
                            }
                        }
                    }
                    for df in [-1i32, 1] {
                        let tof = f as i32 + df;
                        let tor = r as i32 + 1;
                        if tor >= 0 && tor < 8 && tof >= 0 && tof < 8 {
                            let to = (tor * 8 + tof) as usize;
                            if b.color_on(to) == Some(us ^ 1) {
                                if tor == 7 {
                                    for p in [KNIGHT, BISHOP, ROOK, QUEEN] {
                                        moves.push(Move { from: sq, to, piece: PAWN, captured: PAWN, promo: p, flags: M_PROMO });
                                    }
                                } else {
                                    moves.push(Move::normal(sq, to, PAWN, b.piece_on(to)));
                                }
                            }
                        }
                    }
                    if let Some(e) = b.ep {
                        let er = rank_of(e);
                        if er == r + 1 && (f as i32 - file_of(e) as i32).abs() == 1 {
                            moves.push(Move::ep(sq, e));
                        }
                    }
                } else {
                    if r > 0 && b.all & bit(sq - 8) == 0 {
                        let to = sq - 8;
                        if r == 1 {
                            for p in [KNIGHT, BISHOP, ROOK, QUEEN] {
                                moves.push(Move { from: sq, to, piece: PAWN, captured: NO_PIECE, promo: p, flags: M_PROMO });
                            }
                        } else {
                            moves.push(Move::normal(sq, to, PAWN, NO_PIECE));
                            if r == 6 && b.all & bit(sq - 16) == 0 {
                                moves.push(Move::normal(sq, sq - 16, PAWN, NO_PIECE));
                            }
                        }
                    }
                    for df in [-1i32, 1] {
                        let tof = f as i32 + df;
                        let tor = r as i32 - 1;
                        if tor >= 0 && tor < 8 && tof >= 0 && tof < 8 {
                            let to = (tor * 8 + tof) as usize;
                            if b.color_on(to) == Some(us ^ 1) {
                                if tor == 0 {
                                    for p in [KNIGHT, BISHOP, ROOK, QUEEN] {
                                        moves.push(Move { from: sq, to, piece: PAWN, captured: PAWN, promo: p, flags: M_PROMO });
                                    }
                                } else {
                                    moves.push(Move::normal(sq, to, PAWN, b.piece_on(to)));
                                }
                            }
                        }
                    }
                    if let Some(e) = b.ep {
                        let er = rank_of(e);
                        if er + 1 == r && (f as i32 - file_of(e) as i32).abs() == 1 {
                            moves.push(Move::ep(sq, e));
                        }
                    }
                }
            } else if pt == KNIGHT {
                let mut t = KNIGHT_ATTACKS[sq];
                while t != 0 {
                    let to = t.trailing_zeros() as usize;
                    t &= t - 1;
                    if b.color_on(to) != Some(us) {
                        moves.push(Move::normal(sq, to, KNIGHT, b.piece_on(to)));
                    }
                }
            } else if pt == BISHOP {
                let mut t = bishop_attacks(b.all, sq);
                while t != 0 {
                    let to = t.trailing_zeros() as usize;
                    t &= t - 1;
                    if b.color_on(to) != Some(us) {
                        moves.push(Move::normal(sq, to, BISHOP, b.piece_on(to)));
                    }
                }
            } else if pt == ROOK {
                let mut t = rook_attacks(b.all, sq);
                while t != 0 {
                    let to = t.trailing_zeros() as usize;
                    t &= t - 1;
                    if b.color_on(to) != Some(us) {
                        moves.push(Move::normal(sq, to, ROOK, b.piece_on(to)));
                    }
                }
            } else if pt == QUEEN {
                let mut t = queen_attacks(b.all, sq);
                while t != 0 {
                    let to = t.trailing_zeros() as usize;
                    t &= t - 1;
                    if b.color_on(to) != Some(us) {
                        moves.push(Move::normal(sq, to, QUEEN, b.piece_on(to)));
                    }
                }
            } else if pt == KING {
                let mut t = KING_ATTACKS[sq];
                while t != 0 {
                    let to = t.trailing_zeros() as usize;
                    t &= t - 1;
                    if b.color_on(to) != Some(us) {
                        moves.push(Move::normal(sq, to, KING, b.piece_on(to)));
                    }
                }
            }

            if pt == KING && sq == b.kingsq[us] && !b.in_check(us) {
                if us == WHITE {
                    if b.castle & WK != 0
                        && b.all & bit(5) == 0
                        && b.all & bit(6) == 0
                        && b.pieces[us][ROOK] & bit(7) != 0
                        && !b.attacked(BLACK, 5)
                    {
                        moves.push(Move::castle(4, 6));
                    }
                    if b.castle & WQ != 0
                        && b.all & bit(1) == 0
                        && b.all & bit(2) == 0
                        && b.all & bit(3) == 0
                        && b.pieces[us][ROOK] & bit(0) != 0
                        && !b.attacked(BLACK, 3)
                    {
                        moves.push(Move::castle(4, 2));
                    }
                } else {
                    if b.castle & BK != 0
                        && b.all & bit(61) == 0
                        && b.all & bit(62) == 0
                        && b.pieces[us][ROOK] & bit(63) != 0
                        && !b.attacked(WHITE, 61)
                    {
                        moves.push(Move::castle(60, 62));
                    }
                    if b.castle & BQ != 0
                        && b.all & bit(57) == 0
                        && b.all & bit(58) == 0
                        && b.all & bit(59) == 0
                        && b.pieces[us][ROOK] & bit(56) != 0
                        && !b.attacked(WHITE, 59)
                    {
                        moves.push(Move::castle(60, 58));
                    }
                }
            }
        }
        moves
    }
}