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

pub const MATE: i32 = 32000;

pub const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0];

const PST_PAWN: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
];

const PST_KNIGHT: [i32; 64] = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
];

const PST_BISHOP: [i32; 64] = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
];

const PST_ROOK: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
];

const PST_QUEEN: [i32; 64] = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
];

const PST_KING: [i32; 64] = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
];

const PST_KING_EG: [i32; 64] = [
    -30, -20, -10, -10, -10, -10, -20, -30,
    -20, -10, 0, 0, 0, 0, -10, -20,
    -10, 0, 10, 15, 15, 10, 0, -10,
    -10, 5, 15, 20, 20, 15, 5, -10,
    -10, 0, 15, 20, 20, 15, 0, -10,
    -10, 5, 10, 15, 15, 10, 5, -10,
    -20, -10, 0, 5, 5, 0, -10, -20,
    -30, -20, -10, -10, -10, -10, -20, -30,
];

const PST: [[i32; 64]; 6] = [
    PST_PAWN, PST_KNIGHT, PST_BISHOP, PST_ROOK, PST_QUEEN, PST_KING,
];

const PST_PAWN_EG: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
];

const PST_EG: [[i32; 64]; 6] = [
    PST_PAWN_EG, PST_KNIGHT, PST_BISHOP, PST_ROOK, PST_QUEEN, PST_KING_EG,
];

const EG_PASSED: i32 = 28;
const KING_PROX: i32 = 10;
const PIN_PEN: [i32; 6] = [10, 35, 30, 45, 60, 0];

const MOB_WEIGHT: [i32; 6] = [0, 4, 4, 3, 2, 0];
const BISHOP_PAIR: i32 = 32;
const DOUBLED_PAWN: i32 = 10;
const ISOLATED_PAWN: i32 = 14;
const BACKWARD_PAWN: i32 = 12;
const ROOK_BATTERY: i32 = 9;
const QUEEN_ROOK_BATTERY: i32 = 12;
const BATTERY_HALF_OPEN: i32 = 7;
const BATTERY_AT_KING: i32 = 10;
const BISHOP_QUEEN_BATTERY: i32 = 7;
const KING_SHIELD: i32 = 12;
const KING_EXPOSED: i32 = 45;
const ROOK_SEMI_OPEN: i32 = 10;
const ROOK_OPEN: i32 = 20;
const DEVELOPED_MINOR: i32 = 10;
const KING_ATK_COV: i32 = 3;
const KING_ATK_TWO: i32 = 18;
const KING_ATK_THREE: i32 = 30;
const KS_W: [i32; 6] = [0, 35, 35, 55, 70, 0];

fn ahead_mask(color: usize, rank: usize, file: usize) -> u64 {
    let fm = FILE_MASK[file];
    let adj = fm | FILE_MASK[file.saturating_sub(1)] | FILE_MASK[(file + 1).min(7)];
    if color == WHITE {
        let hi = !(bit(8 * (rank + 1)) - 1);
        adj & hi
    } else {
        let lo = bit(8 * rank) - 1;
        adj & lo
    }
}

fn king_dist(a: usize, b: usize) -> i32 {
    (file_of(a) as i32 - file_of(b) as i32)
        .abs()
        .max((rank_of(a) as i32 - rank_of(b) as i32).abs())
}

fn pinned_mask(b: &Board, side: usize) -> u64 {
    let ks = b.kingsq[side];
    let enemy = side ^ 1;
    let own = b.occ[side];
    let occ = b.all;
    let dia = b.pieces[enemy][BISHOP] | b.pieces[enemy][QUEEN];
    let orth = b.pieces[enemy][ROOK] | b.pieces[enemy][QUEEN];

    let dirs: [(i32, i32); 8] = [
        (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1),
    ];
    let mut pinned = 0u64;
    for (dr, df) in dirs {
        let diag = dr != 0 && df != 0;
        let sliders = if diag { dia } else { orth };
        let mut kr = rank_of(ks) as i32 + dr;
        let mut kf = file_of(ks) as i32 + df;
        let mut blocker: Option<usize> = None;
        while (0..8).contains(&kr) && (0..8).contains(&kf) {
            let sq = kr as usize * 8 + kf as usize;
            if own & bit(sq) != 0 {
                if blocker.is_some() {
                    break;
                }
                blocker = Some(sq);
            } else if occ & bit(sq) != 0 {
                if sliders & bit(sq) != 0 {
                    if let Some(bs) = blocker {
                        pinned |= bit(bs);
                    }
                }
                break;
            }
            kr += dr;
            kf += df;
        }
    }
    pinned
}

pub fn evaluate(b: &Board) -> i32 {
    let mut score = 0i32;

    let mut mat_phase = 0i32;
    for c in 0..2 {
        mat_phase += 4 * b.pieces[c][QUEEN].count_ones() as i32;
        mat_phase += 2 * b.pieces[c][ROOK].count_ones() as i32;
        mat_phase += (b.pieces[c][KNIGHT].count_ones() + b.pieces[c][BISHOP].count_ones()) as i32;
    }
    let p = (24 - mat_phase).clamp(0, 24);
    let mut attacked_by = [0u64; 2];
    let mut king_cover = [0u64; 2];
    let mut atk_force = [0i32; 2];
    let mut atk_pieces = [0i32; 2];

    for c in 0..2 {
        let mut side = 0i32;
        let own = b.occ[c];
        let ering = KING_ATTACKS[b.kingsq[c ^ 1]] | bit(b.kingsq[c ^ 1]);
        let mut attackers = 0;
        let pinned = pinned_mask(b, c);
        let enemy_mm = b.pieces[c ^ 1][KNIGHT]
            | b.pieces[c ^ 1][BISHOP]
            | b.pieces[c ^ 1][ROOK]
            | b.pieces[c ^ 1][QUEEN]
            | b.pieces[c ^ 1][KING];

        for pc in 0..5 {
            let mut bb = b.pieces[c][pc];
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                let mapped = if c == BLACK { sq } else { sq ^ 56 };
                let ps = (PST[pc][mapped] * (24 - p) + PST_EG[pc][mapped] * p) / 24;
                side += PIECE_VALUES[pc] + ps;
                if pc == KNIGHT || pc == BISHOP || pc == ROOK || pc == QUEEN {
                    let attacks = match pc {
                        KNIGHT => KNIGHT_ATTACKS[sq],
                        BISHOP => bishop_attacks(b.all, sq),
                        ROOK => rook_attacks(b.all, sq),
                        _ => queen_attacks(b.all, sq),
                    };
                    attacked_by[c] |= attacks;
                    let pin = pinned & bit(sq) != 0;
                    if pin {
                        side -= PIN_PEN[pc];
                    }
                    let mob = (attacks & !own).count_ones() as i32;
                    side += MOB_WEIGHT[pc] * if pin { mob / 2 } else { mob };
                    if pin {
                        continue;
                    }
                    let on_ring = attacks & ering;
                    side += KING_ATK_COV * on_ring.count_ones() as i32;
                    if on_ring != 0 {
                        attackers += 1;
                        king_cover[c ^ 1] |= on_ring;
                        atk_force[c ^ 1] += KS_W[pc];
                        atk_pieces[c ^ 1] += 1;
                    }
                    let fork_hits = attacks & enemy_mm;
                    if fork_hits.count_ones() >= 2 {
                        let mut a = 0i32;
                        let mut bv = 0i32;
                        let mut h = fork_hits;
                        while h != 0 {
                            let s2 = h.trailing_zeros() as usize;
                            h &= h - 1;
                            let v = PIECE_VALUES[b.piece_on(s2)];
                            if v > a {
                                bv = a;
                                a = v;
                            } else if v > bv {
                                bv = v;
                            }
                        }
                        if bv > 0 {
                            let bonus = if pc == KNIGHT { 8 + bv / 4 } else { 5 + bv / 6 };
                            let bonus = bonus.min(a - bv + 25).min(150);
                            side += bonus;
                        }
                    }
                }
            }
        }
        attacked_by[c] |= KING_ATTACKS[b.kingsq[c]];

        if attackers >= 3 {
            side += KING_ATK_THREE;
        } else if attackers >= 2 {
            side += KING_ATK_TWO;
        }

        if b.pieces[c][BISHOP].count_ones() >= 2 {
            side += BISHOP_PAIR;
        }

        let mut nb = b.pieces[c][KNIGHT];
        while nb != 0 {
            let sq = nb.trailing_zeros() as usize;
            nb &= nb - 1;
            let r = rank_of(sq) as i32;
            let outpost_rank = if c == WHITE { (3..=5).contains(&r) } else { (2..=4).contains(&r) };
            if outpost_rank {
                let defended = PAWN_ATTACKS[c ^ 1][sq] & b.pieces[c][PAWN] != 0;
                let chased = PAWN_ATTACKS[c][sq] & b.pieces[c ^ 1][PAWN] != 0;
                if defended && !chased {
                    side += if (c == WHITE && r == 5) || (c == BLACK && r == 4) {
                        24
                    } else {
                        14
                    };
                }
            }
        }

        let home_minors = if c == WHITE {
            bit(1) | bit(6) | bit(2) | bit(5)
        } else {
            bit(57) | bit(62) | bit(58) | bit(61)
        };
        side += DEVELOPED_MINOR * ((b.pieces[c][KNIGHT] | b.pieces[c][BISHOP]) & !home_minors).count_ones() as i32;

        let pawns = b.pieces[c][PAWN];
        let mut bb = pawns;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            attacked_by[c] |= PAWN_ATTACKS[c][sq];
            let f = file_of(sq);
            let r = rank_of(sq);

            if (pawns & FILE_MASK[f]).count_ones() > 1 {
                side -= DOUBLED_PAWN;
            }

            let adj_files = FILE_MASK[f.saturating_sub(1)] | FILE_MASK[(f + 1).min(7)];
            if pawns & adj_files == 0 {
                side -= ISOLATED_PAWN;
            }

            let protected_by = pawns & PAWN_ATTACKS[c ^ 1][sq] != 0;
            if pawns & PAWN_ATTACKS[c][sq] != 0 {
                side += 7;
            }
            if protected_by {
                side += 6;
            }
            let front_sq = if c == WHITE {
                if r == 7 { 64 } else { sq + 8 }
            } else if r == 0 {
                64
            } else {
                sq - 8
            };
            if !protected_by
                && front_sq < 64
                && b.pieces[c ^ 1][PAWN] & PAWN_ATTACKS[c][front_sq] != 0
            {
                let behind = if c == WHITE { bit(8 * r) - 1 } else { !(bit(8 * (r + 1)) - 1) };
                if pawns & adj_files & behind == 0 {
                    side -= BACKWARD_PAWN;
                }
            }

            if b.pieces[c ^ 1][PAWN] & ahead_mask(c, r, f) == 0 {
                let progress = if c == WHITE { r as i32 } else { 7 - r as i32 };
                side += 14 + 12 * progress;
                side += (EG_PASSED + 8 * progress) * p / 24;
                let adj = FILE_MASK[f.saturating_sub(1)] | FILE_MASK[(f + 1).min(7)];
                if pawns & adj != 0 {
                    side += (15 + 12 * progress.min(3)) * p / 24;
                }
                if b.pieces[c][ROOK] & FILE_MASK[f] != 0 {
                    let mut behind = false;
                    let mut rr = b.pieces[c][ROOK] & FILE_MASK[f];
                    while rr != 0 {
                        let rsq = rr.trailing_zeros() as usize;
                        rr &= rr - 1;
                        let rk = rank_of(rsq) as i32;
                        if (c == WHITE && rk < r as i32) || (c == BLACK && rk > r as i32) {
                            behind = true;
                        }
                    }
                    if behind {
                        side += (25 + 8 * progress) * p / 24;
                    }
                }
            }
        }

        let mut rooks = b.pieces[c][ROOK];
        while rooks != 0 {
            let sq = rooks.trailing_zeros() as usize;
            rooks &= rooks - 1;
            let f = file_of(sq);
            let own_p = pawns & FILE_MASK[f];
            let opp_p = b.pieces[c ^ 1][PAWN] & FILE_MASK[f];
            if own_p == 0 {
                side += if opp_p == 0 { ROOK_OPEN } else { ROOK_SEMI_OPEN };
            }
            let on_7th = if c == WHITE {
                rank_of(sq) == 6
            } else {
                rank_of(sq) == 1
            };
            if on_7th {
                let enemy_zone = if c == WHITE {
                    RANK_MASK[6] | RANK_MASK[7]
                } else {
                    RANK_MASK[0] | RANK_MASK[1]
                };
                let checks = (b.pieces[c ^ 1][PAWN] | bit(b.kingsq[c ^ 1])) & enemy_zone;
                side += if checks != 0 { 30 } else { 18 };
            }
        }

        let ks = b.kingsq[c];
        let (kf, kr) = (file_of(ks), rank_of(ks));
        let sr = if c == WHITE { kr + 1 } else { kr.saturating_sub(1) };
        if sr < 8 {
            for df in -1i32..=1i32 {
                let sf = kf as i32 + df;
                if (0..8).contains(&sf) && pawns & bit(sr * 8 + sf as usize) != 0 {
                    side += KING_SHIELD;
                }
            }
        }
        let queens = b.pieces[0][QUEEN] | b.pieces[1][QUEEN];
        let rooks_all = b.pieces[0][ROOK] | b.pieces[1][ROOK];
        let midgame = queens.count_ones() > 0 || rooks_all.count_ones() >= 3;
        let home_rank = if c == WHITE { 0 } else { 7 };
        if midgame && kr as i32 != home_rank {
            side -= KING_EXPOSED;
            // Additional penalty for king exposure in open positions
            let open_files = (0..8).filter(|&f| {
                let file_mask = FILE_MASK[f];
                let pawns_in_file = (b.pieces[0][PAWN] | b.pieces[1][PAWN]) & file_mask;
                pawns_in_file == 0
            }).count() as i32;
            if open_files >= 3 {
                side -= KING_EXPOSED / 2;
            }
        }

        let enemy_half = if c == WHITE {
            RANK_MASK[4] | RANK_MASK[5] | RANK_MASK[6]
        } else {
            RANK_MASK[1] | RANK_MASK[2] | RANK_MASK[3]
        };
        let space_cnt = (attacked_by[c] & enemy_half & !b.pieces[c ^ 1][PAWN]).count_ones() as i32;
        side += (space_cnt * 2).min(30);

        let enemy_king_file = file_of(b.kingsq[c ^ 1]) as i32;
        {
            let mut pcs = b.pieces[c][ROOK] | b.pieces[c][QUEEN];
            let mut file_pieces = [0u64; 8];
            let mut file_cnt = [0u32; 8];
            while pcs != 0 {
                let sq = pcs.trailing_zeros() as usize;
                pcs &= pcs - 1;
                let f = file_of(sq);
                file_cnt[f] += 1;
                file_pieces[f] |= bit(sq);
            }
            for f in 0..8 {
                if file_cnt[f] != 2 {
                    continue;
                }
                let m = file_pieces[f];
                let sq1 = m.trailing_zeros() as usize;
                let sq2 = (m ^ bit(sq1)).trailing_zeros() as usize;
                let (r1, r2) = (
                    rank_of(sq1).min(rank_of(sq2)),
                    rank_of(sq1).max(rank_of(sq2)),
                );
                let mut clear = true;
                for r in (r1 + 1)..r2 {
                    if b.all & bit(r * 8 + f) != 0 {
                        clear = false;
                        break;
                    }
                }
                if clear {
                    let mut bonus = if m & b.pieces[c][QUEEN] != 0 {
                        QUEEN_ROOK_BATTERY
                    } else {
                        ROOK_BATTERY
                    };
                    if b.pieces[c ^ 1][PAWN] & FILE_MASK[f] == 0 {
                        bonus += BATTERY_HALF_OPEN;
                    }
                    if (enemy_king_file - f as i32).abs() <= 1 {
                        bonus += BATTERY_AT_KING;
                    }
                    side += bonus;
                }
            }
        }
        {
            let mut pcs = b.pieces[c][BISHOP] | b.pieces[c][QUEEN];
            let mut diag1 = [0u64; 15];
            let mut diag2 = [0u64; 15];
            while pcs != 0 {
                let sq = pcs.trailing_zeros() as usize;
                pcs &= pcs - 1;
                let f = file_of(sq);
                let r = rank_of(sq);
                diag1[f + r] |= bit(sq);
                diag2[f + 7 - r] |= bit(sq);
            }
            for di in 0..15 {
                for acc in [&diag1[..], &diag2[..]] {
                    let m = acc[di];
                    if m.count_ones() != 2 || m & b.pieces[c][BISHOP] == 0 || m & b.pieces[c][QUEEN] == 0 {
                        continue;
                    }
                    let sq1 = m.trailing_zeros() as usize;
                    let sq2 = (m ^ bit(sq1)).trailing_zeros() as usize;
                    let (lo, hi) = if sq1 < sq2 { (sq1, sq2) } else { (sq2, sq1) };
                    let step = if (hi - lo) % 9 == 0 { 9 } else { 7 };
                    let mut clear = true;
                    let mut cur = lo + step;
                    while cur < hi {
                        if b.all & bit(cur) != 0 {
                            clear = false;
                            break;
                        }
                        cur += step;
                    }
                    if clear {
                        let ek = b.kingsq[c ^ 1];
                        let same_diag =
                            (file_of(ek) + rank_of(ek) == di) || (file_of(ek) + 7 - rank_of(ek) == di);
                        let mut bonus = BISHOP_QUEEN_BATTERY;
                        if same_diag {
                            bonus += BATTERY_AT_KING;
                        }
                        side += bonus;
                    }
                }
            }
        }

        let mapped = if c == BLACK { ks } else { ks ^ 56 };
        let king_pst = (PST_KING[mapped] * (24 - p) + PST_KING_EG[mapped] * p) / 24;
        let contrib = if c == WHITE { side + king_pst } else { -(side + king_pst) };
        score += contrib;
    }

    for c in 0..2 {
        let their = attacked_by[c ^ 1];
        let mut pen = 0i32;
        for pc in PAWN..=QUEEN {
            let mut bb = b.pieces[c][pc];
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                if their & bit(sq) == 0 {
                    continue;
                }
                let defended = attacked_by[c] & bit(sq) != 0;
                pen += if defended {
                    PIECE_VALUES[pc] / 10  // Reduced penalty for defended pieces
                } else {
                    PIECE_VALUES[pc] / 2 + PIECE_VALUES[pc] / 4  // Increased penalty for undefended
                };
            }
        }
        score += if c == WHITE { -pen } else { pen };
    }

    for c in 0..2 {
        let cover = king_cover[c].count_ones() as i32;
        let force = atk_force[c];
        if force < 30 || cover < 2 || atk_pieces[c] < 2 {
            continue;
        }
        let (kf, kr) = (file_of(b.kingsq[c]), rank_of(b.kingsq[c]));
        let ahead = if c == WHITE { kr + 1 } else { kr.saturating_sub(1) };
        let mut shield = 0;
        if ahead < 8 {
            for df in -1i32..=1i32 {
                let sf = kf as i32 + df;
                if (0..8).contains(&sf) && b.pieces[c][PAWN] & bit(ahead * 8 + sf as usize) != 0 {
                    shield += 1;
                }
            }
        }
        let shield_f = if shield == 0 { 36 } else { 36 - 8 * shield };
        let mut pen = (force * cover * shield_f / 12) * p / 24;
        if shield == 0 {
            pen = pen * 3 / 2;
        }
        let pen = pen.min(850);
        score += if c == WHITE { -pen } else { pen };
    }

    let w_atk = atk_force[0] * king_cover[0].count_ones() as i32;
    let b_atk = atk_force[1] * king_cover[1].count_ones() as i32;
    let init = (w_atk - b_atk).clamp(-900, 900) / 110;
    if init.abs() >= 2 {
        let sign = if init > 0 { 1 } else { -1 };
        score += sign * init.abs().min(9) * p / 24;
    }

    let mut prog = [0i32; 2];
    for c in 0..2 {
        let mut bb = b.pieces[c ^ 1][PAWN];
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            let f = file_of(sq);
            let r = rank_of(sq);
            if b.pieces[c][PAWN] & ahead_mask(c ^ 1, r, f) != 0 {
                continue;
            }
            let sign = if c == WHITE { 1 } else { -1 };
            let def = (4 - king_dist(b.kingsq[c], sq)).max(0) * 18;
            score += sign * def * p / 24;
        }
    }
    for c in 0..2 {
        let mut bb = b.pieces[c][PAWN];
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            let f = file_of(sq);
            let r = rank_of(sq);
            if b.pieces[c ^ 1][PAWN] & ahead_mask(c, r, f) != 0 {
                continue;
            }
            let progress = if c == WHITE { r as i32 } else { 7 - r as i32 };
            prog[c] = prog[c].max(progress);
            let sign = if c == WHITE { 1 } else { -1 };
            let support = (4 - king_dist(b.kingsq[c], sq)).max(0) * 26;
            let stop = (4 - king_dist(b.kingsq[c ^ 1], sq)).max(0) * 18;
            score += sign * (support - stop) * p / 24;
            if progress >= 5 {
                score += sign * 32 * p / 24;
            }
        }
    }
    score += (prog[0] - prog[1]) * 22 * p / 24;

    let cd = ((file_of(b.kingsq[0]) as i32 - file_of(b.kingsq[1]) as i32).abs())
            .max((rank_of(b.kingsq[0]) as i32 - rank_of(b.kingsq[1]) as i32).abs());
    score += (7 - cd) * KING_PROX * p / 24;

    // Additional penalty for pawn pushes that expose the king
    for c in 0..2 {
        let king_sq = b.kingsq[c];
        let king_file = file_of(king_sq);
        let king_rank = rank_of(king_sq);
        
        // Check if there are pawns in front of the king that could be pushed
        let pawn_direction = if c == WHITE { 1 } else { -1 };
        let pawn_rank = if c == WHITE { king_rank - 1 } else { king_rank + 1 };
        
        if pawn_rank < 8 && pawn_rank < 8 {
            let pawn_mask = RANK_MASK[pawn_rank] & FILE_MASK[king_file];
            let pawns = b.pieces[c][PAWN] & pawn_mask;
            
            if pawns != 0 {
                // Penalty for having pawns in front of the king that could be pushed
                score += if c == WHITE { -15 } else { 15 };
            }
        }
    }

    if b.pieces[0][QUEEN] == 0 && b.pieces[1][QUEEN] == 0 {
        let color_of_sq = |sq: usize| (file_of(sq) + rank_of(sq)) & 1;
        let wob = b.pieces[0][BISHOP].count_ones();
        let bob = b.pieces[1][BISHOP].count_ones();
        if wob > 0 && bob > 0 {
            let w_sq_color = color_of_sq(b.pieces[0][BISHOP].trailing_zeros() as usize);
            let b_sq_color = color_of_sq(b.pieces[1][BISHOP].trailing_zeros() as usize);
            let mut w_same = true;
            let mut bb = b.pieces[0][BISHOP];
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                if color_of_sq(sq) != w_sq_color {
                    w_same = false;
                }
            }
            let mut b_same = true;
            let mut bb = b.pieces[1][BISHOP];
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                if color_of_sq(sq) != b_sq_color {
                    b_same = false;
                }
            }
            if w_same && b_same && w_sq_color != b_sq_color {
                score = score * 3 / 4;
            }
        }
    }

    if b.side == WHITE {
        score + 12
    } else {
        -score + 12
    }
}