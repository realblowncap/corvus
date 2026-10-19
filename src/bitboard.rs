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

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;

pub const PAWN: usize = 0;
pub const KNIGHT: usize = 1;
pub const BISHOP: usize = 2;
pub const ROOK: usize = 3;
pub const QUEEN: usize = 4;
pub const KING: usize = 5;

pub const fn bit(sq: usize) -> u64 {
    1u64 << sq
}

pub const fn file_of(sq: usize) -> usize {
    sq & 7
}

pub const fn rank_of(sq: usize) -> usize {
    sq >> 3
}

pub const FILE_A: u64 = 0x0101_0101_0101_0101;
pub const FILE_B: u64 = 0x0202_0202_0202_0202;
pub const FILE_C: u64 = 0x0404_0404_0404_0404;
pub const FILE_D: u64 = 0x0808_0808_0808_0808;
pub const FILE_E: u64 = 0x1010_1010_1010_1010;
pub const FILE_F: u64 = 0x2020_2020_2020_2020;
pub const FILE_G: u64 = 0x4040_4040_4040_4040;
pub const FILE_H: u64 = 0x8080_8080_8080_8080;

pub const NOT_A: u64 = !FILE_A;
pub const NOT_H: u64 = !FILE_H;

pub const FILE_MASK: [u64; 8] = [
    FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H,
];

pub const RANK_1: u64 = 0x0000_0000_0000_00FF;
pub const RANK_2: u64 = 0x0000_0000_0000_FF00;
pub const RANK_3: u64 = 0x0000_0000_00FF_0000;
pub const RANK_4: u64 = 0x0000_0000_FF00_0000;
pub const RANK_5: u64 = 0x0000_00FF_0000_0000;
pub const RANK_6: u64 = 0x0000_FF00_0000_0000;
pub const RANK_7: u64 = 0x00FF_0000_0000_0000;
pub const RANK_8: u64 = 0xFF00_0000_0000_0000;

pub const RANK_MASK: [u64; 8] = [
    RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8,
];

pub const SQUARE_NAMES: [&str; 64] = {
    let mut a = [""; 64];
    let mut i = 0;
    while i < 64 {
        let f = i & 7;
        let r = i >> 3;
        a[i] = match (f, r) {
            (0, 0) => "a1",
            (1, 0) => "b1",
            (2, 0) => "c1",
            (3, 0) => "d1",
            (4, 0) => "e1",
            (5, 0) => "f1",
            (6, 0) => "g1",
            (7, 0) => "h1",
            (0, 1) => "a2",
            (1, 1) => "b2",
            (2, 1) => "c2",
            (3, 1) => "d2",
            (4, 1) => "e2",
            (5, 1) => "f2",
            (6, 1) => "g2",
            (7, 1) => "h2",
            (0, 2) => "a3",
            (1, 2) => "b3",
            (2, 2) => "c3",
            (3, 2) => "d3",
            (4, 2) => "e3",
            (5, 2) => "f3",
            (6, 2) => "g3",
            (7, 2) => "h3",
            (0, 3) => "a4",
            (1, 3) => "b4",
            (2, 3) => "c4",
            (3, 3) => "d4",
            (4, 3) => "e4",
            (5, 3) => "f4",
            (6, 3) => "g4",
            (7, 3) => "h4",
            (0, 4) => "a5",
            (1, 4) => "b5",
            (2, 4) => "c5",
            (3, 4) => "d5",
            (4, 4) => "e5",
            (5, 4) => "f5",
            (6, 4) => "g5",
            (7, 4) => "h5",
            (0, 5) => "a6",
            (1, 5) => "b6",
            (2, 5) => "c6",
            (3, 5) => "d6",
            (4, 5) => "e6",
            (5, 5) => "f6",
            (6, 5) => "g6",
            (7, 5) => "h6",
            (0, 6) => "a7",
            (1, 6) => "b7",
            (2, 6) => "c7",
            (3, 6) => "d7",
            (4, 6) => "e7",
            (5, 6) => "f7",
            (6, 6) => "g7",
            (7, 6) => "h7",
            (0, 7) => "a8",
            (1, 7) => "b8",
            (2, 7) => "c8",
            (3, 7) => "d8",
            (4, 7) => "e8",
            (5, 7) => "f8",
            (6, 7) => "g8",
            (7, 7) => "h8",
            _ => "??",
        };
        i += 1;
    }
    a
};

pub fn square_name(sq: usize) -> String {
    SQUARE_NAMES[sq].to_string()
}

const DIRS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (-1, -1),
];

pub const DIRD: [i32; 8] = [8, 9, 1, 7, -8, -9, -1, -7];

const fn build_rays() -> [[u64; 8]; 64] {
    let mut rays = [[0u64; 8]; 64];
    let mut sq = 0;
    while sq < 64 {
        let rf = (sq >> 3) as i32;
        let ff = (sq & 7) as i32;
        let mut d = 0;
        while d < 8 {
            let (dr, df) = DIRS[d];
            let mut cr = rf + dr;
            let mut cf = ff + df;
            let mut m = 0u64;
            while cr >= 0 && cr < 8 && cf >= 0 && cf < 8 {
                m |= 1u64 << (cr * 8 + cf);
                cr += dr;
                cf += df;
            }
            rays[sq][d] = m;
            d += 1;
        }
        sq += 1;
    }
    rays
}

pub const RAY: [[u64; 8]; 64] = build_rays();

const fn build_knight() -> [u64; 64] {
    let mut t = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let rf = (sq >> 3) as i32;
        let ff = (sq & 7) as i32;
        let offs = [-2, -1, 1, 2];
        let mut i = 0;
        while i < 4 {
            let mut j = 0;
            while j < 4 {
                let dr: i32 = offs[i];
                let df: i32 = offs[j];
                if dr.abs() + df.abs() == 3 {
                    let nr = rf + dr;
                    let nf = ff + df;
                    if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                        t[sq] |= 1u64 << (nr * 8 + nf);
                    }
                }
                j += 1;
            }
            i += 1;
        }
        sq += 1;
    }
    t
}

pub const KNIGHT_ATTACKS: [u64; 64] = build_knight();

const fn build_king() -> [u64; 64] {
    let mut t = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let rf = (sq >> 3) as i32;
        let ff = (sq & 7) as i32;
        let mut dr = -1;
        while dr <= 1 {
            let mut df = -1;
            while df <= 1 {
                if dr != 0 || df != 0 {
                    let nr = rf + dr;
                    let nf = ff + df;
                    if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                        t[sq] |= 1u64 << (nr * 8 + nf);
                    }
                }
                df += 1;
            }
            dr += 1;
        }
        sq += 1;
    }
    t
}

pub const KING_ATTACKS: [u64; 64] = build_king();

const fn build_pawn() -> [[u64; 64]; 2] {
    let mut t = [[0u64; 64]; 2];
    let mut sq = 0;
    while sq < 64 {
        let rf = (sq >> 3) as i32;
        let ff = (sq & 7) as i32;
        let mut c = 0;
        while c < 2 {
            let dr = if c == 0 { 1 } else { -1 };
            let mut df = -1;
            while df <= 1 {
                if df != 0 {
                    let nr = rf + dr;
                    let nf = ff + df;
                    if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                        t[c][sq] |= 1u64 << (nr * 8 + nf);
                    }
                }
                df += 2;
            }
            c += 1;
        }
        sq += 1;
    }
    t
}

pub const PAWN_ATTACKS: [[u64; 64]; 2] = build_pawn();

/// Reference ray-walking slider attacks.  Used to build the magic tables and
/// by tests; not called on the search hot path.
pub fn slider_attack(occ: u64, sq: usize, dirs_list: &[usize]) -> u64 {
    let mut att = 0;
    for &d in dirs_list {
        let ray = RAY[sq][d];
        let blockers = ray & occ;
        if blockers == 0 {
            att |= ray;
        } else if DIRD[d] > 0 {
            let first = blockers & blockers.wrapping_neg();
            att |= ray & (first | (first - 1));
        } else {
            let last = 1u64 << (63 - blockers.leading_zeros());
            att |= ray & !(last - 1);
        }
    }
    att
}

// ---- Sliding attacks: magic bitboards -------------------------------------
//
// Classic "plain" magics: each slider square has a reduced occupancy mask
// (outermost edge squares omitted), a magic constant and a shift; attacks are
// one multiply + shift + table lookup instead of the ray loop above.  The
// magic constants were found offline and validated by the
// `magic_attacks_match_reference` test; the table build below fills the flat
// lookup table from the ray-walking reference.

pub const ROOK_DIRS: [usize; 4] = [0, 2, 4, 6]; // N, E, S, W
pub const BISHOP_DIRS: [usize; 4] = [1, 3, 5, 7]; // NE, SE, SW, NW

/// Reduced mask: all ray squares minus the outermost edge square in each
/// direction — that square's occupancy can never alter the attack set.
const fn slider_mask(sq: usize, dirs: &[usize; 4]) -> u64 {
    let mut out = 0u64;
    let mut i = 0;
    while i < 4 {
        let d = dirs[i];
        let mut r = RAY[sq][d];
        if r != 0 {
            if DIRD[d] > 0 {
                // Furthest (highest-index) square along a positive ray.
                r &= !(1u64 << (63 - r.leading_zeros()));
            } else {
                // Furthest (lowest-index) square along a negative ray.
                r &= !(r & r.wrapping_neg());
            }
            out |= r;
        }
        i += 1;
    }
    out
}

const fn build_masks(dirs: &[usize; 4]) -> [u64; 64] {
    let mut m = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        m[sq] = slider_mask(sq, dirs);
        sq += 1;
    }
    m
}

pub const ROOK_MASK: [u64; 64] = build_masks(&ROOK_DIRS);
pub const BISHOP_MASK: [u64; 64] = build_masks(&BISHOP_DIRS);

/// Spread the low bits of `subset` onto the set bits of `mask`.
fn expand(subset: u64, mask: u64) -> u64 {
    let mut occ = 0;
    let mut sub = subset;
    let mut m = mask;
    while m != 0 {
        let b = m & m.wrapping_neg();
        m &= m - 1;
        if sub & 1 != 0 {
            occ |= b;
        }
        sub >>= 1;
    }
    occ
}

// Hardcoded magic constants, found once offline and validated by the test
// `magic_attacks_match_reference` which compares the built tables against
// the ray-walking reference across 1024 random occupancies per square.

pub const ROOK_MAGICS: [u64; 64] = [
    0x0080024002208090, 0x02C0002000421000, 0x8200120040088220, 0x4080080082445000,
    0x408008000C008006, 0x110002A804000300, 0x0100020002890004, 0x2080002080005B00,
    0x84018000802CC000, 0x1048402010004000, 0x0020802001100280, 0x8000801000800800,
    0x0029000C48009100, 0x000180800C000200, 0x0201000100140200, 0x0004802243000180,
    0x8021228000400280, 0x0005020022004880, 0x1081010040102000, 0x0001010010000824,
    0x2002020010A00408, 0x8181010002080C00, 0x0043140048010210, 0x04501200024400A7,
    0x0080004240082000, 0x4030014840072000, 0x0120006180100880, 0x0008101A00220040,
    0x0228580080040081, 0x0422000200049048, 0x0051000100020034, 0x0000008200010044,
    0x8006204000801080, 0x2130082000404002, 0x0242002082004010, 0x1220380280801000,
    0x10C0080080804C00, 0x2084008004800200, 0x0018286A24000110, 0x3000512042000084,
    0x81006040008A8008, 0x4821A01000404000, 0x0010002000808031, 0x41410010000D0020,
    0x00020010200A0004, 0x3101440006008080, 0x2401000200010084, 0x00000088410A000C,
    0x2024402080010100, 0x0888201003C00040, 0x040050C180220A00, 0x0802004020502E00,
    0x0005002800126500, 0x0804002010082401, 0x0020021810011400, 0x00400041040C8200,
    0x0902002081009242, 0x0003604000850011, 0x8020012042110009, 0x0004900100142821,
    0x8242002004100812, 0x8001000802140011, 0x0002000919840816, 0x0000890400418832,
];

pub const BISHOP_MAGICS: [u64; 64] = [
    0xC2286258004C0280, 0x0004410404088610, 0x0009020C21002108, 0x0412208209000100,
    0x2201104114820024, 0x0001242060000010, 0x1911280E02208400, 0x001124A804100800,
    0x2000A00485280100, 0x0000220204040480, 0x4001100400404000, 0x0010440400800040,
    0x0000040420302081, 0x820482022A208403, 0x0042220104024202, 0x000080AC01080802,
    0x0044805020062400, 0x2020010418008702, 0x0408013000802028, 0x0294012049002000,
    0x4202000422010000, 0x000A800030100820, 0x1201204201104604, 0x0409000200421200,
    0x642824004A901000, 0x08104400020C0400, 0x0040500103040080, 0x0010808018020152,
    0x000100102900C010, 0x4200430002008200, 0x000C040000455C22, 0x050400CA11004212,
    0x8108021805400882, 0x0818280C46820420, 0x1080105400880802, 0x0820400A00002200,
    0x3004110400080408, 0x000810010008A080, 0x0804040C10204900, 0x0A682A8088002204,
    0x0880940420004051, 0x60A1412860008840, 0x0006020202108100, 0x0008004200801809,
    0x0010088100409400, 0x4108810800208202, 0x0004A41404080044, 0x4012008210800200,
    0x010C110410842810, 0x2402860090048002, 0x0005009048280008, 0x0020208084044010,
    0xA014402021490088, 0x61C0A83014482090, 0x0040522484048000, 0x0020021A02042000,
    0x08010B0080844000, 0x0121422218241440, 0x8380004164020880, 0x43201C0108218810,
    0xC608030008030400, 0x1040001020020424, 0x87000460520C0900, 0x8C401126408E0280,
];

const fn compute_shifts(masks: &[u64; 64]) -> [u32; 64] {
    let mut s = [0u32; 64];
    let mut i = 0;
    while i < 64 {
        s[i] = 64 - masks[i].count_ones();
        i += 1;
    }
    s
}

const ROOK_SHIFTS: [u32; 64] = compute_shifts(&ROOK_MASK);
const BISHOP_SHIFTS: [u32; 64] = compute_shifts(&BISHOP_MASK);

// ---- Flat static storage (no pointer indirection on hot path) ---------------
// Magics and shifts live in `.rodata` as const statics; offsets and the flat
// attack table are built once (at first attack lookup) into a heap-allocated
// boxed slice.  After init the hot path is: mask + multiply + shift + one
// table load through the OnceLock.

use std::sync::OnceLock;

struct MagicTable {
    offsets: [u32; 64],
    table: Box<[u64]>,
}

fn build_table(mask: [u64; 64], dirs: [usize; 4], magics: &[u64; 64], shifts: &[u32; 64]) -> MagicTable {
    let mut offsets = [0u32; 64];
    let total: usize = (0..64).map(|sq| 1usize << (64 - shifts[sq]) as usize).sum();
    let mut tbl = vec![0u64; total];
    let mut base = 0usize;
    for sq in 0..64 {
        offsets[sq] = base as u32;
        let size = 1usize << (64 - shifts[sq]) as usize;
        for sub in 0..size {
            let occ = expand(sub as u64, mask[sq]);
            let idx = ((occ.wrapping_mul(magics[sq])) >> shifts[sq]) as usize;
            tbl[base + idx] = slider_attack(occ, sq, &dirs);
        }
        base += size;
    }
    MagicTable { offsets, table: tbl.into_boxed_slice() }
}

static ROOK_TABLES: OnceLock<MagicTable> = OnceLock::new();
static BISHOP_TABLES: OnceLock<MagicTable> = OnceLock::new();

#[inline(always)]
pub fn rook_attacks(occ: u64, sq: usize) -> u64 {
    let t = ROOK_TABLES
        .get_or_init(|| build_table(ROOK_MASK, ROOK_DIRS, &ROOK_MAGICS, &ROOK_SHIFTS));
    let idx =
        ((occ & ROOK_MASK[sq]).wrapping_mul(ROOK_MAGICS[sq]) >> ROOK_SHIFTS[sq]) as usize;
    t.table[t.offsets[sq] as usize + idx]
}

#[inline(always)]
pub fn bishop_attacks(occ: u64, sq: usize) -> u64 {
    let t = BISHOP_TABLES
        .get_or_init(|| build_table(BISHOP_MASK, BISHOP_DIRS, &BISHOP_MAGICS, &BISHOP_SHIFTS));
    let idx =
        ((occ & BISHOP_MASK[sq]).wrapping_mul(BISHOP_MAGICS[sq]) >> BISHOP_SHIFTS[sq]) as usize;
    t.table[t.offsets[sq] as usize + idx]
}

#[inline(always)]
pub fn queen_attacks(occ: u64, sq: usize) -> u64 {
    rook_attacks(occ, sq) | bishop_attacks(occ, sq)
}

pub fn pawn_attacks(color: usize, sq: usize) -> u64 {
    PAWN_ATTACKS[color][sq]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::splitmix;

    #[test]
    fn reduced_masks_have_expected_sizes() {
        for sq in 0..64 {
            // Reachable squares = 7 rank + 7 file (14); each direction that
            // reaches the board edge loses one outermost square.
            let dirs = (rank_of(sq) < 7) as u32
                + (rank_of(sq) > 0) as u32
                + (file_of(sq) < 7) as u32
                + (file_of(sq) > 0) as u32;
            let expect_r = 14 - dirs;
            assert_eq!(ROOK_MASK[sq].count_ones(), expect_r, "rook mask popcount at {sq}");
            assert!(BISHOP_MASK[sq].count_ones() <= 9, "bishop mask popcount at {sq}");
        }
        let rook_total: usize = ROOK_MASK.iter().map(|m| 1usize << m.count_ones()).sum();
        let bishop_total: usize = BISHOP_MASK.iter().map(|m| 1usize << m.count_ones()).sum();
        assert_eq!(rook_total, 102400);
        assert_eq!(bishop_total, 5248);
    }

    #[test]
    fn magic_attacks_match_reference() {
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        for sq in 0..64 {
            for _ in 0..1024 {
                rng = splitmix(rng);
                let occ = rng | (rng.rotate_left(19) ^ 0xFF_DB_6F_8A_52_00_00_2B);
                assert_eq!(
                    rook_attacks(occ, sq),
                    slider_attack(occ, sq, &ROOK_DIRS),
                    "rook attack mismatch on {sq}"
                );
                assert_eq!(
                    bishop_attacks(occ, sq),
                    slider_attack(occ, sq, &BISHOP_DIRS),
                    "bishop attack mismatch on {sq}"
                );
            }
        }
    }
}