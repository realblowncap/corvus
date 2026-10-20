// NNUE forward pass for the SFNNv16 network (`nn-1a298aa575a0.nnue`), ported
// faithfully from the Stockfish `sf_19` tag sources:
//   - features: HalfKAv2_hm (PSQ), FullThreats, PP_3wide
//   - accumulator: int16, PSQT: int32
//   - transform: int8 via clamp(0,255) pairwise product / 512 (SCReLU)
//   - network: 8 layer stacks (fc_0 sparse-input, SCReLU/ClippedReLU, skip
//     connection), output scaled by 600*OutputScale / (HiddenOneVal<<WeightScaleBits)*2
//
// The reference semantics are the scalar (non-SIMD) code paths of `sf_19`.
// The file stores weights in natural row-major order; Stockfish's `permute_weights`
// and `get_weight_index_scrambled` are SIMD memory-layout optimizations that are
// compensated downstream, so the final score is architecture-independent and the
// natural ordering used here reproduces it exactly.

use crate::bitboard::{
    bishop_attacks, pawn_attacks, queen_attacks, rook_attacks, KING_ATTACKS, KNIGHT_ATTACKS, BLACK,
    WHITE,
};
use crate::board::Board;
use crate::nnue::{ParsedNetwork, FULL_THREATS_DIM, L1, L2, PSQ_T_BUCKETS};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Stockfish encodings (differ from corvus's internal 0-based piece types).
// PieceType: PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6.
// Piece = (color << 3) + type.  Color: WHITE=0, BLACK=1 (same as corvus).
// ---------------------------------------------------------------------------
const PAWN: usize = 1;
const KNIGHT: usize = 2;
const BISHOP: usize = 3;
const ROOK: usize = 4;
const QUEEN: usize = 5;
const KING: usize = 6;

const PS_NB: usize = 11 * 64; // 704, HalfKAv2_hm piece-square stride

// Constants from nnue_common.h
const FT_MAX_VAL: i32 = 255;
const WEIGHT_SCALE_BITS: i32 = 6;
const OUTPUT_SCALE: i32 = 16;
const HIDDEN_ONE_VAL: i64 = 128;

// File masks (a1 = 0 convention, matching Stockfish).
const FILE_A: u64 = 0x0101010101010101;
const FILE_H: u64 = 0x8080808080808080;
const RANK_1: u64 = 0xFF;
const RANK_8: u64 = 0xFF00_0000_0000_0000;

// Direction deltas (Stockfish `Direction`).
const NORTH_EAST: i32 = 9;
const NORTH_WEST: i32 = 7;
const SOUTH_WEST: i32 = -9;
const SOUTH_EAST: i32 = -7;

// ---------------------------------------------------------------------------
// Minimal position view derived from a corvus `Board`, using Stockfish's
// piece encoding so the feature extraction matches the C++ source directly.
// ---------------------------------------------------------------------------
pub struct NnuePos {
    piece: [u8; 64],     // Stockfish Piece per square (0 = empty)
    occ: u64,            // all occupied squares
    pct: [u64; 16],      // bitboard per Stockfish Piece code
    ksq: [usize; 2],     // king squares, indexed by color
    side: usize,         // side to move
    piece_count: usize,  // total pieces (for bucket selection)
}

impl NnuePos {
    pub fn from_board(board: &Board) -> NnuePos {
        let mut piece = [0u8; 64];
        let mut pct = [0u64; 16];
        for sq in 0..64 {
            if let Some(c) = board.color_on(sq) {
                let cpt = board.piece_on(sq); // corvus piece type 0..5
                let spt = cpt + 1; // Stockfish piece type 1..6
                piece[sq] = ((c << 3) + spt) as u8;
                pct[(c << 3) + spt] |= 1u64 << sq;
            }
        }
        NnuePos {
            piece,
            occ: board.all,
            pct,
            ksq: board.kingsq,
            side: board.side,
            piece_count: board.all.count_ones() as usize,
        }
    }

    #[inline]
    fn pieces_c(&self, c: usize, pt: usize) -> u64 {
        self.pct[(c << 3) + pt]
    }

    #[inline]
    fn pieces_type(&self, pt: usize) -> u64 {
        self.pct[pt] | self.pct[8 + pt]
    }
}

// ---------------------------------------------------------------------------
// Bitboard helpers (mirroring Stockfish `shift` / `pawn_pair_bb`).
// ---------------------------------------------------------------------------
#[inline]
fn shift_east(b: u64) -> u64 {
    (b & !FILE_H) << 1
}
#[inline]
fn shift_west(b: u64) -> u64 {
    (b & !FILE_A) >> 1
}

#[inline]
fn shift_dir(b: u64, dir: i32) -> u64 {
    match dir {
        NORTH_EAST => (b & !FILE_H) << 9,
        NORTH_WEST => (b & !FILE_A) << 7,
        SOUTH_WEST => (b & !FILE_A) >> 9,
        SOUTH_EAST => (b & !FILE_H) >> 7,
        _ => 0,
    }
}

// Squares that can host a pawn forming a pair with a pawn on `s`: own file plus
// adjacent files, restricted to ranks 2-7, excluding `s` itself.
fn pawn_pair_bb(s: usize) -> u64 {
    let file = FILE_A << (s & 7);
    let files = file | shift_east(file) | shift_west(file);
    files & !(RANK_1 | RANK_8) & !(1u64 << s)
}

#[inline]
fn lsb(b: u64) -> usize {
    b.trailing_zeros() as usize
}

// Stockfish `Attacks::PseudoAttacks` indexed by the repurposed layout:
// [0]=white pawn, [1]=black pawn, [2]=KNIGHT..[6]=KING.
#[inline]
fn pseudo_attacks(sf_idx: usize, sq: usize) -> u64 {
    match sf_idx {
        0 => pawn_attacks(WHITE, sq),
        1 => pawn_attacks(BLACK, sq),
        2 => KNIGHT_ATTACKS[sq],
        3 => bishop_attacks(0, sq),
        4 => rook_attacks(0, sq),
        5 => queen_attacks(0, sq),
        6 => KING_ATTACKS[sq],
        _ => 0,
    }
}

#[inline]
fn attacks_bb(pt: usize, from: usize, occ: u64) -> u64 {
    match pt {
        KNIGHT => KNIGHT_ATTACKS[from],
        BISHOP => bishop_attacks(occ, from),
        ROOK => rook_attacks(occ, from),
        QUEEN => queen_attacks(occ, from),
        KING => KING_ATTACKS[from],
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Static feature-extraction tables (computed once).
// ---------------------------------------------------------------------------

// HalfKAv2_hm: PieceSquareIndex[color][piece].
const PIECE_SQUARE_INDEX: [[u32; 16]; 2] = [
    // WHITE perspective: own pieces get {0,128,256,384,512}, enemy {64,192,320,448,576}, KING 640.
    [0, 0, 128, 256, 384, 512, 640, 0, 0, 64, 192, 320, 448, 576, 640, 0],
    // BLACK perspective (colors swapped).
    [0, 64, 192, 320, 448, 576, 640, 0, 0, 0, 128, 256, 384, 512, 640, 0],
];

// HalfKAv2_hm OrientTBL: files a-d -> SQ_H1 (7), files e-h -> SQ_A1 (0).
#[inline]
fn halfka_orient(sq: usize) -> usize {
    if (sq & 4) != 0 {
        0
    } else {
        7
    }
}

// FullThreats / PP_3Wide OrientTBL: files a-d -> SQ_A1 (0), files e-h -> SQ_H1 (7).
#[inline]
fn threat_orient(sq: usize) -> usize {
    if (sq & 4) != 0 {
        7
    } else {
        0
    }
}

// KingBuckets[sq] = bucket * PS_NB, bucket = 4*(7-rank) + min(file, 7-file).
fn build_king_buckets() -> [u32; 64] {
    let mut tb = [0u32; 64];
    for sq in 0..64 {
        let rank = sq / 8;
        let file = sq & 7;
        let bucket = 4 * (7 - rank) + file.min(7 - file);
        tb[sq] = (bucket * PS_NB) as u32;
    }
    tb
}

const FULL_THREATS_MAP: [[i32; 6]; 6] = [
    [-1, 0, -1, 1, -1, -1], // PAWN attacker
    [0, 1, 2, 3, 4, -1],    // KNIGHT
    [0, 1, 2, 3, -1, -1],   // BISHOP
    [0, 1, 2, 3, -1, -1],   // ROOK
    [0, 1, 2, 3, 4, -1],    // QUEEN
    [-1, -1, -1, -1, -1, -1], // KING
];

const NUM_VALID_TARGETS: [i32; 16] = [0, 4, 10, 8, 8, 10, 0, 0, 0, 4, 10, 8, 8, 10, 0, 0];

// AllPieces order: W_PAWN..W_KING, B_PAWN..B_KING.
const ALL_PIECES: [usize; 12] = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14];

struct ThreatTables {
    king_buckets: [u32; 64],
    offsets: [[u32; 64]; 16],         // [attacker][from]
    helper_cum_piece: [u32; 16],      // cumulativePieceOffset per piece
    helper_cum_offset: [u32; 16],     // cumulativeOffset per piece
    index_lut1: [[[u32; 2]; 16]; 16], // [attacker][attacked][from < to]
    index_lut2: [[[u8; 64]; 64]; 16], // [piece][from][to]
}

fn build_threat_tables() -> ThreatTables {
    let mut t = ThreatTables {
        king_buckets: build_king_buckets(),
        offsets: [[0; 64]; 16],
        helper_cum_piece: [0; 16],
        helper_cum_offset: [0; 16],
        index_lut1: [[[0; 2]; 16]; 16],
        index_lut2: [[[0; 64]; 64]; 16],
    };

    // init_threat_offsets
    let mut cumulative_offset: u32 = 0;
    for &piece in ALL_PIECES.iter() {
        let mut cumulative_piece_offset: u32 = 0;
        for from in 0..64 {
            t.offsets[piece][from] = cumulative_piece_offset;
            let ptype = piece & 7;
            if ptype != PAWN {
                cumulative_piece_offset += pseudo_attacks(ptype, from).count_ones();
            } else if (8..=55).contains(&from) {
                // Pawns only exist on ranks 2-7.
                let color_idx = if piece < 8 { 0 } else { 1 };
                cumulative_piece_offset += pseudo_attacks(color_idx, from).count_ones();
            }
        }
        t.helper_cum_piece[piece] = cumulative_piece_offset;
        t.helper_cum_offset[piece] = cumulative_offset;
        cumulative_offset += (NUM_VALID_TARGETS[piece] as u32) * cumulative_piece_offset;
    }

    // index_lut2_array
    for &piece in ALL_PIECES.iter() {
        let ptype = piece & 7;
        let color = piece >> 3;
        for from in 0..64 {
            let attacks = if ptype == PAWN {
                pseudo_attacks(color, from)
            } else {
                pseudo_attacks(ptype, from)
            };
            for to in 0..64 {
                let mask = if to == 0 { 0 } else { (1u64 << to) - 1 };
                t.index_lut2[piece][from][to] = (mask & attacks).count_ones() as u8;
            }
        }
    }

    // init_index_luts
    for &attacker in ALL_PIECES.iter() {
        for &attacked in ALL_PIECES.iter() {
            let enemy = (attacker ^ attacked) == 8;
            let attacker_type = attacker & 7;
            let attacked_type = attacked & 7;
            let map = FULL_THREATS_MAP[attacker_type - 1][attacked_type - 1];
            let semi_excluded =
                attacker_type == attacked_type && (enemy || attacker_type != PAWN);
            let attacked_color = (attacked >> 3) as i32;
            let feature = t.helper_cum_offset[attacker]
                as i64
                + (attacked_color * (NUM_VALID_TARGETS[attacker] / 2) + map) as i64
                    * (t.helper_cum_piece[attacker] as i64);
            let excluded = map < 0;
            t.index_lut1[attacker][attacked][0] =
                if excluded { FULL_THREATS_DIM as u32 } else { feature as u32 };
            t.index_lut1[attacker][attacked][1] = if excluded || semi_excluded {
                FULL_THREATS_DIM as u32
            } else {
                feature as u32
            };
        }
    }

    t
}

fn tables() -> &'static ThreatTables {
    static TB: OnceLock<ThreatTables> = OnceLock::new();
    TB.get_or_init(build_threat_tables)
}

// ---------------------------------------------------------------------------
// Feature index computation.
// ---------------------------------------------------------------------------

// HalfKAv2_hm make_index.
fn make_index_psq(perspective: usize, s: usize, pc: u8, ksq: usize, tb: &ThreatTables) -> u32 {
    let flip = 56 * perspective;
    (s ^ halfka_orient(ksq) ^ flip) as u32
        + PIECE_SQUARE_INDEX[perspective][pc as usize]
        + tb.king_buckets[ksq ^ flip]
}

// FullThreats make_index.
fn make_index_threat(
    perspective: usize,
    attacker: u8,
    from: usize,
    to: usize,
    attacked: u8,
    ksq: usize,
    tb: &ThreatTables,
) -> u32 {
    let orientation = threat_orient(ksq) ^ (56 * perspective);
    let from_o = from ^ orientation;
    let to_o = to ^ orientation;
    let swap = 8 * perspective;
    let attacker_o = (attacker as usize) ^ swap;
    let attacked_o = (attacked as usize) ^ swap;
    tb.index_lut1[attacker_o][attacked_o][(from_o < to_o) as usize]
        + tb.offsets[attacker_o][from_o]
        + tb.index_lut2[attacker_o][from_o][to_o] as u32
}

// PP_3Wide make_index.
fn make_index_pair(
    perspective: usize,
    color: usize,
    from: usize,
    to: usize,
    paired_color: usize,
    ksq: usize,
) -> u32 {
    let orientation = threat_orient(ksq) ^ (56 * perspective);
    let from_o = from ^ orientation;
    let to_o = to ^ orientation;
    let color_o = color ^ perspective;
    let paired_o = paired_color ^ perspective;
    let id_a = (48 * color_o + from_o - 8) as u32; // - SQ_A2 (8)
    let id_b = (48 * paired_o + to_o - 8) as u32;
    let hi = id_a.max(id_b);
    let lo = id_a.min(id_b);
    hi * (hi - 1) / 2 + lo + FULL_THREATS_DIM as u32
}

// ---------------------------------------------------------------------------
// Active feature enumeration (from-scratch / refresh path).
// ---------------------------------------------------------------------------

fn active_psq(pos: &NnuePos, perspective: usize, tb: &ThreatTables) -> Vec<u32> {
    let ksq = pos.ksq[perspective];
    let mut out = Vec::with_capacity(pos.piece_count);
    let mut occ = pos.occ;
    while occ != 0 {
        let s = lsb(occ);
        occ &= occ - 1;
        out.push(make_index_psq(perspective, s, pos.piece[s], ksq, tb));
    }
    out
}

fn active_threats(pos: &NnuePos, perspective: usize, tb: &ThreatTables) -> Vec<u32> {
    let ksq = pos.ksq[perspective];
    let occ = pos.occ;
    let pawn_targets = pos.pieces_type(KNIGHT) | pos.pieces_type(ROOK);
    let minor_slider_targets = pos.pieces_type(PAWN)
        | pos.pieces_type(KNIGHT)
        | pos.pieces_type(BISHOP)
        | pos.pieces_type(ROOK);
    let queen_targets = minor_slider_targets | pos.pieces_type(QUEEN);

    let mut out = Vec::new();

    // Pawn diagonal attacks. Pawns target knights and rooks only.
    let pawn_moves = [
        (WHITE, NORTH_EAST),
        (WHITE, NORTH_WEST),
        (BLACK, SOUTH_WEST),
        (BLACK, SOUTH_EAST),
    ];
    for &(c, dir) in pawn_moves.iter() {
        let cpawns = pos.pieces_c(c, PAWN);
        let mut attacks = shift_dir(cpawns, dir) & pawn_targets;
        while attacks != 0 {
            let to = lsb(attacks);
            attacks &= attacks - 1;
            let from = (to as i32 - dir) as usize;
            let attacked = pos.piece[to];
            let attacker = ((c << 3) + PAWN) as u8;
            let idx = make_index_threat(perspective, attacker, from, to, attacked, ksq, tb);
            if (idx as usize) < FULL_THREATS_DIM {
                out.push(idx);
            }
        }
    }

    // Sliding/minor attacks (pt from KNIGHT to QUEEN, i.e. pt < KING).
    for &c in [WHITE, BLACK].iter() {
        for &pt in [KNIGHT, BISHOP, ROOK, QUEEN].iter() {
            let attacker = ((c << 3) + pt) as u8;
            let mut bb = pos.pieces_c(c, pt);
            let targets = if pt == KNIGHT || pt == QUEEN {
                queen_targets
            } else {
                minor_slider_targets
            };
            while bb != 0 {
                let from = lsb(bb);
                bb &= bb - 1;
                let mut attacks = attacks_bb(pt, from, occ) & targets;
                while attacks != 0 {
                    let to = lsb(attacks);
                    attacks &= attacks - 1;
                    let attacked = pos.piece[to];
                    let idx = make_index_threat(perspective, attacker, from, to, attacked, ksq, tb);
                    if (idx as usize) < FULL_THREATS_DIM {
                        out.push(idx);
                    }
                }
            }
        }
    }

    out
}

fn active_pairs(pos: &NnuePos, perspective: usize) -> Vec<u32> {
    let ksq = pos.ksq[perspective];
    let white = pos.pieces_c(WHITE, PAWN);
    let black = pos.pieces_c(BLACK, PAWN);
    let mut out = Vec::new();

    // White pawns: pair with remaining white pawns (each unordered pair once,
    // because `bb` is consumed as we iterate) and with all black pawns.
    let mut bb = white;
    while bb != 0 {
        let from = lsb(bb);
        bb &= bb - 1;
        let band = pawn_pair_bb(from);
        let mut ww = band & bb;
        while ww != 0 {
            let to = lsb(ww);
            ww &= ww - 1;
            out.push(make_index_pair(perspective, WHITE, from, to, WHITE, ksq));
        }
        let mut wb = band & black;
        while wb != 0 {
            let to = lsb(wb);
            wb &= wb - 1;
            out.push(make_index_pair(perspective, WHITE, from, to, BLACK, ksq));
        }
    }

    // Black pawns: pair with remaining black pawns (black-white already covered).
    let mut bb = black;
    while bb != 0 {
        let from = lsb(bb);
        bb &= bb - 1;
        let band = pawn_pair_bb(from);
        let mut bbk = band & bb;
        while bbk != 0 {
            let to = lsb(bbk);
            bbk &= bbk - 1;
            out.push(make_index_pair(perspective, BLACK, from, to, BLACK, ksq));
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Accumulator + transform.
// ---------------------------------------------------------------------------

struct PerspectiveAcc {
    acc: [i16; L1],
    psqt: [i32; PSQ_T_BUCKETS],
}

fn accumulate(pos: &NnuePos, perspective: usize, net: &ParsedNetwork, tb: &ThreatTables) -> PerspectiveAcc {
    let mut acc = [0i16; L1];
    for j in 0..L1 {
        acc[j] = net.ft_biases[j];
    }
    let mut psqt = [0i32; PSQ_T_BUCKETS];

    // HalfKAv2_hm (PSQ) features: int16 weights + int32 psqt.
    for &f in active_psq(pos, perspective, tb).iter() {
        let row = f as usize * L1;
        for j in 0..L1 {
            acc[j] = acc[j].wrapping_add(net.ft_weights[row + j]);
        }
        let prow = f as usize * PSQ_T_BUCKETS;
        for b in 0..PSQ_T_BUCKETS {
            psqt[b] += net.ft_psqt_weights[prow + b];
        }
    }

    // FullThreats features: int8 weights (sign-extended) + int32 psqt.
    for &g in active_threats(pos, perspective, tb).iter() {
        let row = g as usize * L1;
        for j in 0..L1 {
            acc[j] = acc[j].wrapping_add(net.threat_weights[row + j] as i16);
        }
        let prow = g as usize * PSQ_T_BUCKETS;
        for b in 0..PSQ_T_BUCKETS {
            psqt[b] += net.threat_psqt[prow + b];
        }
    }

    // PP_3Wide features (indices offset by FULL_THREATS_DIM).
    for &gp in active_pairs(pos, perspective).iter() {
        let g = gp as usize - FULL_THREATS_DIM;
        let row = g * L1;
        for j in 0..L1 {
            acc[j] = acc[j].wrapping_add(net.pawn_pair_weights[row + j] as i16);
        }
        let prow = g * PSQ_T_BUCKETS;
        for b in 0..PSQ_T_BUCKETS {
            psqt[b] += net.pawn_pair_psqt[prow + b];
        }
    }

    PerspectiveAcc { acc, psqt }
}

// SCReLU transform: output[offset + j] = clamp(acc[j],0,255)*clamp(acc[j+512],0,255)/512.
fn transform(acc: &[i16; L1], perspective_p: usize, output: &mut [u8; L1]) {
    let offset = (L1 / 2) * perspective_p;
    for j in 0..L1 / 2 {
        let sum0 = (acc[j] as i32).clamp(0, FT_MAX_VAL);
        let sum1 = (acc[j + L1 / 2] as i32).clamp(0, FT_MAX_VAL);
        output[offset + j] = ((sum0 * sum1) / 512) as u8;
    }
}

// ---------------------------------------------------------------------------
// Network propagate (one layer stack).
// ---------------------------------------------------------------------------

fn propagate(input: &[u8; L1], stack: &crate::nnue::LayerStack) -> i32 {
    // fc_0: AffineTransformSparseInput<L1, L2>. Natural dense dot product.
    let mut fc0 = [0i32; L2];
    for j in 0..L2 {
        let mut sum = stack.fc_0.biases[j];
        let row = j * stack.fc_0.padded_input_dims;
        for i in 0..L1 {
            sum += (stack.fc_0.weights[row + i] as i32) * (input[i] as i32);
        }
        fc0[j] = sum;
    }

    // concat[0..32] = SqrClippedReLU(fc0), wsb = 7; concat[32..64] = ClippedReLU(fc0), wsb = 7.
    let mut concat = [0u8; 2 * L2 * 2];
    for i in 0..L2 {
        let x = fc0[i] as i64;
        concat[i] = (127i64).min((x * x) >> (2 * 7 + 7)) as u8;
    }
    for i in 0..L2 {
        concat[L2 + i] = (fc0[i] >> 7).clamp(0, 127) as u8;
    }

    // fc_1: AffineTransform<2*L2, L2>.
    let mut fc1 = [0i32; L2];
    for j in 0..L2 {
        let mut sum = stack.fc_1.biases[j];
        let row = j * stack.fc_1.padded_input_dims;
        for i in 0..2 * L2 {
            sum += (stack.fc_1.weights[row + i] as i32) * (concat[i] as i32);
        }
        fc1[j] = sum;
    }

    // concat[64..96] = SqrClippedReLU(fc1), wsb = 6; concat[96..128] = ClippedReLU(fc1), wsb = 6.
    for i in 0..L2 {
        let x = fc1[i] as i64;
        concat[2 * L2 + i] = (127i64).min((x * x) >> (2 * 6 + 7)) as u8;
    }
    for i in 0..L2 {
        concat[2 * L2 + L2 + i] = (fc1[i] >> 6).clamp(0, 127) as u8;
    }

    // fc_2: AffineTransform<2*L2 + 2*L3, 1>.
    let mut sum = stack.fc_2.biases[0];
    for i in 0..4 * L2 {
        sum += (stack.fc_2.weights[i] as i32) * (concat[i] as i32);
    }

    // Skip connection from raw fc_0 outputs and final output scaling.
    let fwd_out = sum + (fc0[L2 - 2] - fc0[L2 - 1]);
    let multiplier: i64 = 600 * OUTPUT_SCALE as i64;
    let denominator: i64 = HIDDEN_ONE_VAL * (1i64 << WEIGHT_SCALE_BITS) * 2;
    ((fwd_out as i64 * multiplier) / denominator) as i32
}

// ---------------------------------------------------------------------------
// Top-level evaluation (side-to-move perspective, internal units).
// ---------------------------------------------------------------------------

/// Evaluate a position with the SFNNv16 network, returning Stockfish's two
/// output halves separately (each already divided by `OutputScale` = 16), both
/// from the **side-to-move's** perspective.  This mirrors `Network::evaluate`,
/// which returns `{psqt/16, positional/16}`; Stockfish's `Eval::evaluate` then
/// combines them as `nnue = psqt + positional` and applies optimism/complexity/
/// material/rule50 adjustments.  Exposing the halves lets callers reproduce
/// those adjustments (the complexity term needs `|psqt - positional|`).
pub fn evaluate_parts(net: &ParsedNetwork, board: &Board) -> (i32, i32) {
    let pos = NnuePos::from_board(board);
    let tb = tables();

    let stm = pos.side;
    let nstm = stm ^ 1;
    let bucket = (pos.piece_count - 1) / 4;

    let acc_stm = accumulate(&pos, stm, net, tb);
    let acc_nstm = accumulate(&pos, nstm, net, tb);

    // psqt = (psqtAccum[stm][bucket] - psqtAccum[nstm][bucket]) / 2.
    let psqt_raw = (acc_stm.psqt[bucket] - acc_nstm.psqt[bucket]) / 2;

    let mut transformed = [0u8; L1];
    transform(&acc_stm.acc, 0, &mut transformed);
    transform(&acc_nstm.acc, 1, &mut transformed);

    let positional_raw = propagate(&transformed, &net.stacks[bucket]);

    (psqt_raw / OUTPUT_SCALE, positional_raw / OUTPUT_SCALE)
}

/// Evaluate a position with the SFNNv16 network, returning Stockfish's internal
/// "NNUE evaluation (side to move, internal units)" value: `psqt/16 + positional/16`.
pub fn evaluate(net: &ParsedNetwork, board: &Board) -> i32 {
    let (psqt, positional) = evaluate_parts(net, board);
    psqt + positional
}
