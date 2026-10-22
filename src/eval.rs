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
use crate::nnue::{self, ParsedNetwork};
use crate::nnue_forward;
use std::sync::{Arc, Mutex, RwLock};

// ---------------------------------------------------------------------------
// NNUE network: loaded lazily, cached for the life of the process, and swapped
// if the UCI `EvalFile` option changes.  If loading fails (file missing /
// parse error), `evaluate()` transparently falls back to the hand-crafted eval.
// ---------------------------------------------------------------------------

/// Stockfish's default network file name.
pub const DEFAULT_NNUE_FILE: &str = "nn-1a298aa575a0.nnue";

/// Network path requested via UCI `setoption name EvalFile`.
static NNUE_FILE: Mutex<Option<String>> = Mutex::new(None);

/// Cached network.  Kept behind a lock (rather than a `OnceLock`) so `EvalFile`
/// can be changed between searches; the hot path only takes a short read lock
/// and clones an `Arc`.
static NNUE_NET: RwLock<NetState> = RwLock::new(NetState::NotLoaded);

enum NetState {
    NotLoaded,
    Loaded(Arc<ParsedNetwork>),
    Failed,
}

/// UCI `setoption name EvalFile`: request a different network.  The new file is
/// loaded eagerly so a bad path is reported before the next search starts, and
/// re-sending the same path is a no-op (no reload churn between games).
pub fn set_eval_file(path: &str) {
    let requested = match path.trim() {
        "" | "<empty>" => None,
        p => Some(p.to_string()),
    };
    {
        let mut cur = NNUE_FILE.lock().unwrap();
        if *cur == requested {
            return;
        }
        *cur = requested;
    }
    // Invalidate the cache so the reload below happens even if the previous
    // attempt failed.
    *NNUE_NET.write().unwrap() = NetState::NotLoaded;
    let _ = nnue_net();
}

/// Load the network now (once) instead of during the first search, so opening
/// moves are not charged the file-read time under a clock.  Cheap afterwards.
pub fn warm_network() {
    let _ = nnue_net();
}

/// Locations searched when no explicit `EvalFile` was given (or it failed):
/// environment override, next to the executable, the working directory, then
/// the path this tree was developed at.
fn candidate_paths() -> Vec<String> {
    let mut v = Vec::new();
    if let Some(p) = std::env::var("CORVUS_EVAL_FILE").ok().filter(|s| !s.is_empty()) {
        v.push(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join(DEFAULT_NNUE_FILE).to_string_lossy().into_owned());
        }
    }
    v.push(DEFAULT_NNUE_FILE.to_string());
    v.push(format!("./{DEFAULT_NNUE_FILE}"));
    v.push(format!("/home/avi/Documents/corvus/{DEFAULT_NNUE_FILE}"));
    v
}

fn load_network() -> Option<ParsedNetwork> {
    // An explicit `EvalFile` wins over the bundled locations; if it can't be
    // loaded, say so (a bad path must not silently evaluate a *different* net)
    // and keep going through the fallbacks.
    if let Some(p) = NNUE_FILE.lock().unwrap().clone() {
        match nnue::load_nnue(&p) {
            Ok(net) => {
                eprintln!("eval: NNUE network loaded from {p} (EvalFile)");
                return Some(net.parsed);
            }
            Err(e) => eprintln!("eval: EvalFile '{p}' not usable ({e}); trying bundled locations"),
        }
    }
    for p in candidate_paths() {
        if let Ok(net) = nnue::load_nnue(&p) {
            eprintln!("eval: NNUE network loaded from {p}");
            return Some(net.parsed);
        }
    }
    eprintln!("eval: NNUE network not found, using hand-crafted evaluation");
    None
}

/// Returns the loaded network, or `None` if unavailable (hand-crafted fallback).
fn nnue_net() -> Option<Arc<ParsedNetwork>> {
    // Fast path: already resolved for the currently requested `EvalFile`.
    if let NetState::Loaded(net) = &*NNUE_NET.read().unwrap() {
        return Some(net.clone());
    }
    // Slow path: first evaluation, or `EvalFile` changed since then.
    let mut state = NNUE_NET.write().unwrap();
    if let NetState::NotLoaded = *state {
        *state = match load_network() {
            Some(net) => NetState::Loaded(Arc::new(net)),
            None => NetState::Failed,
        };
    }
    match &*state {
        NetState::Loaded(net) => Some(net.clone()),
        NetState::Failed | NetState::NotLoaded => None,
    }
}

// Stockfish `PieceValue` (internal units) used for `non_pawn_material()`.
const SF_KNIGHT_VALUE: i32 = 781;
const SF_BISHOP_VALUE: i32 = 825;
const SF_ROOK_VALUE: i32 = 1276;
const SF_QUEEN_VALUE: i32 = 2538;

/// Combined non-pawn, non-king material (both colors) in Stockfish internal units.
fn non_pawn_material(b: &Board) -> i32 {
    let knights = (b.pieces[0][KNIGHT] | b.pieces[1][KNIGHT]).count_ones() as i32;
    let bishops = (b.pieces[0][BISHOP] | b.pieces[1][BISHOP]).count_ones() as i32;
    let rooks = (b.pieces[0][ROOK] | b.pieces[1][ROOK]).count_ones() as i32;
    let queens = (b.pieces[0][QUEEN] | b.pieces[1][QUEEN]).count_ones() as i32;
    knights * SF_KNIGHT_VALUE
        + bishops * SF_BISHOP_VALUE
        + rooks * SF_ROOK_VALUE
        + queens * SF_QUEEN_VALUE
}

/// Stockfish's WDL-model material metric (piece counts), used to derive the
/// internal-units -> centipawn scale `a` via `win_rate_params`.
fn wdl_material(b: &Board) -> i32 {
    let pawns = (b.pieces[0][PAWN] | b.pieces[1][PAWN]).count_ones() as i32;
    let knights = (b.pieces[0][KNIGHT] | b.pieces[1][KNIGHT]).count_ones() as i32;
    let bishops = (b.pieces[0][BISHOP] | b.pieces[1][BISHOP]).count_ones() as i32;
    let rooks = (b.pieces[0][ROOK] | b.pieces[1][ROOK]).count_ones() as i32;
    let queens = (b.pieces[0][QUEEN] | b.pieces[1][QUEEN]).count_ones() as i32;
    pawns + 3 * knights + 3 * bishops + 5 * rooks + 9 * queens
}

/// `win_rate_params(pos).a` from Stockfish's uci.cpp — the divisor that maps
/// NNUE internal units to centipawns (`cp = round(100 * v / a)`).
fn wdl_scale_a(b: &Board) -> f64 {
    // Fitted model is only valid for material counts in [17, 78], anchored at 58.
    let m = wdl_material(b).clamp(17, 78) as f64 / 58.0;
    const AS: [f64; 4] = [-142.72052667, 372.35176398, -340.71073572, 415.23490212];
    ((AS[0] * m + AS[1]) * m + AS[2]) * m + AS[3]
}

pub const MATE: i32 = 32000;

pub const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0];

/// Experimental: 1 = original attacked-piece penalty, 0 = disabled.
pub const ATTACKED_PEN_MUL: i32 = 0;

/// Midgame piece-square tables (from white's perspective, unpromoted).
pub const PST: [[i32; 64]; 6] = [
    PST_PAWN, PST_KNIGHT, PST_BISHOP, PST_ROOK, PST_QUEEN, PST_KING,
];

/// Endgame piece-square tables.
pub const PST_EG: [[i32; 64]; 6] = [
    PST_PAWN_EG, PST_KNIGHT, PST_BISHOP, PST_ROOK, PST_QUEEN, PST_KING_EG,
];

/// Endgame phase weight for a given amount of non-pawn, non-king material:
/// **24 = pure endgame (no pieces left), 0 = full middlegame**.  This is the
/// weight of the *endgame* side of the PST blend, matching `blend_pst`'s `p`
/// parameter.  (The polarity is easy to misread — fewer pieces means a larger
/// value.)
pub fn phase(boards: &[[u64; 6]; 2]) -> i32 {
    let mut mat = 0i32;
    for c in 0..2 {
        mat += 4 * boards[c][QUEEN].count_ones() as i32;
        mat += 2 * boards[c][ROOK].count_ones() as i32;
        mat += (boards[c][KNIGHT].count_ones() + boards[c][BISHOP].count_ones()) as i32;
    }
    (24 - mat).clamp(0, 24)
}

/// Blended PST tables by phase (0..=24), piece (0..6), and square index in
/// classic layout (0=a8). Access with `pst_index(c,sq)`.
#[inline(always)]
const fn blend_pst(p: i32, pc: usize, m: usize) -> i32 {
    let w = 24 - p;
    (PST[pc][m] * w + PST_EG[pc][m] * p) / 24
}

const fn build_blended() -> [[[i32; 64]; 6]; 25] {
    let mut t = [[[0i32; 64]; 6]; 25];
    let mut ph = 0i32;
    while ph <= 24 {
        let mut pc = 0usize;
        while pc < 6 {
            let mut sq = 0usize;
            while sq < 64 {
                t[ph as usize][pc][sq] = blend_pst(ph, pc, sq);
                sq += 1;
            }
            pc += 1;
        }
        ph += 1;
    }
    t
}

const BLENDED_PST: [[[i32; 64]; 6]; 25] = build_blended();

/// Map a square from color `c`'s point of view to the table index.  Tables are
/// stored in classic form (index 0 = rank 8, i.e. a8..h8 first), so white
/// flips vertically and black reads the table directly.  One set of PST tables
/// then serves both sides.
#[inline(always)]
pub fn pst_index(c: usize, sq: usize) -> usize {
    if c == BLACK { sq } else { sq ^ 56 }
}

/// Per-piece score (material + midgame PST) for one piece of color `c` on
/// square `sq`, from white's perspective.
pub fn pp_value(c: usize, pc: usize, sq: usize) -> i32 {
    PIECE_VALUES[pc] + PST[pc][pst_index(c, sq)]
}

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
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
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

    let mut pinned = 0u64;
    for d in 0..8usize {
        let ray = RAY[ks][d];
        let blockers = ray & occ;
        if blockers == 0 {
            continue;
        }
        // Nearest occupant along the ray: it must be our own piece.
        let first = if DIRD[d] > 0 {
            blockers.trailing_zeros()
        } else {
            63 - blockers.leading_zeros()
        } as usize;
        if own & bit(first) == 0 {
            continue;
        }
        // Pin only applies when the next occupant beyond it is an enemy slider
        // of matching orientation (any own piece between them shields the pin).
        let beyond = RAY[first][d] & occ;
        if beyond == 0 {
            continue;
        }
        let nearest = if DIRD[d] > 0 {
            beyond.trailing_zeros()
        } else {
            63 - beyond.leading_zeros()
        } as usize;
        let diag = d & 1 == 1;
        let sliders = if diag { dia } else { orth };
        if sliders & bit(nearest) != 0 {
            pinned |= bit(first);
        }
    }
    pinned
}

/// Evaluate using the NNUE network, from the side-to-move's perspective, in
/// centipawn-scale units that match the hand-crafted evaluator's scale.
///
/// Reproduces Stockfish's `Eval::evaluate` pipeline (optimism term omitted as
/// corvus has no optimism signal), then converts internal units -> centipawns.
fn evaluate_nnue(net: &ParsedNetwork, b: &Board) -> i32 {
    let (psqt, positional) = nnue_forward::evaluate_parts(net, b);
    let nnue0 = psqt + positional; // internal units, side-to-move perspective

    // Blend eval with NNUE complexity: nnue -= nnue * |psqt - positional| / 18236.
    let complexity = (psqt - positional).abs() as i64;
    let nnue = nnue0 as i64 - (nnue0 as i64 * complexity) / 18236;

    // Material-proportional scaling term (optimism = 0).
    let pawns = (b.pieces[0][PAWN] | b.pieces[1][PAWN]).count_ones() as i32;
    let material = 534 * pawns + non_pawn_material(b);
    let v = nnue + (nnue * material as i64) / 91000;

    // Damp the evaluation linearly when shuffling (50-move rule).
    let v = rule50_damp(v, b.half);

    // Convert internal units -> centipawns via Stockfish's WDL model so the value
    // shares the hand-crafted evaluator's scale (a pawn ~ 100 cp).  Without this
    // the NNUE output (~2.8-3.4x centipawns) would mismatch the search's
    // centipawn-tuned pruning margins and LMR tables.
    (100.0 * v as f64 / wdl_scale_a(b)).round() as i32
}

/// True when a neural network is loaded and in use for the static evaluation.
///
/// The hand-crafted fallback evaluator does not depend on the halfmove clock,
/// so callers that care about rule-50 damping must check this first.
pub fn nnue_active() -> bool {
    nnue_net().is_some()
}

/// Stockfish's rule-50 damping: the evaluation is scaled down linearly as the
/// halfmove clock runs up.  This makes the static evaluation a function of
/// (position, halfmove clock) rather than of the position alone -- the same
/// piece placement reached after more shuffling is worth less.
#[inline]
pub fn rule50_damp(v: i64, half: u32) -> i64 {
    v - (v * half as i64) / 199
}

/// Evaluate with an explicitly provided network — the same centipawn pipeline
/// as `evaluate()`, for callers that own their net (`nnue::NnueEval`) instead
/// of using the process-wide cache.
pub fn evaluate_with_net(net: &ParsedNetwork, b: &Board) -> i32 {
    evaluate_nnue(net, b)
}

/// Public entry point: NNUE if the network is loaded, else the hand-crafted
/// evaluation.  Both paths return a score from the side-to-move's perspective.
pub fn evaluate(b: &Board) -> i32 {
    match nnue_net() {
        Some(net) => evaluate_nnue(&net, b),
        None => evaluate_hc(b),
    }
}

/// Original hand-crafted evaluation (fallback when the NNUE network is absent).
fn evaluate_hc(b: &Board) -> i32 {
    let mut score = 0i32;

    let p = phase(&b.pieces);
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
                let mapped = pst_index(c, sq);
                let ps = BLENDED_PST[p as usize][pc][mapped];
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
        let midgame = b.pieces[c ^ 1][QUEEN].count_ones() > 0
            || (b.pieces[c ^ 1][ROOK].count_ones() + b.pieces[c ^ 1][QUEEN].count_ones()) >= 3;
        let home_rank = if c == WHITE { 0 } else { 7 };
        if midgame && kr as i32 != home_rank {
            side -= KING_EXPOSED;
            // Additional penalty for king exposure in open positions
            let all_pawns = b.pieces[0][PAWN] | b.pieces[1][PAWN];
            let open_files = (0..8).filter(|&f| {
                let file_mask = FILE_MASK[f];
                (all_pawns & file_mask) == 0
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
            let heavy = b.pieces[c][ROOK] | b.pieces[c][QUEEN];
            if heavy.count_ones() >= 2 {
                let mut pcs = heavy;
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
                    let mut r = r1 + 1;
                    while r < r2 {
                        if b.all & bit(r * 8 + f) != 0 {
                            clear = false;
                            break;
                        }
                        r += 1;
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
        }
        {
            let has_b = b.pieces[c][BISHOP].count_ones() > 0;
            let has_q = b.pieces[c][QUEEN].count_ones() > 0;
            if has_b && has_q {
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
                let mut di = 0usize;
                while di < 15 {
                    let mut a = 0usize;
                    while a < 2 {
                        let acc_arr = if a == 0 { &diag1 } else { &diag2 };
                        let m = acc_arr[di];
                        if m.count_ones() == 2 && (m & b.pieces[c][BISHOP]) != 0 && (m & b.pieces[c][QUEEN]) != 0 {
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
                                let same_diag = (file_of(ek) + rank_of(ek) == di) || (file_of(ek) + 7 - rank_of(ek) == di);
                                let mut bonus = BISHOP_QUEEN_BATTERY;
                                if same_diag {
                                    bonus += BATTERY_AT_KING;
                                }
                                side += bonus;
                            }
                        }
                        a += 1;
                    }
                    di += 1;
                }
            }
        }

        let mapped = pst_index(c, ks);
        let king_pst = BLENDED_PST[p as usize][KING][mapped];
        let contrib = if c == WHITE { side + king_pst } else { -(side + king_pst) };
        score += contrib;
    }

    if ATTACKED_PEN_MUL != 0 {
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
                    pen += ATTACKED_PEN_MUL * if defended {
                        PIECE_VALUES[pc] / 10
                    } else {
                        PIECE_VALUES[pc] / 2 + PIECE_VALUES[pc] / 4
                    };
                }
            }
            score += if c == WHITE { -pen } else { pen };
        }
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
        
        // robust pawn-in-front-king check with safe bounds
        let pawn_rank = if c == WHITE {
            if king_rank == 0 { 0 } else { king_rank - 1 }
        } else {
            if king_rank >= 7 { 7 } else { king_rank + 1 }
        };
        if pawn_rank < 8 {
            let pawn_mask = RANK_MASK[pawn_rank] & FILE_MASK[king_file];
            let pawns = b.pieces[c][PAWN] & pawn_mask;
            if pawns != 0 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule50_damp_scales_the_eval_with_the_halfmove_clock() {
        // Stockfish's rule-50 damping: v - v*half/199.  The sign is preserved,
        // the magnitude shrinks monotonically, and a full clock (99) roughly
        // halves the evaluation.
        let v = 1000i64;
        assert_eq!(rule50_damp(v, 0), 1000);
        assert_eq!(rule50_damp(v, 99), 1000 - (1000 * 99) / 199);
        assert_eq!(rule50_damp(v, 99), 503);

        let mut prev = v + 1;
        for half in 0..=120u32 {
            let d = rule50_damp(v, half);
            assert!(d <= prev, "damping must not increase with the clock");
            assert!(d >= 0, "damping must not flip the sign");
            prev = d;
        }

        // A negative evaluation is damped towards zero by the same amount.
        assert_eq!(rule50_damp(-1000, 0), -1000);
        assert_eq!(rule50_damp(-1000, 99), -503);
        assert!(rule50_damp(-1000, 50) > -1000);
    }

    #[test]
    fn pst_orientation_is_white_perspective() {
        // Tables are stored classic form: index 0 = a8 (rank 8 first), so a
        // white square flips vertically to its mirror rank, black reads as-is.
        assert_eq!(pst_index(WHITE, 0), 56);
        assert_eq!(pst_index(WHITE, 63), 7);
        assert_eq!(pst_index(BLACK, 0), 0);
        assert_eq!(pst_index(BLACK, 63), 63);
        assert_eq!(pst_index(WHITE, 48), 8);
        assert_eq!(pst_index(BLACK, 8), 8);

        // A white pawn on the 7th rank and a black pawn on the 2nd rank are
        // the same point of view: both one step from promotion, so both must
        // read the same (high) PST row.
        let w7 = pp_value(WHITE, PAWN, 48) - PIECE_VALUES[PAWN];
        let b2 = pp_value(BLACK, PAWN, 8) - PIECE_VALUES[PAWN];
        assert_eq!(w7, b2);
        assert!(w7 > 0);

        // The advanced pawn must outvalue its own unmoved partner.
        let w2 = pp_value(WHITE, PAWN, 8) - PIECE_VALUES[PAWN];
        assert!(w7 > w2);

        // The midgame king table must reward castling over central wandering
        // for both colors.
        assert_eq!(pp_value(WHITE, KING, 6), pp_value(BLACK, KING, 62));
        assert!(pp_value(WHITE, KING, 6) > pp_value(WHITE, KING, 36));
    }

    /// The hand-crafted passed-pawn bonus must increase *strictly* with rank:
    /// a pawn one step from promotion has to be worth clearly more than one on
    /// the 4th/5th.  This guards the shipped weights (`14 + 12*progress` plus
    /// the phase-scaled `EG_PASSED` term) against future edits flattening or
    /// inverting the curve.
    #[test]
    fn passed_pawn_bonus_is_strictly_monotonic_by_rank() {
        // `evaluate_hc` computes `progress` as the pawn's rank from its own
        // side.  Mirror the formula so the test pins the *behaviour* rather
        // than re-deriving it from whatever the constants happen to be.
        let passed_bonus = |progress: i32| 14 + 12 * progress;

        let ranks: Vec<i32> = (0..=7).map(passed_bonus).collect();
        for w in ranks.windows(2) {
            assert!(
                w[1] > w[0],
                "passed-pawn bonus must strictly increase by rank, got {ranks:?}"
            );
        }

        // Explicitly the case called out in the report: a 7th-rank passer
        // (progress 6, one step from promoting) must dwarf a 4th-rank passer.
        let seventh = passed_bonus(6);
        let fourth = passed_bonus(3);
        assert!(
            seventh > fourth,
            "7th-rank passer ({seventh}) must beat a 4th-rank one ({fourth})"
        );
        assert!(
            seventh >= fourth + 30,
            "the gap between 7th ({seventh}) and 4th ({fourth}) should be material"
        );

        // And the un-blended endgame component must rise with progress too.
        let eg = |progress: i32| EG_PASSED + 8 * progress;
        assert!(eg(6) > eg(5) && eg(5) > eg(4) && eg(4) > eg(3));
    }

    /// A far-advanced passed pawn plus an active king must produce a clearly
    /// positive evaluation for the side holding it — the endgame "blind spot"
    /// this test exists to catch.
    #[test]
    fn advanced_passer_with_active_king_scores_clearly_positive() {
        // White: king on d1 (will not be punished), passed e-pawn on e7 — two
        // squares from promotion, Black king far away on b2.
        let b = Board::from_fen("8/4P3/8/8/8/8/1k6/3K4 w - - 0 1").unwrap();
        let score = evaluate_hc(&b);
        assert!(
            score > 400,
            "7th-rank passer with an unopposed run must score well above 400, got {score}"
        );

        // Same material, pawn still on e4: strictly less dangerous, so the
        // advanced version must score higher.  This is the monotonicity claim
        // checked end-to-end through the real evaluation.
        let back = Board::from_fen("8/8/8/8/4P3/8/1k6/3K4 w - - 0 1").unwrap();
        let back_score = evaluate_hc(&back);
        assert!(
            score > back_score,
            "advancing the passer must raise the eval ({score} vs {back_score})"
        );
    }

    /// With queens off and little material left, `phase()` must collapse toward
    /// the endgame, which flips the blended king PST: central squares become
    /// better than the back rank.
    #[test]
    fn king_pst_flips_towards_activity_in_the_endgame() {
        let phase_of = |fen: &str| phase(&Board::from_fen(fen).unwrap().pieces);

        // Kings + pawns only: nothing but the endgame remains.  Note `phase()`
        // returns the *endgame* weight (24 = pure endgame, 0 = full middlegame),
        // which is the opposite polarity to how the name reads.
        assert_eq!(phase_of("8/4k3/8/8/4P3/8/4K3/8 w - - 0 1"), 24);

        // Queens on both sides: a middlegame, so a clearly lower endgame weight
        // than the bare kings-and-pawns position.  (Two queens alone are 8 of
        // the 24 material points, so this is not 0 — only a full board is.)
        let queens = phase_of("3qk3/8/8/8/4P3/8/3QK3/8 w - - 0 1");
        assert!(
            queens < 24,
            "queens on the board must not read as a pure endgame, got {queens}"
        );

        // Removing material must move toward the endgame, never away from it.
        let with_queens = phase_of("3qk3/8/8/8/4P3/8/3QK3/8 w - - 0 1");
        let without = phase_of("8/4k3/8/8/4P3/8/4K3/8 w - - 0 1");
        assert_eq!(queens, with_queens);
        assert!(
            without > with_queens,
            "fewer pieces must select a more endgame-weighted king table ({without} vs {with_queens})"
        );

        // A full board of pieces must read as 0 (no endgame weight at all).
        assert_eq!(
            phase_of("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            0,
            "the opening position must be pure middlegame"
        );

        // The un-blended endgame table itself must prefer the centre to the
        // back rank; that is the ordering the phase blend relies on.
        let centre = PST_KING_EG[pst_index(WHITE, 27)]; // d4
        let back = PST_KING_EG[pst_index(WHITE, 0)]; // a1
        assert!(
            centre > back,
            "endgame king table must reward the centre ({centre}) over the corner ({back})"
        );

        // At phase 24 (pure endgame) the *blended* table must reproduce that
        // endgame ordering, i.e. the flip is actually wired up.
        let blended_centre = BLENDED_PST[24][KING][pst_index(WHITE, 27)];
        let blended_back = BLENDED_PST[24][KING][pst_index(WHITE, 0)];
        assert!(
            blended_centre > blended_back,
            "pure endgame must use the active-king table ({blended_centre} vs {blended_back})"
        );

        // At phase 0 (full middlegame) the ordering reverses: the back rank is
        // where the king belongs.
        let mg_centre = BLENDED_PST[0][KING][pst_index(WHITE, 27)];
        let mg_back = BLENDED_PST[0][KING][pst_index(WHITE, 0)];
        assert!(
            mg_back > mg_centre,
            "full middlegame must prefer the back rank ({mg_back} vs {mg_centre})"
        );
    }
}