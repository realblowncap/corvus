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

//! Score presentation helpers, mirroring Stockfish 19's `score.cpp` / `uci.cpp`.
//!
//! Stockfish's internal search values are not centipawns: the NNUE output is
//! normalised per material with the fitted WDL model before being reported to
//! the GUI (`UCIEngine::to_cp`). The win-rate model below is Stockfish 19's
//! `win_rate_params` / `win_rate_model` (fitted against LTC fishtest data,
//! anchored at material count 58), transcribed with its exact constants so the
//! engine reports the same `cp`/`wdl` numbers for equivalent evaluations.

use crate::bitboard::*;
use crate::board::Board;

pub const MATE: i32 = 32000;
/// Stockfish reports tablebase wins as ~+20000cp; reserved for Syzygy probing.
pub const TB_CP: i32 = 20000;

/// Quadratic coefficients for the win-rate model's `a` (value scaling) and `b`
/// (spread), as functions of material count `m` (clamped to [17, 78], /58).
const WIN_RATE_A: [f64; 4] = [-142.72052667, 372.35176398, -340.71073572, 415.23490212];
const WIN_RATE_B: [f64; 4] = [5.93832785, 15.61267078, -30.57816876, 69.63866711];

/// Material count in Stockfish's units: pawns 1, minors 3, rooks 5, queens 9.
pub fn material_count(board: &Board) -> i32 {
    let mut m = 0;
    for color in 0..2 {
        m += board.pieces[color][PAWN].count_ones() as i32;
        m += 3 * board.pieces[color][KNIGHT].count_ones() as i32;
        m += 3 * board.pieces[color][BISHOP].count_ones() as i32;
        m += 5 * board.pieces[color][ROOK].count_ones() as i32;
        m += 9 * board.pieces[color][QUEEN].count_ones() as i32;
    }
    m
}

/// The `(a, b)` parameters of the win-rate model for a position's material.
pub fn win_rate_params(material: i32) -> (f64, f64) {
    let m = (material.clamp(17, 78) as f64) / 58.0;
    let a = ((WIN_RATE_A[0] * m + WIN_RATE_A[1]) * m + WIN_RATE_A[2]) * m + WIN_RATE_A[3];
    let b = ((WIN_RATE_B[0] * m + WIN_RATE_B[1]) * m + WIN_RATE_B[2]) * m + WIN_RATE_B[3];
    (a, b)
}

/// Win rate in per mille: `1 / (1 + exp((a - v) / b))`, rounded to nearest.
pub fn win_rate_model(v: i32, material: i32) -> i32 {
    let (a, b) = win_rate_params(material);
    (0.5 + 1000.0 / (1.0 + ((a - v as f64) / b).exp())) as i32
}

/// `wdl_w wdl_d wdl_l` per mille, as UCI `wdl` expects.
pub fn wdl(v: i32, material: i32) -> (i32, i32, i32) {
    let w = win_rate_model(v, material);
    let l = win_rate_model(-v, material);
    (w, 1000 - w - l, l)
}

/// Convert an internal-unit value to centipawns for UCI reporting: Stockfish's
/// `to_cp`, i.e. `round(100 * v / a)`. Only valid for search values produced by
/// an internal-units evaluation (NNUE); the hand-crafted eval already reports
/// centipawns and must not be rescaled.
pub fn to_cp(v: i32, material: i32) -> i32 {
    let (a, _) = win_rate_params(material);
    (100.0 * v as f64 / a).round() as i32
}

/// `score cp X` or `score mate N`, matching Stockfish's `format_score`
/// (positive mate distance rounds up, negative truncates, as C++ does).
pub fn score_string(v: i32) -> String {
    if v.abs() >= MATE - 1000 {
        let plies = MATE - v.abs();
        let m = if v > 0 {
            (plies + 1) / 2
        } else {
            -((plies + 1) / 2)
        };
        format!("score mate {}", m)
    } else {
        format!("score cp {}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wdl_sums_to_one_thousand() {
        for v in [-2000, -500, 0, 250, 1500] {
            let (w, d, l) = wdl(v, 58);
            assert_eq!(w + d + l, 1000, "wdl must be per-mille: v={}", v);
            assert!(w >= 0 && d >= 0 && l >= 0);
        }
    }

    #[test]
    fn wdl_is_monotonic_and_centred() {
        let (w0, d0, _) = wdl(0, 58);
        let (w1, _, _) = wdl(500, 58);
        assert!(w1 > w0, "positive score must raise the win rate");
        assert!(d0 > 300, "equal position should be mostly drawn: {} permille", d0);
    }

    #[test]
    fn to_cp_keeps_scale_near_stockfish() {
        // At the anchor material (58) the value scaling `a` is ~304, so a
        // 300-internal-unit advantage reports as roughly +100cp.
        let cp = to_cp(304, 58);
        assert!((90..=110).contains(&cp), "unexpected cp {}", cp);
    }

    #[test]
    fn score_string_names_mates() {
        assert_eq!(score_string(MATE - 1), "score mate 1");
        assert_eq!(score_string(MATE - 3), "score mate 2");
        assert_eq!(score_string(-(MATE - 3)), "score mate -2");
        assert_eq!(score_string(42), "score cp 42");
    }
}
