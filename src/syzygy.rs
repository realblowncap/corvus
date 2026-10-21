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

//! Syzygy endgame tablebase probing, via the vendored Fathom prober.
//!
//! Fathom is compiled by `build.rs`.  When it could not be compiled (no C
//! compiler) this module still exists but every probe reports "no tablebases",
//! so the engine degrades to normal search behaviour.
//!
//! Two entry points matter to the search:
//!
//! * [`probe_wdl`] — in-tree probe, returns a WDL score for the side to move.
//! * [`probe_root`] — root probe, returns the best move plus its DTZ, so the
//!   engine can convert won endgames instead of shuffling.
//!
//! Both require a *tablebase position*: no castling rights, and at most
//! `probe_limit + 1` pieces (Fathom reports `TB_LARGEST` for what it has).

#[cfg(corvus_fathom)]
use crate::bitboard::{BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};
use crate::board::{Board, Move};
use std::sync::OnceLock;

/// Fathom's WDL codes (see `vendor/fathom/tbprobe.h`).
pub const TB_LOSS: i32 = 0;
pub const TB_BLESSED_LOSS: i32 = 1;
pub const TB_DRAW: i32 = 2;
pub const TB_CURSED_WIN: i32 = 3;
pub const TB_WIN: i32 = 4;

pub const TB_RESULT_FAILED: u32 = 0xFFFF_FFFF;

/// Highest piece count for which a tablebase is actually loaded (0 = none).
static LARGEST: OnceLock<u32> = OnceLock::new();

/// The path that actually initialised Fathom (Fathom has no "unload", so the
/// first successful `tb_init` is final for the process lifetime).
static LOADED_PATH: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Path that Fathom was initialised from, if any.  Lets the UCI layer tell the
/// user when a later `SyzygyPath` cannot take effect.
pub fn loaded_path() -> Option<String> {
    LOADED_PATH.lock().ok().and_then(|p| p.clone())
}

#[cfg(corvus_fathom)]
mod ffi {
    use std::os::raw::{c_char, c_uint};

    extern "C" {
        pub fn tb_init(path: *const c_char) -> bool;
        pub fn tb_probe_wdl_impl(
            white: u64,
            black: u64,
            kings: u64,
            queens: u64,
            rooks: u64,
            bishops: u64,
            knights: u64,
            pawns: u64,
            ep: c_uint,
            turn: bool,
        ) -> c_uint;
        pub fn tb_probe_root_impl(
            white: u64,
            black: u64,
            kings: u64,
            queens: u64,
            rooks: u64,
            bishops: u64,
            knights: u64,
            pawns: u64,
            rule50: c_uint,
            ep: c_uint,
            turn: bool,
            results: *mut c_uint,
        ) -> c_uint;
        pub static TB_LARGEST: c_uint;
    }
}

/// Directories that are searched for tablebases when the user has not set
/// `SyzygyPath`, in priority order.  The environment variable wins so a user
/// can point the engine at any location, then the obvious local candidates.
fn default_dirs() -> Vec<String> {
    let mut v = Vec::new();
    if let Some(p) = std::env::var("CORVUS_SYZYGY_PATH")
        .ok()
        .filter(|s| !s.trim().is_empty())
    {
        v.push(p);
    }
    v.push("syzygy".to_string());
    v.push("./syzygy".to_string());
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join("syzygy").to_string_lossy().into_owned());
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let mut p = std::path::PathBuf::from(home);
        p.push(".local/share/corvus/syzygy");
        v.push(p.to_string_lossy().into_owned());
    }
    v
}

/// True when `dir` (a `:`/`;`-separated list of directories) actually holds at
/// least one `.rtbw` file.  Fathom accepts an existing-but-empty directory, so
/// without this check a stray empty `syzygy/` folder would look like a
/// successful load while probing nothing.
fn has_tables(path: &str) -> bool {
    path.split([':', ';'])
        .filter(|d| !d.trim().is_empty())
        .any(|d| {
            let d = std::path::Path::new(d.trim());
            if !d.is_dir() {
                return false;
            }
            match std::fs::read_dir(d) {
                Ok(entries) => entries.flatten().any(|e| {
                    e.path()
                        .extension()
                        .map(|x| x.eq_ignore_ascii_case("rtbw"))
                        .unwrap_or(false)
                }),
                Err(_) => false,
            }
        })
}

/// Find a usable tablebase directory when the user did not configure one.
/// Returns `None` when nothing is present, so the engine behaves exactly as
/// before on a machine without tablebases (e.g. CI).
pub fn discover_default() -> Option<String> {
    default_dirs()
        .into_iter()
        .find(|d| has_tables(d))
        .filter(|d| init(d) > 0)
}

/// Initialise the tablebases from a `SyzygyPath`-style string (`:`/`;`
/// separated, as Fathom expects).  Returns the largest supported piece count,
/// 0 when nothing could be loaded.  Safe to call repeatedly; only the first
/// call takes effect (Fathom keeps global state).
pub fn init(path: &str) -> u32 {
    if path.trim().is_empty() {
        return 0;
    }
    let largest = *LARGEST.get_or_init(|| {
        #[cfg(corvus_fathom)]
        {
            // Fathom wants a comma- or semicolon-separated list; normalise the
            // UCI convention (`:`) to what it parses.
            let normalised = path.replace(':', ";");
            let c = match std::ffi::CString::new(normalised) {
                Ok(c) => c,
                Err(_) => return 0,
            };
            // SAFETY: `tb_init` copies the path and owns its own tables; the
            // string outlives the call.  Fathom's globals are only touched here
            // and in `probe_*`, which are read-only after init.
            let ok = unsafe { ffi::tb_init(c.as_ptr()) };
            if ok {
                if let Ok(mut p) = LOADED_PATH.lock() {
                    *p = Some(path.to_string());
                }
                unsafe { ffi::TB_LARGEST as u32 }
            } else {
                0
            }
        }
        #[cfg(not(corvus_fathom))]
        {
            0
        }
    });
    largest
}

/// Largest piece count the loaded tablebases cover (0 when none are loaded).
pub fn max_pieces() -> u32 {
    *LARGEST.get_or_init(|| 0)
}

/// Encode the board into Fathom's (white, black, kings, queens, rooks,
/// bishops, knights, pawns) bitboard layout.
#[cfg(corvus_fathom)]
fn encode(b: &Board) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
    let white = b.occ[0];
    let black = b.occ[1];
    let kings = b.pieces[0][KING] | b.pieces[1][KING];
    let queens = b.pieces[0][QUEEN] | b.pieces[1][QUEEN];
    let rooks = b.pieces[0][ROOK] | b.pieces[1][ROOK];
    let bishops = b.pieces[0][BISHOP] | b.pieces[1][BISHOP];
    let knights = b.pieces[0][KNIGHT] | b.pieces[1][KNIGHT];
    let pawns = b.pieces[0][PAWN] | b.pieces[1][PAWN];
    (white, black, kings, queens, rooks, bishops, knights, pawns)
}

/// True when the position is a legal tablebase candidate: within both the
/// loaded piece limit *and* the user's `SyzygyProbeLimit`, with no castling
/// rights (Fathom rejects positions that still have them).
fn probeable(b: &Board, probe_limit: u32) -> bool {
    if b.castle != 0 {
        return false;
    }
    let loaded = max_pieces();
    if loaded == 0 || probe_limit == 0 {
        return false;
    }
    let count = b.all.count_ones();
    count <= loaded.min(probe_limit)
}

/// In-tree WDL probe.  Returns the Fathom WDL code (`TB_WIN`..`TB_LOSS`), or
/// `None` when the position is not in the tablebases.
pub fn probe_wdl(b: &Board, probe_limit: u32) -> Option<i32> {
    if !probeable(b, probe_limit) {
        return None;
    }
    #[cfg(corvus_fathom)]
    {
        let (white, black, kings, queens, rooks, bishops, knights, pawns) = encode(b);
        let ep = match b.ep {
            Some(sq) => sq as u32,
            None => 0,
        };
        let turn = b.side == WHITE;
        // SAFETY: pure read of Fathom's loaded tables; all inputs are plain
        // values describing a legal positional encoding.
        let r = unsafe {
            ffi::tb_probe_wdl_impl(
                white, black, kings, queens, rooks, bishops, knights, pawns, ep, turn,
            )
        };
        if r == TB_RESULT_FAILED {
            None
        } else {
            Some(r as i32)
        }
    }
    #[cfg(not(corvus_fathom))]
    {
        None
    }
}

/// Root probe: best move plus DTZ.  `use_50move` mirrors the `Syzygy50MoveRule`
/// option — when false, cursed wins and blessed losses are reported as wins and
/// losses so the engine keeps pressing instead of accepting a draw.
pub fn probe_root(b: &Board, probe_limit: u32, use_50move: bool) -> Option<(Move, i32, u32)> {
    if !probeable(b, probe_limit) {
        return None;
    }
    #[cfg(corvus_fathom)]
    {
        let (white, black, kings, queens, rooks, bishops, knights, pawns) = encode(b);
        let ep = match b.ep {
            Some(sq) => sq as u32,
            None => 0,
        };
        let turn = b.side == WHITE;
        let mut results = [0u32; 193];
        // SAFETY: `results` is TB_MAX_MOVES + 1 entries, the size Fathom writes
        // to for a root probe; inputs are plain values.
        let r = unsafe {
            ffi::tb_probe_root_impl(
                white,
                black,
                kings,
                queens,
                rooks,
                bishops,
                knights,
                pawns,
                b.half,
                ep,
                turn,
                results.as_mut_ptr(),
            )
        };
        if r == TB_RESULT_FAILED {
            return None;
        }
        let wdl = (r & 0x0F) as i32;
        let to = ((r & 0x3F0) >> 4) as usize;
        let from = ((r & 0xFC00) >> 10) as usize;
        let promo_code = ((r & 0x70000) >> 16) as usize;
        // DTZ is a 12-bit *signed* field (two's complement) in Fathom's encoding:
        // losing sides get a negative distance-to-zero, so sign-extend it rather
        // than reporting a bogus ~4095.  Only used for reporting / tie-breaking.
        let dtz_raw = (r & 0xFFF0_0000) >> 20;
        let dtz = (((dtz_raw << 20) as i32) >> 20) as u32;

        // Fathom reports promotions as TB_PROMOTES_{QUEEN,ROOK,BISHOP,KNIGHT}
        // = 1..4; Corvus uses its own piece indices.
        let promo = match promo_code {
            1 => QUEEN,
            2 => ROOK,
            3 => BISHOP,
            4 => KNIGHT,
            _ => crate::board::NO_PIECE,
        };

        // Recover the moving piece (and any capture) by looking the move up in
        // the generated legal move list — cheaper to trust than to re-derive
        // Corvus' flags from Fathom's encoding.  `generate_legal` wants `&mut`
        // for its legality filter, so work on a scratch copy.
        let mut scratch = b.clone();
        let mut list = crate::movegen::MoveList::new();
        crate::movegen::generate_legal(&mut scratch, &mut list);
        let mut found: Option<Move> = None;
        for &m in list.iter() {
            if m.from == from && m.to == to && m.promo == promo {
                found = Some(m);
                break;
            }
        }
        let m = found?;

        // 50-move handling, mirroring the in-search probe: with the rule
        // enabled a cursed win / blessed loss is only a draw; with it disabled
        // they count as win / loss so the engine keeps pressing.
        let wdl = if use_50move {
            match wdl {
                TB_CURSED_WIN | TB_BLESSED_LOSS => TB_DRAW,
                other => other,
            }
        } else {
            match wdl {
                TB_CURSED_WIN => TB_WIN,
                TB_BLESSED_LOSS => TB_LOSS,
                other => other,
            }
        };
        Some((m, wdl, dtz))
    }
    #[cfg(not(corvus_fathom))]
    {
        let _ = use_50move;
        None
    }
}
