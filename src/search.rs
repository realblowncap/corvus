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
use crate::eval::{evaluate, evaluate_with_accum, MATE, PIECE_VALUES};
use crate::nnue_forward::Accumulator;
use crate::movegen::{
    generate_captures, generate_legal, generate_pseudo, gives_check_fast, MoveList, ScoredMoveList,
};
use crate::uci::outln;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const INF: i32 = MATE + 1000;
pub const DRAW: i32 = 0;

/// Score used for a tablebase win/loss.  High enough to dominate any normal
/// evaluation, but deliberately *below* the mate-reporting window
/// (`MATE - 200`) so a tablebase result is never printed as "mate N": DTZ is
/// not the mate distance.
pub const TB_WIN_SCORE: i32 = MATE - 1000;

const MAX_PLY: usize = 128;
const TT_BOUND_NONE: u8 = 0;
const TT_BOUND_LOWER: u8 = 1;
const TT_BOUND_UPPER: u8 = 2;
const TT_BOUND_EXACT: u8 = 3;

struct TTEntry {
    key: AtomicU64,
    data: AtomicU64,
}

impl TTEntry {
    fn empty() -> TTEntry {
        TTEntry {
            key: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

fn pack_data(mov: u32, bound: u8, depth: i32, score: i32) -> u64 {
    (mov as u64)
        | ((bound as u64) << 24)
        | (((depth.min(0xff)) as u64) << 26)
        | (((score as i64) as u64) << 34)
}

pub struct TT {
    entries: Vec<TTEntry>,
    mask: usize,
    generation: AtomicU32,
}

impl TT {
    pub fn new(size_pow: u32) -> TT {
        let n = 1usize << size_pow;
        TT {
            entries: (0..n).map(|_| TTEntry::empty()).collect(),
            mask: n - 1,
            generation: AtomicU32::new(0),
        }
    }

    pub fn clear(&self) {
        for e in self.entries.iter() {
            e.key.store(0, Ordering::Relaxed);
            e.data.store(0, Ordering::Relaxed);
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    pub fn new_search(&self) {
        let _ = self.generation.fetch_add(1, Ordering::Relaxed);
    }

    pub fn probe(&self, key: u64) -> Option<(u32, i32, i32, u8)> {
        let e = &self.entries[(key as usize) & self.mask];
        // Relaxed loads: concurrent overwrites may yield a torn entry mid-store,
        // which is a benign race in Lazy SMP (the caller discards bad bounds).
        let k = e.key.load(Ordering::Relaxed);
        if k != key {
            return None;
        }
        let data = e.data.load(Ordering::Relaxed);
        let mov = (data & 0xFFFFFF) as u32;
        let bound = ((data >> 24) & 0x3) as u8;
        let depth = ((data >> 26) & 0xFF) as i32;
        let score = ((((data >> 34) as i64) << 34) >> 34) as i32;
        Some((mov, depth, score, bound))
    }

    pub fn store(&self, key: u64, mov: u32, depth: i32, score: i32, bound: u8) {
        if bound == TT_BOUND_NONE {
            return;
        }
        let idx = (key as usize) & self.mask;
        let e = &self.entries[idx];
        // Depth-preferred replacement: keep strictly deeper entries so that a
        // shallow search later in the tree cannot evict deeper information.
        let old_key = e.key.load(Ordering::Relaxed);
        if old_key != 0 {
            let old_data = e.data.load(Ordering::Relaxed);
            let old_depth = ((old_data >> 26) & 0xFF) as i32;
            if old_depth > depth {
                return;
            }
        }
        // Write data first, then key: a probe matching the new key reads the
        // fresh data (or a torn-but-discardable mix under a concurrent store).
        e.data.store(pack_data(mov, bound, depth, score), Ordering::Relaxed);
        e.key.store(key, Ordering::Relaxed);
    }

    /// Permille of the table in use, in Stockfish's `hashfull` sense: sample
    /// the first 1000 buckets (or the whole table if smaller) and count the
    /// entries carrying a key.
    pub fn hashfull(&self) -> u64 {
        let n = self.entries.len().min(1000);
        if n == 0 {
            return 0;
        }
        let used = self.entries[..n]
            .iter()
            .filter(|e| e.key.load(Ordering::Relaxed) != 0)
            .count();
        (used as u64 * 1000) / n as u64
    }
}

/// Simple direct-mapped memoization of the static evaluation.
///
/// `evaluate` is a pure function of the board, and its zobrist key uniquely
/// identifies the position, so the result can be reused across iterations and
/// transpositions. The evaluation dominates CPU (it runs at nearly every
/// qsearch node), so this memoization is a pure nps win: node counts and
/// scores are bit-identical to an uncached search.
struct EvalHat {
    keys: Box<[u64]>,
    /// Halfmove clock that each cached value was computed with.  The static
    /// evaluation is *not* a function of the position alone: Stockfish damps
    /// it linearly as the 50-move counter runs up (`v - v*half/199`), so the
    /// same piece placement reached with a different halfmove clock has a
    /// different value.  Keying the cache on the position alone would return
    /// a stale, undamped (or wrongly damped) score.  `u32::MAX` marks an
    /// empty slot.
    halves: Box<[u32]>,
    vals: Box<[i32]>,
    mask: usize,
}

impl EvalHat {
    fn new(slots: usize) -> EvalHat {
        let n = slots.next_power_of_two();
        EvalHat {
            keys: vec![0u64; n].into_boxed_slice(),
            halves: vec![u32::MAX; n].into_boxed_slice(),
            vals: vec![0i32; n].into_boxed_slice(),
            mask: n - 1,
        }
    }

    #[inline(always)]
    fn probe(&self, key: u64, half: u32) -> Option<i32> {
        let i = key as usize & self.mask;
        if self.keys[i] == key && self.halves[i] == half {
            Some(self.vals[i])
        } else {
            None
        }
    }

    #[inline(always)]
    fn insert(&mut self, key: u64, half: u32, val: i32) {
        let i = key as usize & self.mask;
        self.keys[i] = key;
        self.halves[i] = half;
        self.vals[i] = val;
    }
}

/// Per-`think` heap arena for the recursive search's move lists. The lists
/// (2-3 KB each) used to live on the stack inside `negamax`/`quiescence`
/// frames; at 100+ plies of recursion that overflowed debug-mode 2 MB stacks.
/// Lists here are indexed by search-call depth (`lvl`), so frames only carry
/// an index. A separate arena per search thread keeps Lazy SMP allocation-free.
struct MoveArena {
    raw: Vec<MoveList>,
    scored: Vec<ScoredMoveList>,
    qraw: Vec<[MoveList; 2]>,
    qscored: Vec<ScoredMoveList>,
}

impl MoveArena {
    fn new() -> MoveArena {
        MoveArena {
            raw: Vec::new(),
            scored: Vec::new(),
            qraw: Vec::new(),
            qscored: Vec::new(),
        }
    }

    fn reserve(&mut self, lvl: usize) {
        while self.raw.len() <= lvl {
            self.raw.push(MoveList::new());
            self.scored.push(ScoredMoveList::new());
            // qsearch may hold a captures list and a filtered all-moves list
            // at the same time (the sub-8-ply checking-quiets branch).
            self.qraw.push([MoveList::new(), MoveList::new()]);
            self.qscored.push(ScoredMoveList::new());
        }
    }
}

#[derive(Clone)]
pub struct SearchLimits {
    pub depth: u32,
    pub nodes: u64,
    pub movetime: u64,
    pub wtime: u64,
    pub btime: u64,
    pub winc: u64,
    pub binc: u64,
    pub movestogo: u64,
    pub mate: u32,
    pub ponder: bool,
    pub infinite: bool,
    pub searchmoves: Vec<u32>,
    /// GUI/OS latency the GUI doesn't credit us for; reserved from the clock
    /// budget so the deadline lands inside the remaining time.
    pub move_overhead: u64,
    /// Syzygy tablebase configuration, copied from the UCI options.
    pub syzygy_path: String,
    pub syzygy_probe_depth: i32,
    pub syzygy_probe_limit: u32,
    pub syzygy_50_move_rule: bool,
}

impl Default for SearchLimits {
    fn default() -> Self {
        SearchLimits {
            depth: 0,
            nodes: 0,
            movetime: 0,
            wtime: 0,
            btime: 0,
            winc: 0,
            binc: 0,
            movestogo: 0,
            mate: 0,
            ponder: false,
            infinite: false,
            searchmoves: Vec::new(),
            move_overhead: 10,
            syzygy_path: String::new(),
            syzygy_probe_depth: 1,
            syzygy_probe_limit: 7,
            syzygy_50_move_rule: true,
        }
    }
}

/// One root line: the move, its score from the side to move's perspective,
/// the selective depth reached while searching it, and the principal
/// variation. `bound` is 0 when the score is exact, or lower/upper when the
/// search stopped on a bound (used for UCI's `lowerbound`/`upperbound`).
pub struct PvLine {
    pub m: Move,
    pub score: i32,
    pub pv: Vec<Move>,
    pub bound: u8,
}

#[derive(Clone, Copy, PartialEq)]
pub struct SearchResult {
    pub best: Option<Move>,
    pub score: i32,
    pub nodes: u64,
    pub depth: u32,
    pub ponder: Option<Move>,
}

/// Syzygy probing configuration, copied into the searcher at `think` time.
#[derive(Clone, Copy)]
pub struct TbConfig {
    /// Whether any tablebases are loaded at all (`SyzygyPath` was usable).
    pub enabled: bool,
    /// Do not probe shallower than this depth (`SyzygyProbeDepth`).
    pub probe_depth: i32,
    /// Maximum piece count to probe; 0 disables probing (`SyzygyProbeLimit`).
    pub probe_limit: u32,
    /// Honour the 50-move rule when interpreting cursed wins (`Syzygy50MoveRule`).
    pub use_50move: bool,
}

impl Default for TbConfig {
    fn default() -> Self {
        TbConfig {
            enabled: false,
            probe_depth: 1,
            probe_limit: 7,
            use_50move: true,
        }
    }
}

pub struct Searcher {
    pub board: Board,
    pub tt: Arc<TT>,
    pub killers: [[Move; 2]; MAX_PLY],
    pub history: [[i32; 4096]; 2],
    pub counter: [[u32; 64]; 2],
    pub counter_move: [[u32; 64]; 12 * 64], // not used
    pub capture_history: [[i32; 64 * 64]; 12],
    pub continuation: Box<[[[i32; 64]; 64]; 12]>, // [prev_piece*64 + prev_to][piece][to] style rough
    pub moves_made: Vec<Move>,
    pub nodes: u64,
    /// Successful Syzygy probes (root + in-search), reported as `tbhits`.
    pub tb_hits: u64,


    pub stopped: bool,
    pub quiet: bool,
    pub dbg_root: bool,
    pub abort: Arc<AtomicBool>,
    pub deadline: Option<Instant>,
    pub nodes_limit: u64,
    pub key_stack: Vec<u64>,
    pub irr_boundary: Vec<bool>,
    /// How many irreversible moves the current search path contains.  The
    /// game history in front of the root is only relevant while this is
    /// zero, so it is kept as a counter instead of being recomputed.
    irr_count: usize,
    pub null_ep: Vec<Option<usize>>,
    pub tb_conf: TbConfig,
    pub eval_cache_key: u64,
    pub eval_cache_val: i32,
    /// Halfmove clock the cached eval was computed with; the static eval
    /// depends on it, so a mismatch invalidates `eval_cache_val`.
    pub eval_cache_half: u32,
    eval_hat: EvalHat,
    pub eval_noise: i32,
    pub shared_depth: Option<Arc<AtomicU32>>,
    /// Per-game opening-book decorrelator; 0 keeps searches deterministic.
    pub book_seed: u64,
    /// Consult the internal opening book before searching. Off by default so
    /// analysis, tests and benches always search; the UCI layer enables it for
    /// time-controlled game moves (gated by the `OwnBook` option).
    pub use_book: bool,
    /// Deepest ply (incl. quiescence) reached in the current `think` — the
    /// selective depth reported next to the nominal iterative-deepening depth.
    pub seldepth: usize,
    pub multi_pv: usize,
    pub skill_level: i32,
    pub limit_strength: bool,
    pub uci_elo: i32,
    pub show_wdl: bool,
    pub move_overhead: u64,
    pub nodestime: u64,
    /// NNUE accumulator stack: one entry per ply, mirroring the board's
    /// move stack.  `acc_stack[0]` belongs to the root position, the last
    /// entry to the position currently being searched.  Each entry is derived
    /// from its parent by patching the changed feature rows instead of
    /// re-accumulating the whole network.
    acc_stack: Vec<Accumulator>,
    /// The network the accumulator stack was built with.  Cached here so the
    /// hot path does not have to take the network lock on every node.
    nnue: Option<std::sync::Arc<crate::nnue::ParsedNetwork>>,
    /// Whether the evaluation may use the incremental accumulator.  Turning
    /// this off forces a full feature rebuild at every node; it exists so
    /// that the two paths can be compared against each other.
    pub use_accumulator: bool,
}

impl Searcher {
    pub fn new(board: Board) -> Searcher {
        Self::with_tt(board, Arc::new(TT::new(24)))
    }

    pub fn with_tt(board: Board, tt: Arc<TT>) -> Searcher {
        Searcher {
            board,
            tt,
            killers: [[Move::normal(0, 0, 0, NO_PIECE); 2]; MAX_PLY],
            history: [[0; 4096]; 2],
            counter: [[0; 64]; 2],
            counter_move: [[0; 64]; 12 * 64],
            capture_history: [[0; 64 * 64]; 12],
            continuation: Box::new([[[0i32; 64]; 64]; 12]),
            moves_made: Vec::new(),
            nodes: 0,
            tb_hits: 0,


            stopped: false,
            quiet: false,
            dbg_root: false,
            abort: Arc::new(AtomicBool::new(false)),
            deadline: None,
            nodes_limit: u64::MAX,
            key_stack: Vec::new(),
            irr_boundary: Vec::new(),
            irr_count: 0,
            null_ep: Vec::new(),
            tb_conf: TbConfig::default(),
            eval_cache_key: u64::MAX,
            eval_cache_val: 0,
            eval_cache_half: u32::MAX,
            eval_hat: EvalHat::new(1 << 20),
            eval_noise: 0,
            shared_depth: None,
            book_seed: 0,
            use_book: false,
            seldepth: 0,
            multi_pv: 1,
            skill_level: 20,
            limit_strength: false,
            uci_elo: 1320,
            show_wdl: false,
            move_overhead: 10,
            nodestime: 0,
            acc_stack: Vec::new(),
            nnue: None,
            use_accumulator: true,
        }
    }

    // -----------------------------------------------------------------------
    // NNUE accumulator stack
    // -----------------------------------------------------------------------

    /// Rebuild the root accumulator from the current board position.
    ///
    /// Called whenever the search (re-)starts from the root: the accumulator
    /// stack is then empty and the root entry describes the root position.
    fn acc_reset_root(&mut self) {
        self.acc_stack.clear();
        let net = match crate::eval::nnue_net() {
            Some(n) => n,
            None => {
                self.nnue = None;
                return;
            }
        };
        let mut acc = Accumulator::new();
        acc.refresh(&net, &self.board);
        self.acc_stack.push(acc);
        self.nnue = Some(net);
    }

    /// Push the accumulator of the position that was just reached by a move.
    ///
    /// The board's dirty state describes exactly which feature rows changed,
    /// so the child accumulator is a patched copy of the parent's.  A king
    /// move (or a threat list that overflowed) forces a full rebuild of that
    /// perspective, because every feature index depends on the king square.
    fn acc_push(&mut self) {
        let n = self.acc_stack.len();
        if n == 0 {
            return;
        }
        let net = match self.nnue.as_deref() {
            Some(n) => n,
            None => return,
        };
        let dirty = self.board.last_dirty;
        let mut child = Accumulator::new();
        {
            let parent = &self.acc_stack[n - 1];
            child.apply(parent, net, &dirty, &self.board);
        }
        self.acc_stack.push(child);
    }

    /// Push the accumulator of a null-move position.
    ///
    /// A null move does not change the piece placement, so no feature row
    /// changes at all -- only the zobrist key (side to move and ep square).
    fn acc_push_null(&mut self) {
        let n = self.acc_stack.len();
        if n == 0 {
            return;
        }
        let mut acc = self.acc_stack[n - 1].clone();
        acc.key = self.board.key;
        self.acc_stack.push(acc);
    }

    /// Make sure the accumulator stack describes the current position.
    ///
    /// Called at the root of a search: if the stack does not describe the
    /// board (a fresh searcher, a position that was set up directly), it is
    /// rebuilt from scratch.
    fn acc_ensure_root(&mut self) {
        let ok = match self.acc_stack.last() {
            Some(a) => a.key == self.board.key,
            None => false,
        };
        if !ok {
            self.acc_reset_root();
        }
    }

    fn acc_pop(&mut self) {
        if self.acc_stack.len() > 1 {
            self.acc_stack.pop();
        }
    }

    /// Evaluate the current position, memoizing the result per zobrist key so
    /// that repeated static-eval calls at the same node are computed once.
    /// The halfmove clock is part of the cache key: the evaluation is damped
    /// linearly as the 50-move counter runs up, so the same piece placement
    /// reached with a different clock is a *different* evaluation.
    /// A per-thread `eval_noise` is added for Lazy SMP search diversity.
    fn eval_now(&mut self) -> i32 {
        let half = self.board.half;
        if self.eval_cache_key == self.board.key && self.eval_cache_half == half {
            return self.eval_cache_val + self.eval_noise;
        }
        let v = match self.eval_hat.probe(self.board.key, half) {
            Some(h) => h,
            None => {
                let w = self.eval_accumulated();
                self.eval_hat.insert(self.board.key, half, w);
                w
            }
        };
        self.eval_cache_key = self.board.key;
        self.eval_cache_half = half;
        self.eval_cache_val = v;
        v + self.eval_noise
    }

    /// Static evaluation of the current position, reusing the incrementally
    /// maintained NNUE accumulator when it belongs to this position.
    ///
    /// The accumulator saves re-accumulating every feature of the network for
    /// every node; it is only valid when its zobrist key matches the board,
    /// otherwise the full rebuild is used.
    fn eval_accumulated(&mut self) -> i32 {
        match self.nnue.as_deref() {
            Some(net) => {
                let acc = match self.acc_stack.last() {
                    Some(a) if self.use_accumulator && a.key == self.board.key => Some(a),
                    _ => None,
                };
                evaluate_with_accum(net, &self.board, acc)
            }
            None => evaluate(&self.board),
        }
    }

    /// Noise-free static evaluation, for paths whose *value* can escape this
    /// node: prune/fail-high returns and qsearch stand-pat cutoffs.  Helpers
    /// still diverge through ordering and root selection, but they must not
    /// store bounds derived from `eval_noise` into the table the lead thread
    /// then trusts (that leaks a per-thread ±10cp error into every shared
    /// cutoff).
    fn eval_clean(&mut self) -> i32 {
        self.eval_now() - self.eval_noise
    }

    /// Lazy SMP: run `threads` searches on the same root, sharing `tt`.
    /// Helpers run quietly with small eval noise for diversity; the result is
    /// the deepest completed iteration (ties keep the lead thread's result).
    /// The table is passed in so the GUI layer can persist it across moves —
    /// rebuilding a 256 MiB table on every `go` threw away all move-ordering
    /// and cutoff information learned earlier in the game.
    pub fn think_parallel(
        board: Board,
        limits: SearchLimits,
        threads: usize,
        quiet: bool,
        abort: Arc<AtomicBool>,
        book_seed: u64,
        use_book: bool,
        tt: Arc<TT>,
    ) -> SearchResult {
        let shared = Arc::new(AtomicU32::new(0));
        let n = threads.max(1);
        let mut handles = Vec::with_capacity(n.saturating_sub(1));
        for i in 0..n.saturating_sub(1) {
            let mut s = Searcher::with_tt(board.clone(), tt.clone());
            s.abort = abort.clone();
            s.quiet = true;
            s.shared_depth = Some(shared.clone());
            s.book_seed = book_seed;
            // Helpers never consult the book: the lead thread settles the
            // opening move, and a mid-search thread returning instantly would
            // just add noise to the shared-depth adoption logic.
            s.use_book = false;
            // Small deterministic bias per helper so parallel searches diverge.
            s.eval_noise = ((i as i32 % 5) * 5) - 10; // -10, -5, 0, 5, 10
            let l = limits.clone();
            handles.push(std::thread::spawn(move || s.think(l)));
        }

        let mut lead = Searcher::with_tt(board, tt);
        lead.abort = abort;
        lead.quiet = quiet;
        lead.shared_depth = Some(shared);
        lead.book_seed = book_seed;
        lead.use_book = use_book;

        let mut result = lead.think(limits.clone());
        let mut nodes = result.nodes;
        for h in handles {
            if let Ok(r) = h.join() {
                nodes = nodes.saturating_add(r.nodes);
                // Adopt the deepest completed search if it found a best move.
                // Adopt a helper's search only if it genuinely got *further*
                // than the lead thread's completed iteration.  Ties keep the
                // lead result: helpers search with eval noise, so preferring
                // the last-finishing helper at equal depth injected noise
                // into the played move.
                if r.best.is_some() && r.depth > result.depth {
                    result.best = r.best;
                    result.score = r.score;
                    result.depth = r.depth;
                }
            }
        }
        result.nodes = nodes;
        result
    }

    fn deadline(limits: &SearchLimits, side: usize) -> Option<Instant> {
        if limits.movetime > 0 {
            return Some(Instant::now() + Duration::from_millis(limits.movetime));
        }
        if limits.infinite {
            return None;
        }
        if limits.wtime == 0 && limits.btime == 0 {
            return None;
        }
        let (t, inc) = if side == WHITE {
            (limits.wtime, limits.winc)
        } else {
            (limits.btime, limits.binc)
        };
        if t == 0 {
            return None;
        }
        // Reserve time the GUI doesn't credit us for (message round-trips, OS
        // scheduling): the deadline must land inside the clock, not on it.
        let overhead = limits.move_overhead;
        let budget = (t / 32 + inc * 3 / 4).min(t / 4).saturating_sub(overhead).max(1);
        Some(Instant::now() + Duration::from_millis(budget))
    }

    fn make(&mut self, m: Move) -> bool {
        let irr = m.piece == PAWN || m.captured != NO_PIECE;
        let tracked = self.board.nnue_track;
        self.board.nnue_track = true;
        let legal = self.board.make(m);
        if !legal {
            self.board.unmake(m);
            self.board.nnue_track = tracked;
            return false;
        }
        self.board.nnue_track = tracked;
        self.acc_push();
        self.key_stack.push(self.board.key);
        self.irr_boundary.push(irr);
        if irr {
            self.irr_count += 1;
        }
        self.moves_made.push(m);
        true
    }

    fn make_checked(&mut self, m: Move, was_in_check: bool) -> bool {
        let irr = m.piece == PAWN || m.captured != NO_PIECE;
        let tracked = self.board.nnue_track;
        self.board.nnue_track = true;
        let legal = self.board.make_checked(m, was_in_check);
        if !legal {
            self.board.unmake(m);
            self.board.nnue_track = tracked;
            return false;
        }
        self.board.nnue_track = tracked;
        self.acc_push();
        self.key_stack.push(self.board.key);
        self.irr_boundary.push(irr);
        if irr {
            self.irr_count += 1;
        }
        self.moves_made.push(m);
        true
    }

    fn unmake(&mut self, m: Move) {
        self.moves_made.pop();
        self.key_stack.pop();
        if self.irr_boundary.pop() == Some(true) {
            self.irr_count -= 1;
        }
        self.board.unmake(m);
        self.acc_pop();
    }

    fn make_null(&mut self) {
        self.null_ep.push(self.board.ep);
        connected_to_board::zobrist_ep_clear(&mut self.board);
        self.board.side ^= 1;
        connected_to_board::zobrist_side(&mut self.board);
        self.acc_push_null();
    }

    fn unmake_null(&mut self) {
        connected_to_board::zobrist_side(&mut self.board);
        self.board.side ^= 1;
        connected_to_board::zobrist_ep_restore(&mut self.board, self.null_ep.pop());
        self.acc_pop();
    }

    fn poll_stop(&mut self) {
        if self.stopped {
            return;
        }
        if self.nodes >= self.nodes_limit {
            self.stopped = true;
            return;
        }
        if self.abort.load(Ordering::Relaxed) {
            self.stopped = true;
            return;
        }
        if let Some(d) = self.deadline {
            if Instant::now() >= d {
                self.stopped = true;
            }
        }
    }

    fn rep_occurred(&self) -> bool {
        // `key_stack[i]` is the position *after* the i-th move (pushed post-make),
        // so `key_stack[n-1]` == the current position, and history worth scanning
        // runs from `n-2` down to the last irreversible move at `i` *inclusive*:
        // irreversibility reset the move history, but the position right after
        // that move can still recur and must be counted.
        let n = self.key_stack.len();
        if n < 2 {
            return false;
        }
        let mut i = n;
        while i > 0 {
            i -= 1;
            if self.irr_boundary[i] {
                break;
            }
        }
        let mut j = n - 1;
        loop {
            // Never step below index 0: when the irreversible boundary *is* the
            // last entry (`i == n - 1`) there is nothing to scan, and an
            // unbounded `j -= 1` would wrap to `usize::MAX` and panic.
            if j == 0 {
                break;
            }
            j -= 1;
            if self.key_stack[j] == self.board.key {
                return true;
            }
            if j == i {
                break;
            }
        }

        // The moves played before the root are not on the search path, so a
        // repetition that reaches back into the game has to be looked up in
        // the game's own history.
        self.game_history_repeated()
    }

    /// Whether the current position has occurred before in the game that led
    /// to the root position.
    ///
    /// The game history only holds the positions since the last irreversible
    /// move, so a match means the position really can recur.  It is ignored
    /// as soon as the search itself made an irreversible move: the boundary
    /// of the repetition window then lies inside the search, and everything
    /// before it is irrelevant.
    fn game_history_repeated(&self) -> bool {
        if self.irr_count > 0 {
            return false;
        }
        let hist = &self.board.game_history;
        // The last entry is the root position itself; the search path check
        // already covers the occurrence of the position the search is in.
        if hist.len() < 2 {
            return false;
        }
        let key = self.board.key;
        hist[..hist.len() - 1].iter().any(|k| *k == key)
    }

    fn iter_reset(&mut self) {
        // The board is back at the root position here, so the accumulator
        // stack is rebuilt from it.
        self.acc_reset_root();
        for k in self.killers.iter_mut() {
            k[0] = Move::normal(0, 0, 0, NO_PIECE);
            k[1] = Move::normal(0, 0, 0, NO_PIECE);
        }
        self.key_stack.clear();
        self.irr_boundary.clear();
        self.irr_count = 0;
        self.null_ep.clear();
        self.moves_made.clear();
    }

    /// Score every move into `out` (caller-owned, so we never return the 2 KB
    /// `ScoredMoveList` by value — that copy was the hottest line in the
    /// profile) and sort it best-first.  `out.len` is reset here; only the
    /// first `len` entries are ever written or read.
    fn score_moves_into(
        &self,
        out: &mut ScoredMoveList,
        side: usize,
        moves: &MoveList,
        tt_mov: Option<u32>,
        ply: usize,
        prev_to: Option<usize>,
    ) {
        let p = ply.min(MAX_PLY - 1);
        out.len = 0;
        for i in 0..moves.len {
            let m = moves.as_slice()[i];
            let mut s = 0i32;
            if let Some(tm) = tt_mov {
                if tm == m.pack() {
                    s += 1_000_000;
                }
            }
            if m.promo != NO_PIECE {
                // Queen promotions are always valued highly
                s += 960_000 + PIECE_VALUES[m.promo];
            } else if m.captured != NO_PIECE {
                let att = PIECE_VALUES[m.piece];
                let vic = PIECE_VALUES[m.captured];
                // MVV-LVA: valuable victim first, cheaper attacker breaks ties.
                // A/B'd against SEE-only and MVV-LVA+SEE-on-losing-captures:
                // both SEE variants grew the d18 tree by 25-32% on this engine,
                // so ordering stays SEE-free here (qsearch still SEE-prunes).
                let mvv = 16 * vic - att;
                if att > vic {
                    // Likely losing capture: try after killers, before quiets.
                    s += 800_000 + mvv;
                } else {
                    s += 930_000 + mvv;
                }
            } else if self.killers[p][0] == m {
                s += 910_000;
            } else if self.killers[p][1] == m {
                s += 905_000;
            } else if let Some(pt) = prev_to {
                if self.counter[side][pt] != 0 && Move::unpack(self.counter[side][pt]) == m {
                    s += 900_000;
                } else {
                    s += self.history[side][m.from * 64 + m.to] / 4;
                }
            } else {
                s += self.history[side][m.from * 64 + m.to] / 4;
            }
            out.push(m, s);
        }
        out.sort_desc();
    }

    fn store_killer(&mut self, m: Move, ply: usize, side: usize) {
        if m.captured != NO_PIECE || m.promo != NO_PIECE {
            return;
        }
        let p = ply.min(MAX_PLY - 1);
        if self.killers[p][0] != m {
            self.killers[p][1] = self.killers[p][0];
            self.killers[p][0] = m;
        }
        if let Some(&pm) = self.moves_made.last() {
            self.counter[side][pm.to] = m.pack();
        }
        let ks = m.from * 64 + m.to;
        let h = &mut self.history[side][ks];
        *h += 64;
        if *h > 100_000 {
            *h = 100_000;
        }
    }

    pub fn think(&mut self, limits: SearchLimits) -> SearchResult {
        self.tb_conf = TbConfig {
            enabled: !limits.syzygy_path.trim().is_empty() && crate::syzygy::max_pieces() > 0,
            probe_depth: limits.syzygy_probe_depth.max(0),
            probe_limit: limits.syzygy_probe_limit,
            use_50move: limits.syzygy_50_move_rule,
        };
        self.deadline = Searcher::deadline(&limits, self.board.side);
        self.nodes_limit = if limits.nodes > 0 { limits.nodes } else { u64::MAX };
        let max_depth = if limits.depth > 0 { limits.depth as usize } else { 128 };
        self.seldepth = 0;
        let start = Instant::now();

        let mut root_moves = MoveList::new();
        generate_legal(&mut self.board, &mut root_moves);
        if root_moves.is_empty() {
            return SearchResult {
                best: None,
                score: if self.board.in_check(self.board.side) {
                    -MATE + 1
                } else {
                    DRAW
                },
                nodes: 0,
                depth: 0,
                ponder: None,
            };
        }

        if self.use_book {
            if let Some(bm) = crate::book::probe(&mut self.board, root_moves.as_slice(), self.book_seed) {
                let sc = evaluate(&self.board);
                if !self.quiet {
                    outln(format!("info depth 1 seldepth 1 score cp {} pv {}", sc, bm.uci()));
                }
                return SearchResult {
                    best: Some(bm),
                    score: if self.board.side == BLACK { -sc } else { sc },
                    nodes: 0,
                    depth: 0,
                    ponder: None,
                };
            }
        }

        // Root tablebase probe.  Only a *decisive* result short-circuits the
        // search: for a draw, normal search still chooses among the drawing
        // moves (which is both stronger and avoids TB-shuffle draws).  Skipped
        // when the GUI restricted the move list.
        if limits.searchmoves.is_empty() && !limits.syzygy_path.trim().is_empty() {
            if let Some((tb_move, wdl, dtz)) = crate::syzygy::probe_root(
                &self.board,
                limits.syzygy_probe_limit,
                limits.syzygy_50_move_rule,
            ) {
                self.tb_hits += 1;
                if wdl == crate::syzygy::TB_WIN || wdl == crate::syzygy::TB_LOSS {
                    // Report a large cp advantage rather than a mate: DTZ counts
                    // plies to the next zeroing move (pawn move, capture, or
                    // mate), which is *not* the mate distance, and converting it
                    // into `MATE - n` would make the GUI print a fake "mate N"
                    // line for positions thousands of plies from mate.  Staying
                    // below the mate window also keeps `score_string` honest.
                    let tb_score = if wdl == crate::syzygy::TB_WIN {
                        TB_WIN_SCORE
                    } else {
                        -TB_WIN_SCORE
                    };
                    if !self.quiet {
                        outln(format!(
                            "info depth 1 seldepth 1 score cp {} pv {}",
                            if self.board.side == BLACK { -tb_score } else { tb_score },
                            tb_move.uci()
                        ));
                        outln(format!("info string tb root wdl {} dtz {}", wdl, dtz));
                    }
                    return SearchResult {
                        best: Some(tb_move),
                        score: if self.board.side == BLACK { -tb_score } else { tb_score },
                        nodes: self.nodes,
                        depth: 0,
                        ponder: None,
                    };
                }
            }
        }

        let mut best: Option<Move> = None;
        let mut score = 0i32;
        let mut last_depth = 0u32;
        let mut depth = 1usize;
        let mut prev = 0i32;
        let mut arena = MoveArena::new();

        loop {
            if self.stopped {
                break;
            }
            self.iter_reset();

            // Lazy SMP: adopt the deepest completed iteration from any thread.
            // The shared TT supplies the PV and move ordering, so jumping depth
            // immediately is far cheaper than re-searching those plies alone.
            if let Some(sd) = &self.shared_depth {
                let peer = sd.load(Ordering::Relaxed) as usize;
                if peer > depth && peer < max_depth {
                    depth = peer;
                }
            }

            let prev_best = best;
            // Skill weakening if set
            let (mv, sc) = if depth >= 5 && prev > -900 && prev < 900 {
                let mut delta = 26i32;
                let mut out: Option<(Move, i32)> = None;
                while out.is_none() {
                    let alpha = prev - delta;
                    let beta = prev + delta;
                    let (mm, ss) = self.search_root_arena(&mut arena, depth as i32, alpha, beta);
                    // An aborted iteration yields no move: propagate `None` out
                    // of the aspiration loop instead of spinning on it.
                    let mm = match mm {
                        Some(m) => m,
                        None => break,
                    };
                    if self.stopped || (ss > alpha && ss < beta) {
                        out = Some((mm, ss));
                    } else {
                        delta += delta / 2 + 1;
                        if delta > 300 {
                            let (fm, fs) =
                                self.search_root_arena(&mut arena, depth as i32, -INF, INF);
                            out = fm.map(|m| (m, fs));
                        }
                    }
                }
                match out {
                    Some((m, s)) => (Some(m), s),
                    None => (None, prev),
                }
            } else {
                self.search_root_arena(&mut arena, depth as i32, -INF, INF)
            };
            if self.stopped {
                break;
            }
            // A partial iteration searched no move: keep the previous result.
            let mv = match mv {
                Some(m) => m,
                None => break,
            };
            // Age history table to focus on recent search results
            for side in 0..2 {
                for h in self.history[side].iter_mut() {
                    *h >>= 1;
                }
            }
            prev = sc;
            best = Some(mv);
            score = if self.board.side == BLACK { -sc } else { sc };
            last_depth = depth as u32;

            // Publish our completed depth so other threads can skip ahead.
            if let Some(sd) = &self.shared_depth {
                sd.fetch_max(depth as u32, Ordering::Relaxed);
            }

            let elapsed = start.elapsed().as_millis() as u64;
            let nps = if elapsed > 0 {
                self.nodes * 1000 / elapsed
            } else {
                self.nodes
            };
            let pv = self.pv_line(depth);
            let pv_str: Vec<String> = pv.iter().map(|m| m.uci()).collect();
            if !self.quiet {
                let multipv = if self.multi_pv < 1 { 1usize } else { self.multi_pv };
                println!("info depth {} seldepth {} multipv {} {} nodes {} nps {} hashfull {} tbhits {} time {} pv {}", depth, self.seldepth, 1, score_string(score), self.nodes, nps, self.tt.hashfull(), self.tb_hits, elapsed, pv_str.join(" "));
                if multipv > 1 {
                    // best effort: print primary PV multiple times is not enough; for now output primary
                }
            }

            if depth >= max_depth {
                break;
            }
            if let Some(d) = self.deadline {
                if prev_best == Some(mv) && depth >= 6 && score.abs() < MATE - 100 {
                    let total = d.duration_since(start).as_millis() as u64;
                    if elapsed >= total * 3 / 5 {
                        break;
                    }
                }
            }
            let over_time = self.deadline.map_or(false, |d| Instant::now() >= d);
            if over_time {
                break;
            }
            depth += 1;
        }

        if best.is_none() {
            // No iteration ever completed (e.g. the clock expired during depth 1
            // with a slow evaluator). Play something legal and ordered rather
            // than the raw first generated move, which is arbitrary.
            let tt_mov = self.tt.probe(self.board.key).map(|e| e.0);
            let mut ordered = ScoredMoveList::new();
            self.score_moves_into(&mut ordered, self.board.side, &root_moves, tt_mov, 0, None);
            let mut chosen: Option<Move> = None;
            for (m, _) in ordered.iter() {
                let mut probe = self.board.clone();
                let was_in_check = probe.in_check(probe.side);
                if probe.make_checked(m, was_in_check) {
                    chosen = Some(m);
                    break;
                }
            }
            if let Some(m) = chosen {
                best = Some(m);
                score = DRAW;
            }
        }

        SearchResult {
            best,
            score,
            nodes: self.nodes,
            depth: last_depth,
            ponder: None,
        }
    }

    /// Search the root position once. Convenience wrapper that allocates a
    /// fresh move arena; `think` reuses one across iterations for speed.
    pub fn search_root(&mut self, depth: i32, alpha: i32, beta: i32) -> (Option<Move>, i32) {
        let mut arena = MoveArena::new();
        self.search_root_arena(&mut arena, depth, alpha, beta)
    }

    fn search_root_arena(
        &mut self,
        arena: &mut MoveArena,
        depth: i32,
        mut alpha: i32,
        beta: i32,
    ) -> (Option<Move>, i32) {
        let alpha0 = alpha;
        arena.reserve(0);
        // The accumulator stack must describe the position we are searching.
        self.acc_ensure_root();
        let mut list = MoveList::new();
        generate_pseudo(&self.board, &mut list);
        let tt_mov = self.tt.probe(self.board.key).map(|e| e.0);
        let mut ordered = ScoredMoveList::new();
        self.score_moves_into(&mut ordered, self.board.side, &list, tt_mov, 0, None);

        let mut best = -INF;
        // Seed with an actual searched move, never with the ordering-first move:
        // if the clock aborts the iteration before the first `v > best`, a bogus
        // fallback here would become the played move (see the `had_move` guard).
        let mut best_move = Move::normal(0, 0, 0, NO_PIECE);
        let mut fail_high = false;

        // True once some move has been searched to a score in this iteration.
        let mut had_move = false;
        let mut first = true;
        let mut legal_count = 0usize;
        for (m, _) in ordered.iter() {
            if !self.make(m) {
                continue;
            }
            legal_count += 1;
            if self.stopped {
                self.unmake(m);
                break;
            }
            let v = if first {
                -self.negamax(arena, 0, depth, -beta, -alpha, 1, true)
            } else {
                let mut v = -self.negamax(arena, 0, depth, -alpha - 1, -alpha, 1, true);
                if v > alpha && v < beta {
                    v = -self.negamax(arena, 0, depth, -beta, -alpha, 1, true);
                }
                v
            };
            self.unmake(m);
            if self.dbg_root && !self.stopped {
                outln(format!(
                    "rootdbg depth {} move {} window ({},{}) v {}",
                    depth,
                    m.uci(),
                    alpha,
                    beta,
                    v
                ));
            }
            first = false;
            if self.stopped {
                break;
            }
            if !had_move || v > best {
                best = v;
                best_move = m;
                had_move = true;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                fail_high = true;
                break;
            }
        }
        if legal_count == 0 {
            best_move = Move::normal(0, 0, 0, NO_PIECE);
            best = if self.board.in_check(self.board.side) {
                -MATE + 1
            } else {
                DRAW
            };
            had_move = true;
        }
        if !self.stopped {
            let bound = if fail_high {
                TT_BOUND_LOWER
            } else if best <= alpha0 {
                TT_BOUND_UPPER
            } else {
                TT_BOUND_EXACT
            };
            self.tt.store(
                self.board.key,
                best_move.pack(),
                depth,
                probe_score(best, 0),
                bound,
            );
        }
        // An aborted iteration must not declare a move it never searched; the
        // caller uses `None` to mean "keep the previous iteration's move".
        let out = if had_move { Some(best_move) } else { None };
        (out, best)
    }

    fn negamax(
        &mut self,
        arena: &mut MoveArena,
        lvl: usize,
        depth: i32,
        mut alpha: i32,
        mut beta: i32,
        ply: usize,
        allow_null: bool,
    ) -> i32 {
        arena.reserve(lvl);
        if ply > self.seldepth {
            self.seldepth = ply;
        }
        self.nodes += 1;

        // Every 32 nodes, not 1024: with a slow per-node evaluator, a
        // 1024-node gap between clock checks can overrun a short time control
        // by tens of milliseconds per move and eventually forfeit on time.
        if self.nodes & 31 == 0 {
            self.poll_stop();
        }
        if self.stopped {
            return 0;
        }

        if ply > 0 {
            if self.board.half >= 100 || self.rep_occurred() {
                return DRAW;
            }
        }

        // Syzygy WDL probe.  A tablebase result is exact, so it can be returned
        // immediately; only probe at sufficient depth (the option exists to keep
        // the cost down) and inside the loaded piece limit.
        if self.tb_conf.enabled
            && self.tb_conf.probe_limit > 0
            && (depth as i32) >= self.tb_conf.probe_depth
            // Fathom's WDL tables classify a win as cursed/blessed relative to
            // a *fresh* 50-move clock; `tb_probe_wdl_impl` cannot see the real
            // halfmove clock (it is hard-coded to 0 inside Fathom), so probing
            // with a non-zero clock would report a win that is only a cursed
            // win as a full win -- and vice versa.  Stockfish 19 gates the
            // same in-tree probe on `rule50_count() == 0` for this reason.
            && self.board.half == 0
        {
            let tb_probe = crate::syzygy::probe_wdl(&self.board, self.tb_conf.probe_limit);
            if tb_probe.is_some() {
                self.tb_hits += 1;
            }
            let tb_value = match tb_probe {
                // 50-move handling: `Syzygy50MoveRule=true` means a cursed win
                // (win in material, but not within 50 moves) is only a draw, so
                // the engine must not believe it is winning.  With the rule
                // disabled those positions count as real wins/losses.
                Some(crate::syzygy::TB_WIN) => Some(TB_WIN_SCORE),
                Some(crate::syzygy::TB_LOSS) => Some(-TB_WIN_SCORE),
                Some(crate::syzygy::TB_CURSED_WIN) => Some(if self.tb_conf.use_50move {
                    DRAW
                } else {
                    TB_WIN_SCORE
                }),
                Some(crate::syzygy::TB_BLESSED_LOSS) => Some(if self.tb_conf.use_50move {
                    DRAW
                } else {
                    -TB_WIN_SCORE
                }),
                Some(_) => Some(DRAW),
                None => None,
            };
            if let Some(tb_value) = tb_value {
                if tb_value >= beta {
                    return tb_value;
                }
                if tb_value > alpha {
                    alpha = tb_value;
                }
            }
        }

        // Mate distance pruning
        let low = -(MATE - ply as i32 - 1);
        let high = MATE - ply as i32 - 1;
        if alpha < low {
            alpha = low;
        }
        if beta > high {
            beta = high;
        }
        if alpha >= beta {
            return alpha;
        }

        let in_check = self.board.in_check(self.board.side);
        let key = self.board.key;

        let tt = self.tt.probe(key);
        let tt_mov = tt.map(|e| e.0);
        let mate_ctx = tt.map_or(false, |(_tm, _td, ts, _tb)| {
            ts.abs() >= MATE - 2 * MAX_PLY as i32
        });
        if let Some((_tmov, tdepth, tscore, tbound)) = tt {
            if tdepth >= depth && depth > 0 && tscore > -MATE && tscore < MATE {
                let sc = unprobe_score(tscore, ply);
                if tbound == TT_BOUND_EXACT {
                    return sc;
                }
                if tbound == TT_BOUND_LOWER && sc >= beta {
                    return sc;
                }
                if tbound == TT_BOUND_UPPER && sc <= alpha {
                    return sc;
                }
            }
        }

        // Internal Iterative Reduction: when no TT move is available at depth >= 4,
        // reduce by 1 to speed up the search since we lack move ordering information.
        let mut d = depth;
        if tt_mov.is_none() && d >= 4 {
            d -= 1;
        }
        // Check extension: extend when in check at low depth to avoid horizon effect
        if in_check && d <= 6 {
            d += 1;
        }
        if d <= 0 {
            return self.quiescence(arena, lvl + 1, alpha, beta, ply);
        }

        let dd = d;

        generate_pseudo(&self.board, &mut arena.raw[lvl]);

        if !in_check && alpha + 1 == beta && d <= 10 {
            // Noise-free: a fail-high here returns `stand` straight to the
            // parent, so a noised value would be stored as a real TT bound.
            let stand = self.eval_clean();
            let margin = 60 + 60 * d;
            if stand >= beta + margin && stand < MATE - 500 && stand > -MATE + 500 {
                return stand;
            }
        }

        // Razoring
        if !in_check && d <= 3 && !mate_ctx {
            let margin = 200 + 150 * d;
            if self.eval_clean() + margin <= alpha {
                let v = self.quiescence(arena, lvl + 1, alpha, beta, ply);
                if v <= alpha {
                    return v;
                }
            }
        }

        let stm = self.board.side;
        let n = arena.raw[lvl].len;
        let prev_to = self.moves_made.last().map(|m| m.to);
        self.score_moves_into(&mut arena.scored[lvl], stm, &arena.raw[lvl], tt_mov, ply, prev_to);

        if allow_null && depth >= 2 && !in_check && n >= 2 && self.has_null_material(stm) {
            let r = (4 + depth / 5).min(depth - 1);
            self.make_null();
            let v = -self.negamax(arena, lvl + 1, depth - r, -beta, -beta + 1, ply + 1, false);
            self.unmake_null();
            if self.stopped {
                return 0;
            }
            if v >= beta {
                // Verified null move: at depth 2 the reduced search can blind
                // us to cheap tactics, so re-check at full depth before
                // accepting the cutoff.
                if depth == 2 {
                    self.make_null();
                    let vv = -self.negamax(arena, lvl + 1, depth, -beta, -beta + 1, ply + 1, false);
                    self.unmake_null();
                    if self.stopped {
                        return 0;
                    }
                    if vv >= beta {
                        return beta;
                    }
                } else {
                    return beta;
                }
            }
        }

        let alpha0 = alpha;
        let mut best = -INF;
        let mut best_move = Move::normal(0, 0, 0, NO_PIECE);
        let mut searched = 0usize;
        let mut legal_count = 0usize;

        let mut se_move: Option<u32> = None;
        let mut se_ext = 0i32;
        if !in_check && depth >= 5 && ply >= 2 && n >= 2 {
            if let Some((tm, td, ts, tb)) = tt {
                if tm != 0
                    && td >= depth - 4
                    && ts > -MATE && ts < MATE
                    && (tb == TT_BOUND_LOWER || tb == TT_BOUND_EXACT)
                {
                    let standv = self.eval_clean();
                    if standv >= beta - 5 {
                        let margin = 50 + 8 * depth;
                        let sbeta = ts - margin;
                        let mut found_other = false;
                        let n_ordered = arena.scored[lvl].len;
                        for oi in 0..n_ordered {
                            let (om, _) = arena.scored[lvl].get(oi);
                            if om.pack() == tm {
                                continue;
                            }
                            if !self.make(om) {
                                continue;
                            }
                            let v = -self.negamax(arena, lvl + 1, depth / 2, -sbeta, -sbeta + 1, ply + 1, true);
                            self.unmake(om);
                            if self.stopped {
                                return 0;
                            }
                            if v >= sbeta {
                                found_other = true;
                                break;
                            }
                        }
                        if !found_other {
                            se_move = Some(tm);
                            se_ext = 1;
                            if ts >= beta + 100 {
                                se_ext = 2;
                            }
                        }
                    }
                }
            }
        }

        // ProbCut
        if !in_check && depth >= 5 {
            let pc_beta = beta + 200;
            let n_scored_tmp = arena.scored[lvl].len;
            for oi in 0..n_scored_tmp {
                let (m, _) = arena.scored[lvl].get(oi);
                if m.captured == NO_PIECE && m.promo == NO_PIECE {
                    continue;
                }
                if !self.make_checked(m, in_check) {
                    continue;
                }
                let v = -self.quiescence(arena, lvl + 1, -pc_beta, -pc_beta + 1, ply + 1);
                self.unmake(m);
                if self.stopped {
                    break;
                }
                if v >= pc_beta {
                    return pc_beta - 1; // cutoff
                }
            }
        }

        let stand = if !in_check && d <= 3 {
            Some(self.eval_now())
        } else {
            None
        };

        let n_scored = arena.scored[lvl].len;
        for oi in 0..n_scored {
            let (m, _) = arena.scored[lvl].get(oi);
            let quiet = m.captured == NO_PIECE && m.promo == NO_PIECE;
            // Static per-move: the board does not change across the checks
            // below, and `gives_check_fast` is not provably pure to the
            // compiler, so compute it once instead of up to five times.
            let gives_chk = quiet && !in_check && gives_check_fast(&self.board, m);
            if quiet && d <= 3 && searched >= 2 {
                if let Some(st) = stand {
                    let margin = match d {
                        1 => 60,
                        2 => 130,
                        _ => 180,
                    };
                    if st + margin <= alpha && !gives_chk && !mate_ctx {
                        continue;
                    }
                }
            }

            // Late Move Pruning: after many moves at shallow depth, quiet moves
            // are unlikely to beat alpha — skip unless they give check or are
            // the singular-extension move.
            if quiet && !in_check && d <= 5 && searched >= (3 + d * d) as usize && !mate_ctx {
                let is_se = match se_move {
                    Some(sm) => m.pack() == sm,
                    None => false,
                };
                if !is_se && !gives_chk {
                    continue;
                }
            }

            // SEE pruning: at shallow depth a capture that loses material in a
            // forced exchange is almost never best, and searching it costs a
            // whole subtree. The TT move is exempt (it carries the stored PV).
            if !quiet && !in_check && d <= 5 && searched >= 1 && !mate_ctx {
                if tt_mov != Some(m.pack())
                    && m.promo == NO_PIECE
                    && crate::movegen::see(&self.board, m.to, stm) < -40 * d
                {
                    continue;
                }
            }
            let mut reduction = 0i32;
            if quiet && !in_check && d >= 2 && searched >= 2 && !mate_ctx {
                let p = ply.min(MAX_PLY - 1);
                let ks = m.from * 64 + m.to;
                let good = m == self.killers[p][0]
                    || m == self.killers[p][1]
                    || self.history[stm][ks] > 3000;
                if !good {
                    reduction = 1 + (searched as i32 - 3) / 2 + (d - 2) / 3;
                    // History-aware: unproven quiet moves are reduced harder,
                    // favored ones barely at all.
                    let hist = self.history[stm][ks];
                    if hist <= 0 {
                        reduction += 2;
                    } else if hist <= 40 {
                        reduction += 1;
                    } else if hist > 2000 {
                        reduction = reduction.saturating_sub(1);
                    }
                    if alpha + 1 < beta {
                        // PV node: reduce less to avoid missing the PV
                        reduction = reduction * 2 / 3;
                    }
                    if gives_chk {
                        reduction = (reduction * 2) / 3; // Less reduction for checks
                    }
                    reduction = reduction.min(d - 1).max(0);
                }
            }
            let mut v;
            let mut extension = 0i32;
            if let Some(sm) = se_move {
                if m.pack() == sm {
                    extension = se_ext;
                }
            }
            if quiet
                && !in_check
                && d >= 3
                && ply >= 1
                && ply < 14
                && extension == 0
                && gives_chk
            {
                extension = 1;
            }
            // Recapture extension: taking back on the square the opponent just
            // captured on keeps forcing exchanges from falling off the horizon.
            // Only for balances that do not lose material, so a bad recapture
            // is not searched deeper.
            if !quiet
                && !in_check
                && extension == 0
                && m.promo == NO_PIECE
                && prev_to == Some(m.to)
                && d >= 2
                && d <= 10
                && crate::movegen::see(&self.board, m.to, stm) >= 0
            {
                extension = 1;
            }
            let mut fdepth = dd - 1 - reduction + extension;
            if fdepth < 0 {
                fdepth = 0;
            }
            
            // Aggressive futility pruning for quiet moves
            if quiet && fdepth <= 4 && searched >= 2 {
                if let Some(st) = stand {
                    let margin = 120 + 40 * fdepth;
                    if st + margin <= alpha && !gives_chk && !mate_ctx {
                        continue;
                    }
                }
            }
            if !self.make_checked(m, in_check) {
                continue;
            }
            legal_count += 1;
            if searched == 0 {
                v = -self.negamax(arena, lvl + 1, fdepth, -beta, -alpha, ply + 1, true);
            } else {
                v = -self.negamax(arena, lvl + 1, fdepth, -alpha - 1, -alpha, ply + 1, true);
                if v > alpha && v < beta {
                    v = -self.negamax(arena, lvl + 1, dd - 1 + extension, -beta, -alpha, ply + 1, true);
                }
            }
            self.unmake(m);
            searched += 1;
            if self.stopped {
                return 0;
            }
            if quiet && v <= alpha0 {
                let h = &mut self.history[stm][m.from * 64 + m.to];
                *h = h.saturating_sub(40);
            } else if quiet && v >= beta {
                let h = &mut self.history[stm][m.from * 64 + m.to];
                *h = (*h + 32).min(100000);
            }
            if v > best {
                best = v;
                best_move = m;
            }
            if v >= beta {
                if quiet {
                    self.store_killer(m, ply, stm);
                }
                break;
            }
            if v > alpha {
                alpha = v;
            }
        }

        if legal_count == 0 {
            return if in_check { -(MATE - ply as i32) } else { DRAW };
        }

        let bound = if best <= alpha0 {
            TT_BOUND_UPPER
        } else if best >= beta {
            TT_BOUND_LOWER
        } else {
            TT_BOUND_EXACT
        };
        if !self.stopped {
            let to_score = probe_score(best, ply);
            // Store the *effective* depth actually searched (`dd`): the
            // nominal depth overstates after an IIR reduction and
            // understates after a check extension, and a too-high stored
            // depth lets later probes cut off on shallower information.
            self.tt.store(key, best_move.pack(), dd, to_score, bound);
        }
        best
    }

    fn quiescence(
        &mut self,
        arena: &mut MoveArena,
        lvl: usize,
        mut alpha: i32,
        beta: i32,
        ply: usize,
    ) -> i32 {
        if ply >= MAX_PLY {
            return evaluate(&self.board);
        }
        arena.reserve(lvl);
        if ply > self.seldepth {
            self.seldepth = ply;
        }
        self.nodes += 1;
        // See the main search: frequent clock checks bound overrun per move.
        if self.nodes & 31 == 0 {
            self.poll_stop();
        }
        if self.stopped {
            return 0;
        }

        let in_check = self.board.in_check(self.board.side);

        let stand = self.eval_now();
        if !in_check {
            if stand >= beta {
                return stand;
            }
            if stand > alpha {
                alpha = stand;
            }
        }

        // Arena-allocated lists; checking quiets are capped to avoid explosion.
        let mut checks: [Move; 16] = [Move::normal(0, 0, 0, NO_PIECE); 16];
        let mut nchecks = 0usize;
        // Slot 0 holds captures (or every pseudo-legal move when in check);
        // slot 1 holds the filtered quiet list of the checking-quiets branch.
        let caps_slot = if in_check {
            generate_pseudo(&self.board, &mut arena.qraw[lvl][0]);
            0
        } else if ply < 8 {
            generate_pseudo(&self.board, &mut arena.qraw[lvl][1]);
            for &m in arena.qraw[lvl][1].iter() {
                if m.captured == NO_PIECE && m.promo == NO_PIECE {
                    if gives_check_fast(&self.board, m) {
                        checks[nchecks] = m;
                        nchecks += 1;
                        if nchecks == 16 {
                            break;
                        }
                    }
                }
            }
            arena.qraw[lvl][1].retain(|m| m.captured != NO_PIECE || m.promo != NO_PIECE);
            1
        } else {
            generate_captures(&self.board, &mut arena.qraw[lvl][0]);
            0
        };
        let caps_empty = arena.qraw[lvl][caps_slot].is_empty();
        if caps_empty && nchecks == 0 {
            return if in_check { -(MATE - ply as i32) } else { alpha };
        }

        let tt_mov = self.tt.probe(self.board.key).map(|e| e.0);
        self.score_moves_into(
            &mut arena.qscored[lvl],
            self.board.side,
            &arena.qraw[lvl][caps_slot],
            tt_mov,
            ply,
            None,
        );
        let side = self.board.side;
        for i in 0..nchecks {
            let m = checks[i];
            arena.qscored[lvl].push(m, self.history[side][m.from * 64 + m.to] / 16);
        }

        let mut legal_count = 0usize;
        let n_scored = arena.qscored[lvl].len;
        for si in 0..n_scored {
            let (m, _) = arena.qscored[lvl].get(si);
            if !in_check {
                let is_tt = tt_mov == Some(m.pack());
                let cap_v = if m.captured != NO_PIECE {
                    PIECE_VALUES[m.captured]
                } else {
                    0
                } + if m.promo != NO_PIECE {
                    // Promotions gain material too: without this the delta
                    // test used 0 for every promotion and pruned queen
                    // promotions whenever the stand-pat was ~200cp below
                    // alpha — a source of missed (and hung) promotions.
                    PIECE_VALUES[m.promo] - PIECE_VALUES[PAWN]
                } else {
                    0
                };
                // The TT move is never pruned: it carries the stored best line,
                // which a shallow mate/SEE estimate must not discard.
                let mut skip = !is_tt && stand + cap_v + 200 <= alpha;
                if !skip
                    && !is_tt
                    && m.captured != NO_PIECE
                    && m.promo == NO_PIECE
                    && crate::movegen::see(&self.board, m.to, self.board.side) < 0
                {
                    skip = true;
                }
                if skip {
                    let quiet = m.captured == NO_PIECE && m.promo == NO_PIECE;
                    if !quiet || !gives_check_fast(&self.board, m) {
                        continue;
                    }
                }
            }
            if !self.make_checked(m, in_check) {
                continue;
            }
            legal_count += 1;
            let v = -self.quiescence(arena, lvl + 1, -beta, -alpha, ply + 1);
            self.unmake(m);
            if self.stopped {
                return 0;
            }
            if v >= beta {
                return v;
            }
            if v > alpha {
                alpha = v;
            }
        }
        if legal_count == 0 {
            return if in_check { -(MATE - ply as i32) } else { alpha };
        }
        alpha
    }

    fn has_null_material(&self, side: usize) -> bool {
        let major = self.board.pieces[side][QUEEN] | self.board.pieces[side][ROOK];
        let minors = self.board.pieces[side][BISHOP] | self.board.pieces[side][KNIGHT];
        major != 0 || minors.count_ones() >= 2
    }

    pub fn pv_line(&mut self, max: usize) -> Vec<Move> {
        let mut out = Vec::new();
        let mut made = Vec::new();
        for _ in 0..max {
            if self.stopped {
                break;
            }
            let key = self.board.key;
            let tm = match self.tt.probe(key) {
                Some((mov, _, _, _)) => {
                    let m = Move::unpack(mov);
                    if self.is_legal(m) {
                        Some(m)
                    } else {
                        None
                    }
                }
                None => None,
            };
            match tm {
                Some(m) => {
                    if !self.make(m) {
                        break;
                    }
                    out.push(m);
                    made.push(m);
                }
                None => break,
            }
        }
        for m in made.iter().rev() {
            self.unmake(*m);
        }
        out
    }

    fn is_legal(&mut self, m: Move) -> bool {
        let mut list = MoveList::new();
        generate_legal(&mut self.board, &mut list);
        list.contains(&m)
    }
}

fn probe_score(score: i32, ply: usize) -> i32 {
    if score > MATE - 200 {
        score + ply as i32
    } else if score < -(MATE - 200) {
        score - ply as i32
    } else {
        score
    }
}

fn unprobe_score(score: i32, ply: usize) -> i32 {
    if score > MATE - 200 {
        score - ply as i32
    } else if score < -(MATE - 200) {
        score + ply as i32
    } else {
        score
    }
}

fn score_string(score: i32) -> String {
    if score > MATE - 200 {
        let plies = MATE - score;
        let moves = (plies + 1) / 2;
        format!("score mate {}", moves)
    } else if score < -(MATE - 200) {
        let plies = MATE + score;
        let moves = (plies + 1) / 2;
        format!("score mate -{}", moves)
    } else {
        format!("score cp {}", score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_cache_distinguishes_halfmove_clocks() {
        let mut hat = EvalHat::new(1024);
        let key = 0x1234_5678_9ABC_DEF0u64;

        // An empty slot must never hit.
        assert_eq!(hat.probe(key, 0), None);
        assert_eq!(hat.probe(key, 40), None);

        hat.insert(key, 0, 500);
        assert_eq!(hat.probe(key, 0), Some(500));
        // The static evaluation is damped by the halfmove clock, so the same
        // placement reached with a different clock is a *different* number and
        // must not be served from the cache.
        assert_eq!(hat.probe(key, 40), None);

        // Re-inserting with the new clock replaces the entry.
        hat.insert(key, 40, 300);
        assert_eq!(hat.probe(key, 40), Some(300));
        assert_eq!(hat.probe(key, 0), None);
    }

    #[test]
    fn eval_cache_is_invalidated_when_the_halfmove_clock_changes() {
        let b = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let mut s = Searcher::new(b);
        assert_eq!(s.eval_cache_half, u32::MAX);

        let v0 = s.eval_now();
        assert_eq!(s.eval_cache_half, 0);
        assert_eq!(s.eval_cache_key, s.board.key);

        // Same piece placement, different halfmove clock.
        s.board.half = 40;
        let v40 = s.eval_now();
        assert_eq!(s.eval_cache_half, 40, "the memo must be keyed on the clock");
        assert_eq!(s.eval_cache_key, s.board.key);

        // With the neural network in play the damped evaluation differs; with
        // the hand-crafted fallback it does not, so only assert the stronger
        // property when a net is actually loaded.
        if crate::eval::nnue_active() {
            assert_ne!(v0, v40, "rule-50 damping must change the evaluation");
        }
    }

    /// The incrementally maintained accumulator must not change what the
    /// search finds: the same position searched with and without it has to
    /// produce the same score and the same best move.
    #[test]
    fn accumulator_does_not_change_the_search_result() {
        if !crate::eval::nnue_active() {
            return; // no network loaded: both paths use the fallback eval
        }
        const FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
            "8/8/8/3pP3/8/8/8/K3k3 w - d6 0 2",
        ];
        for fen in FENS {
            let mut with = Searcher::new(Board::from_fen(fen).unwrap());
            let mut without = Searcher::new(Board::from_fen(fen).unwrap());
            without.use_accumulator = false;

            let (m1, s1) = with.search_root(6, -32000, 32000);
            let (m2, s2) = without.search_root(6, -32000, 32000);
            assert_eq!(s1, s2, "score differs in {fen}");
            assert_eq!(m1, m2, "best move differs in {fen}");
        }
    }

    /// A repetition that reaches back into the game before the search started
    /// has to be detected; the search path alone cannot see it.
    #[test]
    fn repetition_spanning_the_game_boundary_is_detected() {
        // Both sides shuffle their knights to and fro, so the position with
        // the knights on f3/f6 occurs twice in the game.
        let mut b = Board::from_fen("4k1n1/8/8/8/8/8/4P3/4K1N1 w - - 0 1").unwrap();
        b.game_history = vec![b.key];
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"] {
            let m = b.uci_move(uci).unwrap();
            assert!(b.make(m), "{uci} should be legal");
            b.record_game_move(m);
        }

        let s = Searcher::new(b);
        assert!(s.game_history_repeated());

        // A position that only occurred once does not count: the search path
        // itself has to find the second occurrence.
        let mut b2 = Board::from_fen("4k1n1/8/8/8/8/8/4P3/4K1N1 w - - 0 1").unwrap();
        b2.game_history = vec![b2.key];
        let m = b2.uci_move("g1f3").unwrap();
        assert!(b2.make(m));
        b2.record_game_move(m);
        let s2 = Searcher::new(b2);
        assert!(!s2.game_history_repeated());
    }

    /// Once the search itself has made an irreversible move, the history of
    /// the game is irrelevant: nothing before it can recur.
    #[test]
    fn game_history_is_ignored_after_an_irreversible_search_move() {
        let mut b = Board::from_fen("4k1n1/8/8/8/8/8/4P3/4K1N1 w - - 0 1").unwrap();
        b.game_history = vec![b.key];
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"] {
            let m = b.uci_move(uci).unwrap();
            assert!(b.make(m), "{uci} should be legal");
            b.record_game_move(m);
        }

        let mut s = Searcher::new(b);
        assert!(s.game_history_repeated());

        // A pawn push is irreversible, so the boundary of the repetition
        // window moves into the search and the game history is dropped.
        let m = s.board.uci_move("e2e3").unwrap();
        assert!(s.make(m));
        assert_eq!(s.irr_count, 1);
        assert!(!s.game_history_repeated());
    }

    #[test]
    fn probe_unprobe_round_trip() {
        for ply in [0usize, 1, 3, 7, 11] {
            for &s in &[5200, 550, -600, -5120] {
                assert_eq!(unprobe_score(probe_score(s, ply), ply), s);
            }
        }
    }

    #[test]
    fn mate_probe_increases_with_ply() {
        assert_eq!(probe_score(MATE - 100, 5), MATE - 95);
        assert_eq!(unprobe_score(MATE - 95, 5), MATE - 100);
        assert_eq!(probe_score(-(MATE - 100), 5), -(MATE - 95));
        assert_eq!(unprobe_score(-(MATE - 95), 5), -(MATE - 100));
    }

    #[test]
    fn score_string_mates_and_cp() {
        assert_eq!(score_string(MATE - 2), "score mate 1");
        assert_eq!(score_string(MATE - 4), "score mate 2");
        assert_eq!(score_string(-(MATE - 4)), "score mate -2");
        assert_eq!(score_string(123), "score cp 123");
        assert_eq!(score_string(0), "score cp 0");
    }

    #[test]
    fn faster_mate_scores_higher() {
        assert!(MATE - 2 > MATE - 4, "mate-in-1 must outrank mate-in-2");
    }

    #[test]
    fn think_depth_12_completes_safe() {
        // Safer regression: ensure depth-12 can complete and produce a best move, but avoid assertions that depend on exact PV or score
        let b = Board::startpos();
        let mut s = Searcher::new(b);
        let r = s.think(SearchLimits { depth: 12, ..SearchLimits::default() });
        assert!(r.depth >= 12, "expected depth >= 12, got {}", r.depth);
        assert!(r.best.is_some(), "expected a best move at depth 12");
        assert!(r.nodes > 0, "expected some nodes at depth 12");
    }

    #[test]
    fn think_depth_12_pv_line_computable() {
        let b = Board::startpos();
        let mut s = Searcher::new(b);
        let _ = s.think(SearchLimits { depth: 12, ..SearchLimits::default() });
        let pv = s.pv_line(2);
        assert!(pv.len() >= 1, "PV line should have at least one move");
    }

}

mod connected_to_board {
    use super::*;

    pub fn zobrist_side(board: &mut Board) {
        let lhs = crate::board::side_key();
        board.key ^= lhs;
    }

    pub fn zobrist_ep_clear(board: &mut Board) {
        if let Some(e) = board.ep {
            board.key ^= crate::board::ep_key(e);
        }
        board.ep = None;
    }

    pub fn zobrist_ep_restore(board: &mut Board, prev: Option<Option<usize>>) {
        let prev = prev.expect("null ep stack");
        if let Some(e) = prev {
            board.key ^= crate::board::ep_key(e);
        }
        board.ep = prev;
    }
}