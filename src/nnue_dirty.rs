//! Incremental NNUE dirty-state tracking.
//!
//! Stockfish's `sf_19` records, for every position transition, which piece
//! moved and which threat features appeared or disappeared, so that the NNUE
//! accumulator can be patched with a handful of weight rows instead of being
//! rebuilt from scratch.  This module derives the same information directly
//! from the before/after board states.
//!
//! A *threat feature* is a quadruple `(attacker, attacker square, target,
//! target square)`: the attacker really attacks the target square and the
//! target holds a piece the attacker is allowed to threaten.  The rules are
//! Stockfish's:
//!
//! | attacker | threatens                                  |
//! |----------|--------------------------------------------|
//! | pawn     | knights and rooks                         |
//! | knight   | every piece except the king               |
//! | bishop   | pawns, knights, bishops and rooks         |
//! | rook     | pawns, knights, bishops and rooks         |
//! | queen    | every piece except the king               |
//! | king     | nothing                                   |
//!
//! Kings are never threatened, so a king square never carries a feature.
//!
//! Three kinds of change are recorded:
//!
//! * **placements** - a piece appeared on or disappeared from a square;
//!   these drive the HalfKAv2_hm (PSQ) features,
//! * **threats** - a threat feature appeared or disappeared; these drive the
//!   FullThreats features,
//! * **pawn pairs** - the pawn bitboards before and after the move; these
//!   drive the PP_3Wide features.

use crate::bitboard::*;
use crate::board::EMPTY_SQ;

// The opposite of each ray direction.  `bitboard::DIRS` orders the rays as
// N, NE, E, NW, S, SE, W, SW, so the opposite of a diagonal is *not* the
// direction obtained by flipping bit 2 (that would turn NE into SE).
const OPP: [usize; 8] = [4, 7, 6, 5, 0, 3, 2, 1];

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Maximum number of threat features that can change in a single move.
/// Stockfish uses 96 as well; an overflow simply forces a full refresh.
pub const MAX_DIRTY_THREATS: usize = 96;

/// Maximum number of squares whose occupant can change in a single move
/// (source, destination, an en-passant victim and a castling rook).
pub const MAX_DIRTY_PLACEMENTS: usize = 4;

/// A threat feature that appeared or disappeared.
///
/// `attacker`/`attacked` use Stockfish's piece encoding
/// (`(colour << 3) + type`, type 1..=6) so that the feature index can be
/// computed directly.  For a removal the codes describe the *old* position,
/// for an addition the *new* one.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct DirtyThreat {
    pub attacker: u8,
    pub attacked: u8,
    pub from: u8,
    pub to: u8,
    /// `true` when the feature is present in the new position.
    pub add: bool,
}

/// A square whose occupant changed.  `before`/`after` are Stockfish piece
/// codes, `0` meaning "no piece".
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct DirtyPlacement {
    pub sq: u8,
    pub before: u8,
    pub after: u8,
}

/// Everything that changed between two positions.
#[derive(Clone, Copy)]
pub struct DirtyState {
    pub placements: [DirtyPlacement; MAX_DIRTY_PLACEMENTS],
    pub placement_len: usize,
    pub threats: [DirtyThreat; MAX_DIRTY_THREATS],
    pub threat_len: usize,
    /// Set when the threat list overflowed: the caller must refresh.
    pub overflow: bool,
    /// Pawn bitboards before the move.
    pub pawns_before: [u64; 2],
    /// Pawn bitboards after the move.
    pub pawns_after: [u64; 2],
    /// Perspectives whose king moved: their feature indices all change, so
    /// their accumulator has to be refreshed from scratch.
    pub refresh: [bool; 2],
}

impl Default for DirtyState {
    fn default() -> Self {
        DirtyState {
            placements: [DirtyPlacement::default(); MAX_DIRTY_PLACEMENTS],
            placement_len: 0,
            threats: [DirtyThreat::default(); MAX_DIRTY_THREATS],
            threat_len: 0,
            overflow: false,
            pawns_before: [0; 2],
            pawns_after: [0; 2],
            refresh: [false; 2],
        }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Collects the dirty state of a position transition.
///
/// The transition is processed in two phases because the board is mutated in
/// place: [`DirtyBuilder::before`] runs while the board still holds the old
/// position, [`DirtyBuilder::after`] runs once the move has been applied.
pub struct DirtyBuilder {
    pub state: DirtyState,
    /// Squares whose occupant changed.
    changed: [usize; MAX_DIRTY_PLACEMENTS],
    n_changed: usize,
    changed_mask: u64,
}

impl DirtyBuilder {
    pub fn new() -> Self {
        DirtyBuilder {
            state: DirtyState::default(),
            changed: [0; MAX_DIRTY_PLACEMENTS],
            n_changed: 0,
            changed_mask: 0,
        }
    }

    /// Register the squares whose occupant changes during the move.
    pub fn set_changed(&mut self, squares: &[usize]) {
        self.n_changed = squares.len().min(MAX_DIRTY_PLACEMENTS);
        self.changed_mask = 0;
        for i in 0..self.n_changed {
            self.changed[i] = squares[i];
            self.changed_mask |= bit(squares[i]);
        }
    }

    /// Phase one: record the features that exist in the old position.
    pub fn before(&mut self, b: &BoardView) {
        self.state.pawns_before = b.pawns();
        self.state.placement_len = self.n_changed;
        for i in 0..self.n_changed {
            let sq = self.changed[i];
            let code = b.mailbox[sq];
            self.state.placements[i].sq = sq as u8;
            self.state.placements[i].before = if code == EMPTY_SQ { 0 } else { code + 1 };
        }
        for i in 0..self.n_changed {
            let sq = self.changed[i];
            dirty_outgoing(b, sq, false, &mut self.state);
            dirty_incoming(b, sq, false, self.changed_mask, &mut self.state);
        }
    }

    /// Phase two: record the features that exist in the new position and the
    /// discovered attacks that appeared or vanished.
    pub fn after(&mut self, b: &BoardView, occ_before: u64) {
        self.state.pawns_after = b.pawns();
        for i in 0..self.n_changed {
            let sq = self.changed[i];
            let code = b.mailbox[sq];
            self.state.placements[i].after = if code == EMPTY_SQ { 0 } else { code + 1 };
            dirty_outgoing(b, sq, true, &mut self.state);
            dirty_incoming(b, sq, true, self.changed_mask, &mut self.state);
        }
        // Discovered attacks: a slider that sees through a square whose
        // occupancy changed gains or loses the target behind it.
        for i in 0..self.n_changed {
            let sq = self.changed[i];
            dirty_discovered(occ_before, b.occ, self.changed_mask, sq, b, &mut self.state);
        }
    }

    /// Note that the king of `color` moved, so that perspective's feature
    /// indices are no longer valid.
    pub fn mark_king_moved(&mut self, color: usize) {
        self.state.refresh[color] = true;
    }
}

impl Default for DirtyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Read-only view of a position
// ---------------------------------------------------------------------------

/// The pieces the threat enumeration needs.
pub struct BoardView<'a> {
    /// Occupancy of all pieces.
    pub occ: u64,
    /// Bitboards per piece type, indexed `[colour][type]` with the crate's
    /// piece types (`PAWN = 0` .. `KING = 5`).
    pub types: &'a [[u64; 6]; 2],
    /// Piece code per square, `(colour << 3) | type`, or `EMPTY_SQ`.
    pub mailbox: &'a [u8; 64],
}

impl<'a> BoardView<'a> {
    pub fn pawns(&self) -> [u64; 2] {
        [self.types[0][PAWN], self.types[1][PAWN]]
    }
}

// ---------------------------------------------------------------------------
// Threat enumeration
// ---------------------------------------------------------------------------

/// The squares a piece of the given type is allowed to threaten.
#[inline]
fn target_mask(pt: usize, b: &BoardView) -> u64 {
    match pt {
        PAWN => b.types[0][KNIGHT] | b.types[1][KNIGHT] | b.types[0][ROOK] | b.types[1][ROOK],
        KNIGHT | QUEEN => b.occ & !(b.types[0][KING] | b.types[1][KING]),
        BISHOP | ROOK => {
            b.types[0][PAWN]
                | b.types[1][PAWN]
                | b.types[0][KNIGHT]
                | b.types[1][KNIGHT]
                | b.types[0][BISHOP]
                | b.types[1][BISHOP]
                | b.types[0][ROOK]
                | b.types[1][ROOK]
        }
        _ => 0,
    }
}

/// The squares attacked by a piece of the given type standing on `sq`.
#[inline]
fn attack_set(pt: usize, color: usize, sq: usize, occ: u64) -> u64 {
    match pt {
        PAWN => pawn_attacks(color, sq),
        KNIGHT => KNIGHT_ATTACKS[sq],
        BISHOP => bishop_attacks(occ, sq),
        ROOK => rook_attacks(occ, sq),
        QUEEN => bishop_attacks(occ, sq) | rook_attacks(occ, sq),
        _ => 0,
    }
}

/// Can a piece of type `attacker` threaten a piece of type `target`?
#[inline]
fn threatens(attacker: usize, target: usize) -> bool {
    match attacker {
        PAWN => target == KNIGHT || target == ROOK,
        KNIGHT | QUEEN => target != KING,
        BISHOP | ROOK => {
            target == PAWN || target == KNIGHT || target == BISHOP || target == ROOK
        }
        _ => false,
    }
}

/// Record the threat features in which the piece on `sq` is the attacker.
fn dirty_outgoing(b: &BoardView, sq: usize, add: bool, st: &mut DirtyState) {
    let code = b.mailbox[sq];
    if code == EMPTY_SQ {
        return;
    }
    let color = (code >> 3) as usize;
    let pt = (code & 7) as usize;
    if pt == KING {
        return; // kings emit no threats
    }
    let mut attacks = attack_set(pt, color, sq, b.occ) & target_mask(pt, b);
    while attacks != 0 {
        let to = attacks.trailing_zeros() as usize;
        attacks &= attacks - 1;
        st.push_threat(DirtyThreat {
            attacker: code + 1,
            attacked: b.mailbox[to] + 1,
            from: sq as u8,
            to: to as u8,
            add,
        });
    }
}

/// Record the threat features in which the piece on `sq` is the target.
fn dirty_incoming(
    b: &BoardView,
    sq: usize,
    add: bool,
    changed_mask: u64,
    st: &mut DirtyState,
) {
    let tcode = b.mailbox[sq];
    if tcode == EMPTY_SQ {
        return;
    }
    let tpt = (tcode & 7) as usize;
    if tpt == KING {
        return; // kings are never threatened
    }
    let target = tcode + 1;

    // Knights threaten everything but the king.
    let mut knights = KNIGHT_ATTACKS[sq] & (b.types[0][KNIGHT] | b.types[1][KNIGHT]);
    while knights != 0 {
        let from = knights.trailing_zeros() as usize;
        knights &= knights - 1;
        // Attackers on a changed square are handled by the outgoing pass.
        if changed_mask & bit(from) != 0 {
            continue;
        }
        st.push_threat(DirtyThreat {
            attacker: b.mailbox[from] + 1,
            attacked: target,
            from: from as u8,
            to: sq as u8,
            add,
        });
    }

    // Pawns threaten knights and rooks only.
    if tpt == KNIGHT || tpt == ROOK {
        let mut wp = pawn_attacks(BLACK, sq) & b.types[0][PAWN];
        while wp != 0 {
            let from = wp.trailing_zeros() as usize;
            wp &= wp - 1;
            if changed_mask & bit(from) != 0 {
                continue;
            }
            st.push_threat(DirtyThreat {
                attacker: b.mailbox[from] + 1,
                attacked: target,
                from: from as u8,
                to: sq as u8,
                add,
            });
        }
        let mut bp = pawn_attacks(WHITE, sq) & b.types[1][PAWN];
        while bp != 0 {
            let from = bp.trailing_zeros() as usize;
            bp &= bp - 1;
            if changed_mask & bit(from) != 0 {
                continue;
            }
            st.push_threat(DirtyThreat {
                attacker: b.mailbox[from] + 1,
                attacked: target,
                from: from as u8,
                to: sq as u8,
                add,
            });
        }
    }

    // Sliders along the eight rays.
    for d in 0..8 {
        let ray = RAY[sq][d] & b.occ;
        if ray == 0 {
            continue;
        }
        let from = nearest(ray, d);
        // Attackers on a changed square are handled by the outgoing pass.
        if changed_mask & bit(from) != 0 {
            continue;
        }
        let scode = b.mailbox[from];
        let spt = (scode & 7) as usize;
        // The attacker has to be a slider that can move along this ray: a
        // bishop on the file above the target does not attack it.
        let along_ray = if d % 2 == 0 {
            spt == ROOK || spt == QUEEN
        } else {
            spt == BISHOP || spt == QUEEN
        };
        if !along_ray || !threatens(spt, tpt) {
            continue;
        }
        st.push_threat(DirtyThreat {
            attacker: scode + 1,
            attacked: target,
            from: from as u8,
            to: sq as u8,
            add,
        });
    }
}

/// The nearest occupied square of `bb` along direction `d`.
#[inline]
fn nearest(bb: u64, d: usize) -> usize {
    if DIRD[d] > 0 {
        bb.trailing_zeros() as usize
    } else {
        63 - bb.leading_zeros() as usize
    }
}

/// Is `a` closer to the ray origin than `b`?
#[inline]
fn closer(a: usize, b: usize, d: usize) -> bool {
    if DIRD[d] > 0 {
        a < b
    } else {
        a > b
    }
}

/// Record the discovered attacks that changed because the occupancy of `sq`
/// changed.
///
/// A slider that sees *through* `sq` attacks the first occupied square
/// behind it.  When `sq` becomes occupied the attack disappears, when it
/// becomes empty the attack appears.
fn dirty_discovered(
    occ_before: u64,
    occ_after: u64,
    changed_mask: u64,
    sq: usize,
    b: &BoardView,
    st: &mut DirtyState,
) {
    for d in 0..8 {
        let side_a = d; // the slider side
        let side_b = OPP[d]; // the target side

        let before = discovered_feature(
            occ_before,
            changed_mask,
            sq,
            side_a,
            side_b,
            b,
        );
        let after = discovered_feature(occ_after, changed_mask, sq, side_a, side_b, b);

        match (before, after) {
            (Some(x), Some(y)) => {
                if x != y {
                    st.push_threat(DirtyThreat { add: false, ..x });
                    st.push_threat(DirtyThreat { add: true, ..y });
                }
            }
            (Some(x), None) => st.push_threat(DirtyThreat { add: false, ..x }),
            (None, Some(y)) => st.push_threat(DirtyThreat { add: true, ..y }),
            (None, None) => {}
        }
    }
}

/// The discovered (through `sq`) threat feature of one position, if any.
fn discovered_feature(
    occ: u64,
    changed_mask: u64,
    sq: usize,
    side_a: usize,
    side_b: usize,
    b: &BoardView,
) -> Option<DirtyThreat> {
    // The ray only passes through `sq` when the square is empty.
    if occ & bit(sq) != 0 {
        return None;
    }
    let ray_a = RAY[sq][side_a];
    let occ_a = ray_a & occ;
    if occ_a == 0 {
        return None;
    }
    let prev = nearest(occ_a, side_a);
    // If a changed square lies between the slider and `sq`, that square is
    // responsible for the feature and it is recorded there instead.
    let chg_a = ray_a & changed_mask;
    if chg_a != 0 {
        let chg = nearest(chg_a, side_a);
        if closer(chg, prev, side_a) {
            return None;
        }
    }
    let scode = b.mailbox[prev];
    let spt = (scode & 7) as usize;
    // The attacker has to be a slider matching the direction.
    let ok = if side_a % 2 == 0 {
        spt == ROOK || spt == QUEEN
    } else {
        spt == BISHOP || spt == QUEEN
    };
    if !ok {
        return None;
    }
    // The target is the first occupied square on the far side of `sq`.
    let occ_b = RAY[sq][side_b] & occ;
    if occ_b == 0 {
        return None;
    }
    let t = nearest(occ_b, side_b);
    let tcode = b.mailbox[t];
    let tpt = (tcode & 7) as usize;
    if !threatens(spt, tpt) {
        return None;
    }
    // Features that involve a changed square are recorded by the outgoing
    // and incoming passes, not here.
    if changed_mask & (bit(prev) | bit(t)) != 0 {
        return None;
    }
    Some(DirtyThreat {
        attacker: scode + 1,
        attacked: tcode + 1,
        from: prev as u8,
        to: t as u8,
        add: false,
    })
}

impl DirtyState {
    #[inline]
    fn push_threat(&mut self, t: DirtyThreat) {
        if self.threat_len >= MAX_DIRTY_THREATS {
            self.overflow = true;
            return;
        }
        self.threats[self.threat_len] = t;
        self.threat_len += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::movegen::{generate_legal, MoveList};
    use crate::nnue::{load_nnue, NnueNetwork};
    use crate::nnue_forward::Accumulator;
    use std::sync::OnceLock;

    fn net() -> &'static NnueNetwork {
        static NET: OnceLock<NnueNetwork> = OnceLock::new();
        NET.get_or_init(|| {
            load_nnue("nn-1a298aa575a0.nnue").expect("network file present")
        })
    }

    /// A cheap deterministic PRNG so the test does not need a dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    /// The evaluation computed from an incrementally maintained accumulator
    /// must be identical to the one computed from a full rebuild.
    #[test]
    fn accumulated_evaluation_matches_full_evaluation() {
        let parsed = &net().parsed;
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut checked = 0usize;

        for _game in 0..12 {
            let mut board = Board::startpos();
            for _ply in 0..20 {
                let mut list = MoveList::new();
                generate_legal(&mut board, &mut list);
                if list.is_empty() {
                    break;
                }
                let m = list.as_slice()[(rng.next() as usize) % list.len()];

                let mut parent = Accumulator::new();
                parent.refresh(parsed, &board);

                board.nnue_track = true;
                if !board.make(m) {
                    board.unmake(m);
                    board.nnue_track = false;
                    continue;
                }
                let dirty = board.last_dirty;
                board.nnue_track = false;

                let mut incremental = Accumulator::new();
                incremental.apply(&parent, parsed, &dirty, &board);

                let with_acc = crate::eval::evaluate_with_accum(
                    parsed,
                    &board,
                    Some(&incremental),
                );
                let without = crate::eval::evaluate_with_net(parsed, &board);
                assert_eq!(
                    with_acc, without,
                    "evaluation mismatch after {:?} in {}",
                    m,
                    board.to_fen()
                );
                checked += 1;
            }
        }
        assert!(checked > 50, "too few positions checked: {checked}");
    }

    /// A threat list that overflowed its buffer forces a full refresh of both
    /// perspectives.  The deltas of the move must not be added on top of the
    /// rebuilt accumulator, or they would be counted twice.
    #[test]
    fn overflow_refreshes_instead_of_patching() {
        let parsed = &net().parsed;
        let mut board =
            Board::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
                .unwrap();

        let mut parent = Accumulator::new();
        parent.refresh(parsed, &board);

        let m = board.uci_move("f3g5").unwrap();
        board.nnue_track = true;
        assert!(board.make(m));
        let mut dirty = board.last_dirty;
        board.nnue_track = false;
        assert!(!dirty.overflow);
        dirty.overflow = true;

        let mut acc = Accumulator::new();
        acc.apply(&parent, parsed, &dirty, &board);

        let with_acc = crate::eval::evaluate_with_accum(parsed, &board, Some(&acc));
        let without = crate::eval::evaluate_with_net(parsed, &board);
        assert_eq!(with_acc, without);
    }

    /// Check the incremental accumulator against a full rebuild for a list of
    /// hand-picked positions and moves that exercise the tricky parts of the
    /// dirty-state derivation.
    #[test]
    fn incremental_matches_full_on_special_moves() {
        use crate::board::{M_CASTLE, M_EP, M_PROMO};
        let cases: &[(&str, crate::board::Move)] = &[
            // En passant: the captured pawn is not on the destination square.
            (
                "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
                crate::board::Move {
                    from: 36,
                    to: 45,
                    piece: 0,
                    captured: 0,
                    promo: 6,
                    flags: M_EP,
                },
            ),
            // En passant with a discovered attack through the captured pawn.
            (
                "8/8/8/3pP3/8/8/8/K3k3 w - d6 0 2",
                crate::board::Move {
                    from: 36,
                    to: 43,
                    piece: 0,
                    captured: 0,
                    promo: 6,
                    flags: M_EP,
                },
            ),
            // Kingside castling: two squares change their occupant.
            (
                "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
                crate::board::Move {
                    from: 4,
                    to: 6,
                    piece: 5,
                    captured: 6,
                    promo: 6,
                    flags: M_CASTLE,
                },
            ),
            // Queenside castling.
            (
                "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
                crate::board::Move {
                    from: 60,
                    to: 58,
                    piece: 5,
                    captured: 6,
                    promo: 6,
                    flags: M_CASTLE,
                },
            ),
            // Promotion to a queen.
            (
                "8/4P3/8/8/8/8/8/4K1k1 w - - 0 1",
                crate::board::Move {
                    from: 52,
                    to: 60,
                    piece: 0,
                    captured: 6,
                    promo: 4,
                    flags: M_PROMO,
                },
            ),
            // Promotion by capture.
            (
                "1n5k/P7/8/8/8/8/8/4K3 w - - 0 1",
                crate::board::Move {
                    from: 48,
                    to: 57,
                    piece: 0,
                    captured: 1,
                    promo: 3,
                    flags: M_PROMO,
                },
            ),
            // A capture that opens a discovered attack: the rook takes the
            // pawn that shielded it from the queen.
            (
                "4k3/8/8/q7/8/p7/8/R6K w - - 0 1",
                crate::board::Move {
                    from: 0,
                    to: 16,
                    piece: 3,
                    captured: 0,
                    promo: 6,
                    flags: 0,
                },
            ),
            // A knight capture next to the enemy king: the king square is
            // never a threat target.
            (
                "8/8/5k2/8/8/3N4/8/4K3 w - - 0 1",
                crate::board::Move {
                    from: 19,
                    to: 36,
                    piece: 1,
                    captured: 6,
                    promo: 6,
                    flags: 0,
                },
            ),
            // A double check: both the moving piece and a discovered line
            // give check.
            (
                "r3k2r/8/8/8/8/8/6b1/R3K2R w KQkq - 0 1",
                crate::board::Move {
                    from: 0,
                    to: 2,
                    piece: 3,
                    captured: 6,
                    promo: 6,
                    flags: 0,
                },
            ),
        ];

        let parsed = &net().parsed;
        for (fen, m) in cases {
            let mut board = Board::from_fen(fen).expect("valid fen");
            board.nnue_track = true;
            let mut parent = Accumulator::new();
            parent.refresh(parsed, &board);
            assert!(board.make(*m), "move must be legal in {fen}");
            let dirty = board.last_dirty;

            let mut incremental = Accumulator::new();
            incremental.apply(&parent, parsed, &dirty, &board);
            let mut full = Accumulator::new();
            full.refresh(parsed, &board);

            assert_eq!(
                incremental.acc, full.acc,
                "accumulator mismatch after {:?} in {}",
                m, board.to_fen()
            );
            assert_eq!(
                incremental.psqt, full.psqt,
                "psqt mismatch after {:?} in {}",
                m, board.to_fen()
            );
        }
    }

    /// Walk a few random moves from the start position, then check that the
    /// incremental accumulator matches a full rebuild after every move.
    #[test]
    fn incremental_accumulator_matches_full_refresh() {
        let parsed = &net().parsed;
        let mut rng = Rng(0x2545F4914F6CDD1D);
        let mut checked = 0usize;

        for game in 0..24 {
            let mut board = Board::startpos();
            for _ply in 0..24 {
                // Pick a random legal move.
                let mut list = MoveList::new();
                generate_legal(&mut board, &mut list);
                if list.is_empty() {
                    break;
                }
                let m = list.as_slice()[(rng.next() as usize) % list.len()];

                // The parent's accumulator.
                let mut parent = Accumulator::new();
                parent.refresh(parsed, &board);

                board.nnue_track = true;
                let legal = board.make(m);
                if !legal {
                    board.unmake(m);
                    continue;
                }
                let dirty = board.last_dirty;

                // The incremental accumulator.
                let mut incremental = Accumulator::new();
                incremental.apply(&parent, parsed, &dirty, &board);

                // The full rebuild.
                let mut full = Accumulator::new();
                full.refresh(parsed, &board);

                assert_eq!(
                    incremental.acc, full.acc,
                    "accumulator mismatch after {:?} in {} (game {game})",
                    m, board.to_fen()
                );
                assert_eq!(
                    incremental.psqt, full.psqt,
                    "psqt mismatch after {:?} in {} (game {game})",
                    m, board.to_fen()
                );

                board.nnue_track = false;
                checked += 1;

                // Continue the random walk.
                let mut list2 = MoveList::new();
                generate_legal(&mut board, &mut list2);
                if list2.is_empty() {
                    break;
                }
                let m2 = list2.as_slice()[(rng.next() as usize) % list2.len()];
                board.nnue_track = false;
                board.make(m2);
                board.unmake(m2);
            }
        }
        assert!(checked > 100, "too few positions checked: {checked}");
    }
}
