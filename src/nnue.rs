// Corvus Chess Engine
// NNUE — parser for the official Stockfish `sf_19` network format
// (`nn-1a298aa575a0.nnue`).
//
// The binary layout below is NOT guessed. It is transcribed directly from the
// authoritative C++ sources at the `sf_19` release tag of
// https://github.com/official-stockfish/Stockfish:
//
//   * src/nnue/network.cpp            — header + per-component hash dispatch
//   * src/nnue/nnue_common.h          — Version magic, LEB128 framing/decode,
//                                       quantization types
//   * src/nnue/nnue_feature_transformer.h — FT array order and dimensions
//   * src/nnue/nnue_architecture.h    — L1/L2/L3, PSQTBuckets, LayerStacks,
//                                       layer-stack composition + arch hash
//   * src/nnue/layers/affine_transform*.h, clipped_relu.h — layer param layout
//                                       and hash steps
//   * src/nnue/features/*.h           — feature-set Dimensions / HashValue
//
// File-level layout (all integers little-endian):
//
//   [Header]
//     u32 version  == 0x6A448AFA            (Version)
//     u32 archHash == Network::hash         (FT hash ^ arch hash)
//     u32 descLen
//     u8  description[descLen]
//   [FeatureTransformer]                    preceded by u32 FT hash
//     leb128 block : biases          i16 x HalfDimensions
//     raw   LE     : threatWeights   i8  x (FullThreats::Dimensions * L1)
//     leb128 block : threatPsqt      i32 x (FullThreats::Dimensions * PSQTBuckets)
//     raw   LE     : pawnPairWeights i8  x (PP_3Wide::Dimensions * L1)
//     leb128 block : pawnPairPsqt    i32 x (PP_3Wide::Dimensions * PSQTBuckets)
//     leb128 block : weights         i16 x (L1 * HalfKAv2_hm::Dimensions)
//     leb128 block : psqtWeights     i32 x (PSQTBuckets * HalfKAv2_hm::Dimensions)
//   [LayerStacks x 8]                      each preceded by u32 arch hash
//     fc_0 : biases i32 x L2,            weights i8 x (L2 * pad(L1))
//     fc_1 : biases i32 x L2,            weights i8 x (L2 * pad(2*L2))
//     fc_2 : biases i32 x 1,             weights i8 x (1  * pad(2*L2 + 2*L3))
//   ...stream must end exactly at EOF.
//
// Every leb128 block is framed as: 17-byte magic "COMPRESSED_LEB128",
// u32 byte_count, then byte_count bytes of packed signed LEB128 values.

use crate::board::Board;
use crate::bitboard::*;

// ---------------------------------------------------------------------------
// Format constants (transcribed from sf_19 headers)
// ---------------------------------------------------------------------------

/// `Version` in nnue_common.h — the file magic.
pub const NNUE_VERSION: u32 = 0x6A448AFA;

const LEB128_MAGIC: &[u8; 17] = b"COMPRESSED_LEB128";

// NetworkArchitecture dims (nnue_architecture.h).
pub const L1: usize = 1024;
pub const L2: usize = 32;
pub const L3: usize = 32;
pub const PSQ_T_BUCKETS: usize = 8;
pub const LAYER_STACKS: usize = 8;

/// `MaxSimdWidth` (nnue_common.h) — used to pad affine input dims.
const MAX_SIMD_WIDTH: usize = 32;

// Feature-set dimensions (features/*.h).
pub const FULL_THREATS_DIM: usize = 59808; // FullThreats::Dimensions
pub const PP3WIDE_PAWN_IDS: usize = 96; // COLOR_NB * 48
pub const PP3WIDE_DIM: usize = PP3WIDE_PAWN_IDS * (PP3WIDE_PAWN_IDS - 1) / 2; // 4560
pub const SQUARE_NB: usize = 64;
pub const PS_NB: usize = 11 * SQUARE_NB; // 704
pub const HALF_KA_V2_HM_DIM: usize = SQUARE_NB * PS_NB / 2; // 22528

// FeatureTransformer derived dims (nnue_feature_transformer.h).
pub const HALF_DIMENSIONS: usize = L1; // OutputDimensions = HalfDimensions
pub const THREAT_INPUT_DIM: usize = FULL_THREATS_DIM;
pub const PAIR_INPUT_DIM: usize = PP3WIDE_DIM;
pub const PSQ_DIM: usize = HALF_KA_V2_HM_DIM; // PsqDimensions
pub const THREAT_AND_PP_DIM: usize = THREAT_INPUT_DIM + PAIR_INPUT_DIM;
/// Total feature-transformer input width.
pub const FT_INPUT_DIM: usize = PSQ_DIM + THREAT_AND_PP_DIM; // 86896
/// Feature-transformer output width (one perspective).
pub const FT_OUTPUT_DIM: usize = HALF_DIMENSIONS; // 1024

// Hash component constants (features/*.h HashValue).
const FULL_THREATS_HASH: u32 = 0x2e6b9d04;
const PP3WIDE_HASH: u32 = 0x86f2b1dd;
const HALF_KA_V2_HM_HASH: u32 = 0x7f234cb8;

// ---------------------------------------------------------------------------
// Hash-value computation (mirrors get_hash_value() in the C++ sources)
// ---------------------------------------------------------------------------

/// `combine_hash({..})`: rotate-left-1 then XOR, folded over the list.
fn combine_hash(hashes: &[u32]) -> u32 {
    let mut h: u32 = 0;
    for &c in hashes {
        h = h.rotate_left(1) ^ c;
    }
    h
}

/// `FeatureTransformer::get_hash_value()`.
pub fn ft_hash() -> u32 {
    combine_hash(&[FULL_THREATS_HASH, PP3WIDE_HASH, HALF_KA_V2_HM_HASH])
        ^ (HALF_DIMENSIONS as u32 * 2)
}

/// `AffineTransform[<SparseInput>]::get_hash_value(prev)`.
fn fc_hash(prev: u32, out_dims: u32) -> u32 {
    let mut h = 0xCC03DAE4u32.wrapping_add(out_dims);
    h ^= prev >> 1;
    h ^= prev << 31;
    h
}

/// `ClippedReLU::get_hash_value(prev)` (also SqrClippedReLU's).
fn clipped_relu_hash(prev: u32) -> u32 {
    0x538D24C7u32.wrapping_add(prev)
}

/// `NetworkArchitecture::get_hash_value()`.
pub fn arch_hash() -> u32 {
    // input-slice hash
    let mut h = 0xEC42E90Du32 ^ (L1 as u32 * 2);
    h = fc_hash(h, L2 as u32); // fc_0
    h = clipped_relu_hash(h); // ac_0
    h = fc_hash(h, L2 as u32); // fc_1
    h = clipped_relu_hash(h); // ac_1
    h = fc_hash(h, 1); // fc_2
    h
}

/// `Network::hash` — the archHash embedded in the file header.
pub fn network_hash() -> u32 {
    ft_hash() ^ arch_hash()
}

fn ceil_to_multiple(x: usize, m: usize) -> usize {
    ((x + m - 1) / m) * m
}

// ---------------------------------------------------------------------------
// Byte reader
// ---------------------------------------------------------------------------

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        if self.pos >= self.data.len() {
            return Err(format!("unexpected EOF at byte {}", self.pos));
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        if self.pos + 4 > self.data.len() {
            return Err(format!("unexpected EOF reading u32 at {}", self.pos));
        }
        let v = u32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    /// Advance over `n` raw little-endian bytes, returning them as a slice.
    fn read_raw_bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() {
            return Err(format!(
                "unexpected EOF reading {} raw bytes at {} (file len {})",
                n,
                self.pos,
                self.data.len()
            ));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// Decode one signed LEB128 value into the raw u32 bit-pattern the C++
    /// `read_leb_128_detail` would assign before casting to the target type.
    ///
    /// Exact rule from nnue_common.h:
    ///   result |= (byte & 0x7f) << (shift % 32); shift += 7;
    ///   on terminator: value = (shift >= 32 || (byte & 0x40) == 0)
    ///                      ? result
    ///                      : result | !((1 << shift) - 1);
    fn read_sleb128_raw(&mut self) -> Result<u32, String> {
        let mut result: u32 = 0;
        let mut shift: usize = 0;
        loop {
            let byte = self.read_u8()?;
            result |= (byte as u32 & 0x7f) << (shift % 32);
            shift += 7;
            if byte & 0x80 == 0 {
                let value = if shift >= 32 || (byte & 0x40) == 0 {
                    result
                } else {
                    result | !((1u32 << shift) - 1)
                };
                return Ok(value);
            }
        }
    }

    /// Read a framed LEB128 block (magic + byte_count + `count` packed values),
    /// emitting each decoded raw u32 through `emit`. Verifies that exactly
    /// `byte_count` bytes are consumed, matching the C++ post-condition
    /// `bytes_left != 0 => fail`.
    fn read_leb128_block<F>(&mut self, count: usize, mut emit: F) -> Result<(), String>
    where
        F: FnMut(u32),
    {
        // magic
        if self.pos + 17 > self.data.len() {
            return Err(format!("EOF before LEB128 magic at {}", self.pos));
        }
        let magic = &self.data[self.pos..self.pos + 17];
        if magic != LEB128_MAGIC {
            return Err(format!(
                "bad LEB128 magic at {}: {:02x?}",
                self.pos, magic
            ));
        }
        self.pos += 17;

        let byte_count = self.read_u32()? as usize;
        let start = self.pos;
        for _ in 0..count {
            let raw = self.read_sleb128_raw()?;
            emit(raw);
        }
        let consumed = self.pos - start;
        if consumed != byte_count {
            return Err(format!(
                "LEB128 byte_count mismatch: header said {}, decoded {}",
                byte_count, consumed
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Parsed network representation
// ---------------------------------------------------------------------------

/// One fully-connected layer's serialized parameters.
#[derive(Debug, Clone)]
pub struct FcLayer {
    pub biases: Vec<i32>,
    pub weights: Vec<i8>,
    pub input_dims: usize,
    pub padded_input_dims: usize,
    pub output_dims: usize,
}

/// One of the 8 identical-structure layer stacks.
#[derive(Debug, Clone)]
pub struct LayerStack {
    pub fc_0: FcLayer,
    pub fc_1: FcLayer,
    pub fc_2: FcLayer,
}

/// The decompressed, validated network weights.
#[derive(Debug, Clone)]
pub struct ParsedNetwork {
    pub description: String,

    // Feature transformer.
    pub ft_biases: Vec<i16>,           // [L1]
    pub threat_weights: Vec<i8>,       // [FullThreats::Dimensions * L1]
    pub threat_psqt: Vec<i32>,         // [FullThreats::Dimensions * PSQTBuckets]
    pub pawn_pair_weights: Vec<i8>,    // [PP_3Wide::Dimensions * L1]
    pub pawn_pair_psqt: Vec<i32>,      // [PP_3Wide::Dimensions * PSQTBuckets]
    pub ft_weights: Vec<i16>,          // [L1 * PSQ_DIM]
    pub ft_psqt_weights: Vec<i32>,     // [PSQTBuckets * PSQ_DIM]

    // 8 layer stacks.
    pub stacks: Vec<LayerStack>,

    // Validated hashes read from the file.
    pub file_ft_hash: u32,
    pub file_arch_hash: u32, // Network::hash (header)
}

impl ParsedNetwork {
    /// Feature-transformer dimensions, as required for the forward pass.
    pub fn feature_transformer_dims(&self) -> FeatureTransformerDims {
        FeatureTransformerDims {
            input_dim: FT_INPUT_DIM,
            output_dim: FT_OUTPUT_DIM,
            half_dimensions: HALF_DIMENSIONS,
            psq_dim: PSQ_DIM,
            threat_dim: THREAT_INPUT_DIM,
            pair_dim: PAIR_INPUT_DIM,
            psqt_buckets: PSQ_T_BUCKETS,
        }
    }
}

/// The feature-transformer geometry.
#[derive(Debug, Clone, Copy)]
pub struct FeatureTransformerDims {
    pub input_dim: usize,
    pub output_dim: usize,
    pub half_dimensions: usize,
    pub psq_dim: usize,
    pub threat_dim: usize,
    pub pair_dim: usize,
    pub psqt_buckets: usize,
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NnueHeader {
    /// Version magic (== NNUE_VERSION).
    pub magic: u32,
    /// File version (same field as the magic).
    pub version: u32,
    /// `Network::hash` embedded in the header.
    pub arch: u32,
    pub description: String,
    /// First words of the file, for debugging.
    pub raw: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct NnueNetwork {
    pub header: NnueHeader,
    /// Full raw file bytes (kept for the load test / debugging).
    pub data: Vec<u8>,
    /// Fully parsed + validated weights.
    pub parsed: ParsedNetwork,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse and validate an entire `sf_19` NNUE file.
///
/// Checks performed (any failure returns `Err`):
///   * version magic == 0x6A448AFA
///   * header archHash == Network::hash (FT hash ^ arch hash)
///   * per-component FT hash == FeatureTransformer::get_hash_value()
///   * each of the 8 layer stacks' arch hash == NetworkArchitecture::get_hash_value()
///   * every LEB128 block's byte_count matches the decoded stream length
///   * the stream ends exactly at EOF
pub fn parse(data: &[u8]) -> Result<NnueNetwork, String> {
    let mut r = Reader::new(data);

    // ---- Header ----
    let magic = r.read_u32()?;
    if magic != NNUE_VERSION {
        return Err(format!(
            "bad version magic: got 0x{magic:08X}, expected 0x{NNUE_VERSION:08X}"
        ));
    }
    let arch = r.read_u32()?;
    let expected_net_hash = network_hash();
    if arch != expected_net_hash {
        return Err(format!(
            "header archHash mismatch: got 0x{arch:08X}, expected 0x{expected_net_hash:08X}"
        ));
    }
    let desc_len = r.read_u32()? as usize;
    let desc_bytes = r.read_raw_bytes(desc_len)?;
    let description = String::from_utf8_lossy(desc_bytes).into_owned();

    // ---- Feature transformer (prefixed by its hash) ----
    let file_ft_hash = r.read_u32()?;
    let expected_ft_hash = ft_hash();
    if file_ft_hash != expected_ft_hash {
        return Err(format!(
            "FT hash mismatch: got 0x{file_ft_hash:08X}, expected 0x{expected_ft_hash:08X}"
        ));
    }

    // biases: i16 x HalfDimensions
    let mut ft_biases = Vec::with_capacity(HALF_DIMENSIONS);
    r.read_leb128_block(HALF_DIMENSIONS, |v| ft_biases.push(v as i16))?;

    // threatWeights: raw LE i8 x (FullThreats::Dimensions * L1)
    let threat_weight_size = THREAT_INPUT_DIM * HALF_DIMENSIONS;
    let tw = r.read_raw_bytes(threat_weight_size)?;
    let threat_weights: Vec<i8> = tw.iter().map(|&b| b as i8).collect();

    // threatPsqt: i32 x (FullThreats::Dimensions * PSQTBuckets)
    let threat_psqt_size = THREAT_INPUT_DIM * PSQ_T_BUCKETS;
    let mut threat_psqt = Vec::with_capacity(threat_psqt_size);
    r.read_leb128_block(threat_psqt_size, |v| threat_psqt.push(v as i32))?;

    // pawnPairWeights: raw LE i8 x (PP_3Wide::Dimensions * L1)
    let pair_weight_size = PAIR_INPUT_DIM * HALF_DIMENSIONS;
    let pw = r.read_raw_bytes(pair_weight_size)?;
    let pawn_pair_weights: Vec<i8> = pw.iter().map(|&b| b as i8).collect();

    // pawnPairPsqt: i32 x (PP_3Wide::Dimensions * PSQTBuckets)
    let pair_psqt_size = PAIR_INPUT_DIM * PSQ_T_BUCKETS;
    let mut pawn_pair_psqt = Vec::with_capacity(pair_psqt_size);
    r.read_leb128_block(pair_psqt_size, |v| pawn_pair_psqt.push(v as i32))?;

    // weights: i16 x (L1 * PSQ_DIM)
    let ft_weight_size = HALF_DIMENSIONS * PSQ_DIM;
    let mut ft_weights = Vec::with_capacity(ft_weight_size);
    r.read_leb128_block(ft_weight_size, |v| ft_weights.push(v as i16))?;

    // psqtWeights: i32 x (PSQTBuckets * PSQ_DIM)
    let ft_psqt_size = PSQ_T_BUCKETS * PSQ_DIM;
    let mut ft_psqt_weights = Vec::with_capacity(ft_psqt_size);
    r.read_leb128_block(ft_psqt_size, |v| ft_psqt_weights.push(v as i32))?;

    // ---- 8 layer stacks ----
    let expected_arch_hash = arch_hash();
    let fc0_padded = ceil_to_multiple(L1, MAX_SIMD_WIDTH);
    let fc1_padded = ceil_to_multiple(L2 * 2, MAX_SIMD_WIDTH);
    let fc2_padded = ceil_to_multiple(L2 * 2 + L3 * 2, MAX_SIMD_WIDTH);

    let mut stacks = Vec::with_capacity(LAYER_STACKS);
    for i in 0..LAYER_STACKS {
        let stack_arch = r.read_u32()?;
        if stack_arch != expected_arch_hash {
            return Err(format!(
                "layer stack {i} arch hash mismatch: got 0x{stack_arch:08X}, expected 0x{expected_arch_hash:08X}"
            ));
        }

        // fc_0: AffineTransformSparseInput<L1, L2>
        let mut fc0_biases = Vec::with_capacity(L2);
        let b = r.read_raw_bytes(L2 * 4)?;
        for c in b.chunks_exact(4) {
            fc0_biases.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        let w = r.read_raw_bytes(L2 * fc0_padded)?;
        let fc0_weights: Vec<i8> = w.iter().map(|&x| x as i8).collect();

        // fc_1: AffineTransform<2*L2, L2>
        let mut fc1_biases = Vec::with_capacity(L2);
        let b = r.read_raw_bytes(L2 * 4)?;
        for c in b.chunks_exact(4) {
            fc1_biases.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        let w = r.read_raw_bytes(L2 * fc1_padded)?;
        let fc1_weights: Vec<i8> = w.iter().map(|&x| x as i8).collect();

        // fc_2: AffineTransform<2*L2 + 2*L3, 1>
        let b = r.read_raw_bytes(4)?;
        let fc2_biases = vec![i32::from_le_bytes([b[0], b[1], b[2], b[3]])];
        let w = r.read_raw_bytes(1 * fc2_padded)?;
        let fc2_weights: Vec<i8> = w.iter().map(|&x| x as i8).collect();

        stacks.push(LayerStack {
            fc_0: FcLayer {
                biases: fc0_biases,
                weights: fc0_weights,
                input_dims: L1,
                padded_input_dims: fc0_padded,
                output_dims: L2,
            },
            fc_1: FcLayer {
                biases: fc1_biases,
                weights: fc1_weights,
                input_dims: L2 * 2,
                padded_input_dims: fc1_padded,
                output_dims: L2,
            },
            fc_2: FcLayer {
                biases: fc2_biases,
                weights: fc2_weights,
                input_dims: L2 * 2 + L3 * 2,
                padded_input_dims: fc2_padded,
                output_dims: 1,
            },
        });
    }

    // ---- EOF check (network.cpp: stream.peek() == eof) ----
    if r.remaining() != 0 {
        return Err(format!(
            "{} trailing bytes after final layer stack (expected exact EOF)",
            r.remaining()
        ));
    }

    // Debug header words (first 16 u32s of the file).
    let mut raw = Vec::new();
    for k in 0..16 {
        if (k + 1) * 4 <= data.len() {
            raw.push(u32::from_le_bytes([
                data[k * 4],
                data[k * 4 + 1],
                data[k * 4 + 2],
                data[k * 4 + 3],
            ]));
        }
    }

    let header = NnueHeader {
        magic,
        version: magic,
        arch,
        description: description.clone(),
        raw,
    };

    Ok(NnueNetwork {
        header,
        data: data.to_vec(),
        parsed: ParsedNetwork {
            description,
            ft_biases,
            threat_weights,
            threat_psqt,
            pawn_pair_weights,
            pawn_pair_psqt,
            ft_weights,
            ft_psqt_weights,
            stacks,
            file_ft_hash,
            file_arch_hash: arch,
        },
    })
}

/// Load and fully validate an NNUE file from disk.
pub fn load_nnue<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<NnueNetwork> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    parse(&buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ---------------------------------------------------------------------------
// Feature extraction (HalfKAv2_hm-style helper retained from prior code)
// ---------------------------------------------------------------------------

const NUM_PIECES: usize = 12;
const NUM_SQUARES: usize = 64;

#[inline]
fn orient_square(sq: usize, perspective: usize) -> usize {
    if perspective == WHITE {
        sq ^ 56
    } else {
        sq
    }
}

pub fn halfka_hm_features(board: &Board) -> Vec<usize> {
    let mut features = Vec::new();
    let perspective = board.side;
    let king_sq = board.kingsq[perspective];
    let ksq = orient_square(king_sq, perspective);
    for pc in 0..6 {
        for c in 0..2 {
            let mut bb = board.pieces[c][pc];
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                if pc == KING && c == perspective {
                    continue;
                }
                let sqo = orient_square(sq, perspective);
                let p = if c == perspective { pc } else { pc + 6 };
                let idx = ksq * NUM_SQUARES * NUM_PIECES + sqo * NUM_PIECES + p;
                features.push(idx);
            }
        }
    }
    features.sort_unstable();
    features.dedup();
    features
}

// ---------------------------------------------------------------------------
// Evaluation wrapper
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct NnueEval {
    pub net: Option<NnueNetwork>,
}

impl Default for NnueEval {
    fn default() -> Self {
        NnueEval { net: None }
    }
}

impl NnueEval {
    pub fn new_from_file(path: &str) -> Option<Self> {
        if let Ok(net) = load_nnue(path) {
            Some(NnueEval { net: Some(net) })
        } else {
            None
        }
    }

    // Real forward pass (bit-exact with Stockfish's net) when a network is
    // loaded, mapped to centipawns by the same pipeline `eval::evaluate` uses.
    // Without a net this stays a material count (side-to-move perspective) so
    // callers keep working.
    pub fn evaluate(&self, board: &Board) -> i32 {
        if let Some(net) = &self.net {
            return crate::eval::evaluate_with_net(&net.parsed, board);
        }
        use crate::eval::PIECE_VALUES;
        let mut score = 0i32;
        for c in 0..2 {
            for pc in 0..6 {
                score += (board.pieces[c][pc].count_ones() as i32)
                    * PIECE_VALUES[pc]
                    * if c == 0 { 1 } else { -1 };
            }
        }
        if board.side == 1 {
            -score
        } else {
            score
        }
    }
}
