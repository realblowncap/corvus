// Shared helper for tests that need the (large, gitignored) SF19 network
// file `nn-1a298aa575a0.nnue`.  The file is never checked out on CI, so
// those tests skip themselves when it can't be found instead of failing.
use chess_engine::nnue::NnueNetwork;

const NET_NAME: &str = "nn-1a298aa575a0.nnue";

/// Try `CORVUS_EVAL_FILE`, then the package root (cargo runs integration
/// tests with the manifest directory as cwd), then the original development
/// path.  Returns `None` when the file is nowhere to be found.
pub fn load_net() -> Option<NnueNetwork> {
    let candidates = [
        std::env::var("CORVUS_EVAL_FILE").ok(),
        Some(NET_NAME.to_string()),
        Some(format!("/home/avi/Documents/corvus/{NET_NAME}")),
    ];
    for path in candidates.into_iter().flatten() {
        if let Ok(net) = chess_engine::nnue::load_nnue(&path) {
            return Some(net);
        }
    }
    eprintln!(
        "NNUE network file not found (tried $CORVUS_EVAL_FILE, ./{NET_NAME}); \
         set CORVUS_EVAL_FILE or place the net in the repo root to enable these tests"
    );
    None
}
