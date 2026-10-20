use chess_engine::board::Board;
use chess_engine::nnue;

#[test]
fn extract_startpos_features() {
    let b = Board::startpos();
    let feats = nnue::halfka_hm_features(&b);
    println!("num_features={}", feats.len());
    for i in 0..10.min(feats.len()) {
        println!("feat[{}]={}", i, feats[i]);
    }
    assert!(!feats.is_empty());
}
