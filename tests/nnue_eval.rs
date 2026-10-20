use chess_engine::board::Board;
use chess_engine::nnue::NnueEval;

#[test]
fn eval_startpos_basic() {
    let b = Board::startpos();
    let evaler = NnueEval::default();
    let s = evaler.evaluate(&b);
    println!("eval={}", s);
    // should be small (close to 0)
    assert!(s.abs() < 100);
}
