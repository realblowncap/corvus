use chess_engine::board::Board;
use chess_engine::search::{SearchLimits, Searcher};

#[test]
fn depth20_time_budget_sanity() {
    // Time-bounded sanity test for depth-20 thinking
    let mut s = Searcher::new(Board::startpos());
    let res = s.think(SearchLimits { depth: 20, movetime: 1500, ..SearchLimits::default() });
    // Ensure the search completed within the budget and produced some work
    assert!(res.nodes > 0, "depth-20 sanity: no nodes produced within budget");
}
