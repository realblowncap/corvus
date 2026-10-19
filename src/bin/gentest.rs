use chess_engine::board::Board;
use chess_engine::eval::evaluate;
use chess_engine::search::Searcher;

fn main() {
    // after 19.Nxe6, black to move
    let fen = "r4rk1/2q5/3bNpp1/p1np1p2/Pp1Q4/4P3/1PP2PPP/2KR1B1R b - - 0 1";
    let b = Board::from_fen(fen).unwrap();
    println!("static black-persp eval = {}", evaluate(&b));
    let mut s = Searcher::new(b.clone());
    s.dbg_root = true;
    let (mv, sc) = s.search_root(2, -100000, 100000);
    println!("black d2: best={} score={}", mv.uci(), sc);

    // after 19...Nxe6, white to move — what does white think?
    let fen2 = "r4rk1/2q5/3bnpp1/p2p1p2/Pp1Q4/4P3/1PP2PPP/2KR1B1R w - - 0 1";
    let b2 = Board::from_fen(fen2).unwrap();
    let mut s2 = Searcher::new(b2.clone());
    s2.dbg_root = true;
    let (mv2, sc2) = s2.search_root(2, -100000, 100000);
    println!("white d2: best={} score={}", mv2.uci(), sc2);
}