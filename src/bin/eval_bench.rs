use chess_engine::board::Board;
use chess_engine::eval::evaluate;
use std::time::Instant;

fn main() {
    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/ppp2ppp/2n1pn2/q3b3/2BP4/P1N2N2/1PP2PPP/R1BQK2R w KQkq - 0 1",
        "rnbq1rk1/ppp2ppp/4pn2/3p4/2PP4/2NBP3/PP3PPP/R1BQK2R b KQ - 0 1",
    ];
    for f in fens {
        let b = Board::from_fen(f).unwrap();
        let n = 200_000u32;
        let t = Instant::now();
        let mut acc = 0i64;
        for _ in 0..n {
            acc += evaluate(&b) as i64;
        }
        let dt = t.elapsed().as_secs_f64();
        println!(
            "{dt:.3}s for {n} evals = {:.0} ns/eval (acc={acc}, eval={})",
            dt / n as f64 * 1e9,
            evaluate(&b)
        );
    }
}