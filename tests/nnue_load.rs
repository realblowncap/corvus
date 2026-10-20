mod common;

#[test]
fn load_sf19_nnue() {
    let Some(net) = common::load_net() else {
        return;
    };
    println!("magic={:x} ver={:x} arch={:x}", net.header.magic, net.header.version, net.header.arch);
    println!("raw[0..16]: {:?}", &net.header.raw[0..16.min(net.header.raw.len())]);
    // File loaded successfully
    assert!(net.data.len() > 10_000_000);
}
