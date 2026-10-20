// Validates the sf_19 NNUE parser against the real network file and prints
// the feature-transformer dimensions.
mod common;

use chess_engine::nnue;

#[test]
fn parse_and_validate_sf19_nnue() {
    let Some(net) = common::load_net() else {
        return;
    };

    // Header checks.
    assert_eq!(net.header.magic, nnue::NNUE_VERSION, "version magic");
    assert_eq!(
        net.header.arch,
        nnue::network_hash(),
        "header archHash == Network::hash"
    );
    println!("version    = 0x{:08X}", net.header.magic);
    println!("archHash   = 0x{:08X}", net.header.arch);
    println!("FT hash    = 0x{:08X}", net.parsed.file_ft_hash);
    println!("description= {:?}", net.parsed.description);

    // Cross-checks implied by Network::hash == FT hash ^ arch hash.
    assert_eq!(
        net.header.arch,
        net.parsed.file_ft_hash ^ nnue::arch_hash(),
        "archHash == FT hash ^ arch hash"
    );
    assert_eq!(net.parsed.file_ft_hash, nnue::ft_hash(), "FT hash value");

    // Structural checks.
    assert_eq!(net.parsed.stacks.len(), nnue::LAYER_STACKS, "8 layer stacks");
    assert_eq!(net.parsed.ft_biases.len(), nnue::HALF_DIMENSIONS);
    assert_eq!(net.parsed.ft_weights.len(), nnue::HALF_DIMENSIONS * nnue::PSQ_DIM);
    assert_eq!(
        net.parsed.ft_psqt_weights.len(),
        nnue::PSQ_T_BUCKETS * nnue::PSQ_DIM
    );
    assert_eq!(
        net.parsed.threat_weights.len(),
        nnue::THREAT_INPUT_DIM * nnue::HALF_DIMENSIONS
    );
    assert_eq!(
        net.parsed.pawn_pair_weights.len(),
        nnue::PAIR_INPUT_DIM * nnue::HALF_DIMENSIONS
    );

    // Print the feature-transformer dimensions.
    let d = net.parsed.feature_transformer_dims();
    println!("\n--- Feature transformer dimensions ---");
    println!("input_dim        = {}", d.input_dim);
    println!("output_dim       = {}", d.output_dim);
    println!("half_dimensions  = {}", d.half_dimensions);
    println!("psq_dim          = {}", d.psq_dim);
    println!("threat_dim       = {}", d.threat_dim);
    println!("pair_dim         = {}", d.pair_dim);
    println!("psqt_buckets     = {}", d.psqt_buckets);
    println!(
        "layer stack: fc_0 {}->{}, fc_1 {}->{}, fc_2 {}->{}",
        net.parsed.stacks[0].fc_0.input_dims,
        net.parsed.stacks[0].fc_0.output_dims,
        net.parsed.stacks[0].fc_1.input_dims,
        net.parsed.stacks[0].fc_1.output_dims,
        net.parsed.stacks[0].fc_2.input_dims,
        net.parsed.stacks[0].fc_2.output_dims,
    );
}
