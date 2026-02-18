use memory_engine::{
    delta::{Delta, Source},
    epoch::Epoch,
    primitives::{DeltaID, PageID},
    MemoryEngine,
    MemoryEngineConfig,
    MemoryTransition,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tlog_path = std::env::temp_dir().join("memory_engine_minimal_transition.tlog");
    if tlog_path.exists() {
        std::fs::remove_file(&tlog_path)?;
    }

    let engine = MemoryEngine::new(MemoryEngineConfig { tlog_path })?;
    let mut state = engine.genesis();

    let payload = vec![42u8; 4096];
    let mask = vec![true; payload.len()];
    let delta = Delta::new_dense(
        DeltaID(1),
        PageID(0),
        Epoch(0),
        payload,
        mask,
        Source("examples/minimal_transition".into()),
    )?;

    let (next_state, proof) = engine.step(state, delta)?;

    println!("Previous root: {state:?}");
    println!("New root: {next_state:?}");
    println!("Commit proof state hash: {:?}", proof.state_hash);

    state = next_state;
    println!("Transition complete. Current root: {state:?}");

    Ok(())
}
