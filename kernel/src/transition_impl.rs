use crate::{
    kernel::{Kernel, KernelError},
    transition::KernelTransition,
};
use database::{delta::Delta, primitives::StateHash, proofs::CommitProof};

impl KernelTransition for Kernel {
    fn genesis(&self) -> StateHash {
        self.current_root_hash()
    }

    fn step(&self, state: StateHash, delta: Delta) -> Result<(StateHash, CommitProof), KernelError> {
        let current = self.current_root_hash();
        if current != state {
            return Err(KernelError::StateMismatch { expected: current, provided: state });
        }

        let delta_hash = self.register_delta(delta);
        let judgment = self.transition_judgment(state, delta_hash);
        let admission = self.admit_execution(&judgment)?;
        let commit = self.commit_delta(&admission, &delta_hash)?;

        Ok((self.current_root_hash(), commit))
    }
}
