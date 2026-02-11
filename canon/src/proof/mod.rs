pub mod smt_bridge;

pub use smt_bridge::{
    SmtCertificate, SmtError, attach_function_proofs, verify_function_postconditions,
};
