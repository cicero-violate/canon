pub mod smt_bridge;

pub use smt_bridge::{attach_function_proofs, verify_function_postconditions, SmtCertificate, SmtError};
