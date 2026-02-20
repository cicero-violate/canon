//! # Analysis
//!
//! Static analysis domains over Rust programs, ingested via `compiler_capture`.
//!
//! Each domain operates on `GraphSnapshot` / `GraphDelta` wire types from `database`,
//! populated by `compiler_capture::frontends::rustc::RustcFrontend`.
//!
//! ## Domains
//!
//! - `call_graph`     — call graph construction and reachability
//! - `cfg`            — control flow graph over basic blocks
//! - `dataflow`       — reaching definitions, use/def chains
//! - `usedef`         — use-before-define, def-use chains
//! - `alias`          — alias and points-to analysis
//! - `escape`         — escape analysis
//! - `lifetime`       — lifetime region analysis
//! - `taint`          — taint source/sink tracking
//! - `deadcode`       — unreachable symbols and branches
//! - `scc`            — strongly connected components in call graph
//! - `interproc`      — interprocedural summaries
//! - `concurrency`    — race condition and lock analysis
//! - `effect`         — side effect tracking
//! - `abstract_interp`— abstract interpretation framework
//! - `types`          — type-level analysis

pub mod abstract_interp;
pub mod alias;
pub mod call_graph;
pub mod cfg;
pub mod concurrency;
pub mod dataflow;
pub mod deadcode;
pub mod effect;
pub mod escape;
pub mod interproc;
pub mod lifetime;
pub mod scc;
pub mod taint;
pub mod types;
pub mod usedef;
