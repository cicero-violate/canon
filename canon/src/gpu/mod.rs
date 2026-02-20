//! GPU execution support for Canon bytecode.
//!
//! Converts math-only bytecode into WGSL compute shaders and dispatches them via wgpu.

pub mod codegen;
pub mod dispatch;
pub mod fusion;

pub use codegen::{generate_shader, GpuProgram};
pub use dispatch::{GpuExecutor, GpuExecutorError};
pub use fusion::{analyze_fusion_candidates, FusionCandidate};
