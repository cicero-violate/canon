//! WGSL shader generation from Canon bytecode.
//!
//! Only math-only bytecode is allowed (no branching, Calls, or deltas).

use std::collections::HashMap;

use crate::ir::{Function, GpuFunction, VectorPort};
use crate::runtime::bytecode::{FunctionBytecode, Instruction};
use crate::runtime::value::{ScalarValue, Value};

/// Generated GPU program plus layout metadata.
#[derive(Debug, Clone)]
pub struct GpuProgram {
    pub shader: String,
    pub workgroup_size: u32,
    pub lanes: u32,
    pub input_offsets: Vec<u32>,
    pub output_offsets: Vec<u32>,
}

/// Convert a math-only bytecode to WGSL compute shader.
pub fn generate_shader(gpu: &GpuFunction, function: &Function) -> Result<GpuProgram, String> {
    if gpu.outputs.len() != 1 {
        return Err(format!(
            "gpu kernel `{}` must have exactly one output for now",
            gpu.id
        ));
    }
    let bytecode = FunctionBytecode::from_function(function).map_err(|err| err.to_string())?;
    let lanes = lane_count(gpu)?;
    let input_offsets = compute_offsets(&gpu.inputs, lanes);
    let output_offsets = compute_offsets(&gpu.outputs, lanes);

    let mut lines = Vec::new();
    lines.push("@group(0) @binding(0) var<storage, read> input_buffer: array<f32>;".into());
    lines.push("@group(0) @binding(1) var<storage, read_write> output_buffer: array<f32>;".into());
    lines.push(
        "@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {"
            .into(),
    );
    lines.push("  let lane = id.x;".into());
    lines.push(format!("  if (lane >= {}u) {{ return; }}", lanes));
    let binding_insert_at = lines.len();
    lines.push("  var stack: array<f32, 64>;".into());
    lines.push("  var sp: u32 = 0u;".into());

    let mut binding_vars: HashMap<String, String> = HashMap::new();
    let mut binding_order = Vec::new();

    for inst in &bytecode.instructions {
        match inst {
            Instruction::LoadInput(name) => {
                let idx = gpu
                    .inputs
                    .iter()
                    .position(|port| port.name.as_str() == name)
                    .ok_or_else(|| format!("input `{name}` missing in gpu kernel {}", gpu.id))?;
                let offset = input_offsets[idx];
                lines.push(format!("  stack[sp] = input_buffer[{offset}u + lane];"));
                lines.push("  sp = sp + 1u;".into());
            }
            Instruction::LoadConst(value) => {
                let literal = literal_to_f32(value)
                    .ok_or_else(|| format!("literal `{value:?}` unsupported in GPU kernels"))?;
                lines.push(format!("  stack[sp] = {literal};"));
                lines.push("  sp = sp + 1u;".into());
            }
            Instruction::StoreBinding(name) => {
                let entry = binding_vars.entry(name.to_string()).or_insert_with(|| {
                    let sym = format!("binding_{}", sanitize_ident(name.as_str()));
                    binding_order.push(sym.clone());
                    sym
                });
                lines.push(format!("  {entry} = stack[sp - 1u];"));
            }
            Instruction::LoadBinding(name) => {
                let var = binding_vars.get(name.as_str()).ok_or_else(|| {
                    format!(
                        "binding `{name}` referenced before store in gpu kernel {}",
                        gpu.id
                    )
                })?;
                lines.push(format!("  stack[sp] = {var};"));
                lines.push("  sp = sp + 1u;".into());
            }
            Instruction::Add => push_binary_op(&mut lines, "+"),
            Instruction::Sub => push_binary_op(&mut lines, "-"),
            Instruction::Mul => push_binary_op(&mut lines, "*"),
            Instruction::Return => break,
            other => {
                return Err(format!(
                    "instruction `{other:?}` is not supported in GPU kernels"
                ));
            }
        }
    }

    for (idx, var) in binding_order.iter().enumerate() {
        lines.insert(binding_insert_at + idx, format!("  var {var}: f32 = 0.0;"));
    }

    lines.push("  sp = sp - 1u;".into());
    lines.push("  let result = stack[sp];".into());
    let output_offset = output_offsets[0];
    lines.push(format!(
        "  output_buffer[{output_offset}u + lane] = result;"
    ));
    lines.push("}".into());

    Ok(GpuProgram {
        shader: lines.join("\n"),
        workgroup_size: 64,
        lanes,
        input_offsets,
        output_offsets,
    })
}

fn lane_count(gpu: &GpuFunction) -> Result<u32, String> {
    let source = gpu
        .inputs
        .first()
        .or_else(|| gpu.outputs.first())
        .ok_or_else(|| format!("gpu kernel `{}` has no ports", gpu.id))?;
    let lanes = source.lanes;
    if lanes == 0 {
        return Err(format!("gpu kernel `{}` lanes must be non-zero", gpu.id));
    }
    for port in gpu.inputs.iter().chain(gpu.outputs.iter()) {
        if port.lanes != lanes {
            return Err(format!(
                "gpu kernel `{}` mixes lane counts ({} vs {})",
                gpu.id, port.lanes, lanes
            ));
        }
    }
    Ok(lanes)
}

fn compute_offsets(ports: &[VectorPort], lanes: u32) -> Vec<u32> {
    ports
        .iter()
        .enumerate()
        .map(|(idx, _)| (idx as u32) * lanes)
        .collect()
}

/// Flatten per-port lane data into a single buffer in struct-of-array layout.
pub fn flatten_ports(ports: &[VectorPort], data: &[Vec<f32>]) -> Result<Vec<f32>, String> {
    if ports.len() != data.len() {
        return Err("port data length mismatch".into());
    }
    if ports.is_empty() {
        return Ok(Vec::new());
    }
    let lanes = ports[0].lanes as usize;
    for (idx, port) in ports.iter().enumerate() {
        if port.lanes as usize != lanes {
            return Err(format!(
                "port `{}` lanes mismatch (expected {lanes}, found {})",
                port.name, port.lanes
            ));
        }
        if data[idx].len() != lanes {
            return Err(format!(
                "port `{}` data length {} != lanes {}",
                port.name,
                data[idx].len(),
                lanes
            ));
        }
    }
    let mut buffer = Vec::with_capacity(ports.len() * lanes);
    for (idx, _) in ports.iter().enumerate() {
        for lane in 0..lanes {
            buffer.push(data[idx][lane]);
        }
    }
    Ok(buffer)
}

fn push_binary_op(lines: &mut Vec<String>, op: &str) {
    lines.push("  sp = sp - 1u;".into());
    lines.push("  let rhs = stack[sp];".into());
    lines.push("  sp = sp - 1u;".into());
    lines.push("  let lhs = stack[sp];".into());
    lines.push(format!("  stack[sp] = lhs {op} rhs;"));
    lines.push("  sp = sp + 1u;".into());
}

fn literal_to_f32(value: &Value) -> Option<f32> {
    match value {
        Value::Scalar(ScalarValue::F32(v)) => Some(*v),
        Value::Scalar(ScalarValue::I32(v)) => Some(*v as f32),
        Value::Scalar(ScalarValue::U32(v)) => Some(*v as f32),
        _ => None,
    }
}

fn sanitize_ident(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
