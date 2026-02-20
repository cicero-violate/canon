use std::collections::BTreeMap;
use std::path::PathBuf;

use canon::gpu::codegen::{flatten_ports, generate_shader};
use canon::gpu::dispatch::GpuExecutor;
use canon::runtime::value::ScalarValue;
use canon::runtime::{ExecutionContext, FunctionExecutor, Value};
use canon::CanonicalIr;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pollster::block_on;

fn load_fixture(name: &str) -> CanonicalIr {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join(name);
    let data = std::fs::read(path).expect("fixture exists");
    serde_json::from_slice(&data).expect("valid CanonicalIr")
}

fn bench_bytecode_interpreter(c: &mut Criterion) {
    let ir = load_fixture("simple_add.json");
    let executor = FunctionExecutor::new(&ir);
    let function_id = "fn.add".to_string();
    let mut group = c.benchmark_group("bytecode_interpreter");
    group.bench_function(BenchmarkId::from_parameter("add"), |b| {
        b.iter(|| {
            let mut ctx = ExecutionContext::new(BTreeMap::new());
            let mut inputs = BTreeMap::new();
            inputs.insert("Lhs".into(), Value::Scalar(ScalarValue::I32(5)));
            inputs.insert("Rhs".into(), Value::Scalar(ScalarValue::I32(7)));
            let outputs = executor.execute_by_id(&function_id, inputs, &mut ctx).expect("bytecode execution");
            assert!(outputs.contains_key("Sum"));
        });
    });
    group.finish();
}

fn bench_gpu_dispatch(c: &mut Criterion) {
    let ir = load_fixture("gpu_add.json");
    let gpu = match ir.gpu_functions.first() {
        Some(g) => g.clone(),
        None => return,
    };
    let function = match ir.functions.iter().find(|f| f.id == gpu.function) {
        Some(f) => f.clone(),
        None => return,
    };
    let program = match generate_shader(&gpu, &function) {
        Ok(shader) => shader,
        Err(err) => {
            eprintln!("skipping gpu bench: {err}");
            return;
        }
    };
    let lanes = gpu.inputs[0].lanes as usize;
    let lhs: Vec<f32> = (0..lanes).map(|i| i as f32).collect();
    let rhs: Vec<f32> = (0..lanes).map(|i| (i * 2) as f32).collect();
    let Ok(input_buffer) = flatten_ports(&gpu.inputs, &[lhs.clone(), rhs.clone()]) else {
        return;
    };
    let mut outputs = vec![0.0f32; lanes];
    let executor = match block_on(GpuExecutor::new()) {
        Ok(exec) => exec,
        Err(err) => {
            eprintln!("skipping gpu bench: {err}");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_dispatch");
    group.bench_function(BenchmarkId::from_parameter("add"), |b| {
        b.iter(|| {
            let mut gpu_out = outputs.clone();
            if let Err(err) = block_on(executor.execute(&program, &input_buffer, &mut gpu_out)) {
                panic!("gpu dispatch failed: {err}");
            }
        });
    });
    group.finish();
}

fn runtime_profiles(c: &mut Criterion) {
    bench_bytecode_interpreter(c);
    bench_gpu_dispatch(c);
}

criterion_group!(benches, runtime_profiles);
criterion_main!(benches);
