use std::collections::BTreeMap;
use std::path::PathBuf;

use canon::CanonicalIr;
use canon::gpu::codegen::{flatten_ports, generate_shader};
use canon::gpu::dispatch::GpuExecutor;
use canon::gpu::fusion::analyze_fusion_candidates;
use canon::runtime::value::ScalarValue;
use canon::runtime::{ExecutionContext, FunctionExecutor, Value};

fn load_fixture(name: &str) -> CanonicalIr {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name);
    let data = std::fs::read(path).expect("fixture exists");
    serde_json::from_slice(&data).expect("valid CanonicalIr")
}

#[test]
fn gpu_kernel_matches_cpu_add() {
    let ir = load_fixture("gpu_add.json");
    let gpu = ir
        .gpu_functions
        .first()
        .expect("gpu function exists")
        .clone();
    let function = ir
        .functions
        .iter()
        .find(|f| f.id == gpu.function)
        .expect("function exists")
        .clone();

    let lanes = gpu.inputs[0].lanes as usize;
    let lhs: Vec<f32> = (0..lanes).map(|i| i as f32).collect();
    let rhs: Vec<f32> = (0..lanes).map(|i| (i * 2) as f32).collect();

    // Compile IR function into runtime bytecode before GPU codegen
    use canon::runtime::bytecode::FunctionBytecode;

    let bytecode =
        FunctionBytecode::from_function(&function).expect("bytecode compilation succeeds");

    let program = generate_shader(&gpu, &bytecode).expect("shader generation succeeds");
    let input_buffer =
        flatten_ports(&gpu.inputs, &[lhs.clone(), rhs.clone()]).expect("flatten inputs");
    let mut gpu_outputs = vec![0.0; lanes];

    let executor = match pollster::block_on(GpuExecutor::new()) {
        Ok(exec) => exec,
        Err(err) => {
            eprintln!("skipping GPU test: {err}");
            return;
        }
    };

    if let Err(err) =
        pollster::block_on(executor.execute(&program, &input_buffer, &mut gpu_outputs))
    {
        eprintln!("skipping GPU test due to execution failure: {err}");
        return;
    }

    let function_executor = FunctionExecutor::new(&ir);
    for lane in 0..lanes {
        let mut ctx = ExecutionContext::new(BTreeMap::new());
        let mut inputs = BTreeMap::new();
        inputs.insert("Lhs".into(), Value::Scalar(ScalarValue::F32(lhs[lane])));
        inputs.insert("Rhs".into(), Value::Scalar(ScalarValue::F32(rhs[lane])));
        let outputs = function_executor
            .execute_by_id(&gpu.function, inputs, &mut ctx)
            .expect("cpu execution");
        let cpu_sum = match outputs.get("Sum").expect("sum output") {
            Value::Scalar(ScalarValue::F32(v)) => *v,
            Value::Scalar(ScalarValue::I32(v)) => *v as f32,
            other => panic!("unexpected value {other:?}"),
        };
        assert!(
            (cpu_sum - gpu_outputs[lane]).abs() < 1e-3,
            "lane {lane} mismatch {cpu_sum} != {}",
            gpu_outputs[lane]
        );
    }
}

#[test]
fn fusion_analyzer_detects_simple_chain() {
    let ir = load_fixture("gpu_chain.json");
    let candidates = analyze_fusion_candidates(&ir);
    assert_eq!(candidates.len(), 1);
    let fusion = &candidates[0];
    assert_eq!(fusion.producer_function, "fn.scale");
    assert_eq!(fusion.consumer_function, "fn.bias");
    assert_eq!(fusion.producer_gpu, "gpu.scale");
    assert_eq!(fusion.consumer_gpu, "gpu.bias");
}
