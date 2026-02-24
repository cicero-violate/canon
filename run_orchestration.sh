rm -rf /tmp/test_emit
cargo run -p orchestration -- \
  test_projects/test_rust_project/model_ir.json \
  /tmp/test_emit

cd /tmp/test_emit && cargo build
cargo run
