RUSTFLAGS="-A warnings" cargo run --bin canon -- ingest   --src ./canon   --semantic-out ./canon.semantic.json
RUSTFLAGS="-A warnings" cargo run --bin canon -- diagnose   ./canon.semantic.json
RUSTFLAGS="-A warnings" cargo run --bin canon -- materialize   ./canon.semantic.json   target/materialized_project   --layout ./canon.semantic.layout.json

# cd target/materialized_project
# cargo build 2>&1 | tee materialized_build.txt
# grep -o "E[0-9]\{4\}" materialized_build.txt | sort | uniq -c | sort -nr
# grep -c "error\[" materialized_build.txt
# sed -n '1,200p' materialized_build.txt
# tail -n 200 materialized_build.txt
