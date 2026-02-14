echo "=== src/agent/capability.rs (tail -60) ==="
bat -n src/agent/capability.rs | tail -60

echo "=== src/agent/reward.rs (tail -60) ==="
bat -n src/agent/reward.rs | tail -60

echo "=== src/runtime/tick_executor/planning.rs (head -50) ==="
bat -n src/runtime/tick_executor/planning.rs | head -50

echo "=== src/ir/policy.rs ==="
bat -n src/ir/policy.rs

echo "=== src/agent/mod.rs ==="
bat -n src/agent/mod.rs

echo "=== src/runtime/mod.rs (head -20) ==="
bat -n src/runtime/mod.rs | head -20
