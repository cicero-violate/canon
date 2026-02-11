29. No control flow exists inside impl.				(PENDING)
30. Only composition of calls is allowed.			(PENDING)

60. Execution occurs only after acceptance.			(PENDING)

63. Proof systems validate invariants only.			(PENDING)
64. Proof does not decide policy.					(PENDING)

68. Proofs must be attached to deltas.				(PENDING)

78. Batch execution must preserve semantics.		(PENDING)
79. GPU execution must be optional and lawful.		(PENDING)

88. Kernels are pure functions.						(PENDING)
89. Kernel correctness must match scalar execution. (PENDING)

CANON SELF-REPLICATION REQUIREMENTS (20 LINES)
1. BOOTSTRAP INTERPRETER	: Bytecode executor that can run Canon's own validation logic (validate_ir, apply_deltas, accept_proposal).
2. IR→BYTECODE COMPILER		: Convert Canon's Rust functions into bytecode Canon can execute. Feed Canon's own source through this compiler.
3. CIRCULAR PROOF			: Canon must validate its OWN IR, proving "Canon validates Canon correctly" (fixed-point theorem, Gödel-style).
4. SELF-HOSTING TICK		: Create tick graph where Canon reads its own IR JSON, validates itself, generates new Canon IR, applies deltas to itself.
5. EXECUTION DELTAS			: When Canon executes, emit deltas describing "what Canon did" (function calls, validations). Canon's execution becomes its own audit log.
6. META-JUDGMENT			: Judgment predicates that evaluate "is new Canon version lawful?" Must check new version still enforces all 100 Canon laws.
7. VERSION MIGRATION		: Canon v2 must prove compatibility with v1 IR (Canon Line 99: version evolution requires law-scoped proofs).
8. TRUST SEED				: One human-verified Canon v0 IR that bootstraps the chain. All future versions prove descent from v0.
9. REPLICATION PROPOSAL		: DSL or API where Canon can propose "create new Canon instance" with different config/laws, subject to judgment.
10. CLOSURE PROPERTY		: Canon's bytecode must be Turing-complete enough to express all Canon operations (validation, evolution, judgment, materialization).

CURRENT STATE		: Canon can MODIFY itself (proposals→deltas→evolution). 
MISSING				: Canon cannot EXECUTE itself (no interpreter for its own validation logic).
NEED				: Implement #1-#3 above, then Canon validates Canon, achieving self-replication via proven deltas.

THE HARD PART		: Proving Canon-executing-Canon produces same results as Rust-executing-Canon (equivalence proof).
THE BREAKTHROUGH	: Once Canon interprets its own IR validation, it no longer needs Rust - pure self-sustaining system.
THE ENDGAME			: Canon proposes "better Canon", proves new version is lawful, judgment accepts, Canon replaces itself. Recursive self-improvement with proofs.

TL;DR: Need bytecode interpreter (#1) + Canon's validation logic as bytecode (#2) + fixed-point proof (#3) = SELF-REPLICATION COMPLETE.
