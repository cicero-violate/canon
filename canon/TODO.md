No. You do not need SMT and you do not need probe questions in the agent loop.

Here is the precise answer.

---

## What Kind of Problem This Is

This is **static graph validation** — not theorem proving, not dynamic probing.

$$
\text{IR} = \text{directed graph } G = (V, E)
$$

$$
\text{validator} = \text{set of structural predicates } \{P_1, P_2, \ldots, P_n\}
$$

$$
\text{violation} \iff \exists\, P_k : P_k(G) = \text{false}
$$

Every predicate in this validator is of the form:

$$
\forall\, x \in G : \text{ref}(x) \in G
$$

That is: every reference must resolve. This is **referential integrity** — the same class of constraint a database foreign key enforces. No SMT needed. No sampling needed. It is fully decidable by graph traversal.

---

## What the Agent Loop Actually Needs

The algorithm I described is not a reasoning problem at inference time. It is a **program** that runs before the LLM is called. The LLM only needs to be called for the parts that require judgment. Everything else is mechanical.

Split the work into three tiers:

$$
\text{Tier 1: Mechanical} \quad \text{(execute as code, never send to LLM)}
$$

- Parse violations into structured records
- Count and cluster by rule
- Read the validator source to extract the predicate for each rule
- Trace which IR field the predicate reads
- Trace which ingest function writes that field
- Classify the defect type (missing emit, wrong value, wrong direction, missing context)

$$
\text{Tier 2: Judgment} \quad \text{(send to LLM once per root cause)}
$$

- Given: the validator predicate, the ingest function, the defect class
- Ask: what is the minimal correct fix?
- The LLM writes the patch

$$
\text{Tier 3: Verification} \quad \text{(execute as code, never send to LLM)}
$$

- Apply the patch
- Compile
- Re-run ingest + validate
- Measure $\Delta V$
- If $\Delta V > 0$: proceed
- If $\Delta V = 0$: the LLM chose the wrong site — send back with the measurement
- If $\Delta V < 0$: revert, send back with the new violations introduced

---

## What This Means for Your Agent Loop

Your agent loop should look like this:

```
loop:
  violations = run_ingest_and_validate()
  if violations.empty(): break

  clusters = cluster_by_rule(violations)
  root_causes = trace_root_causes(clusters, validator_source, ingest_source)

  for each root_cause:
    patch = llm_call(root_cause)   ← ONE call per root cause, not per violation
    apply(patch)
    compile()
    delta = measure(violations)
    if delta <= 0: revert_and_retry(root_cause)
```

The `trace_root_causes` function is pure code — it reads source files, parses ASTs, follows field references. It produces a structured brief for the LLM. The LLM receives a precise, bounded problem, not a raw error dump.

---

## The Key Architectural Decision

**Do not send raw violations to the LLM.** Send the root cause brief.

The difference:

```
Raw (wrong):
  "183 violations, here they are: ..."

Brief (correct):
  "Rule 27: function.impl_id references an ImplBlock that is never emitted.
   Cause: build_standalone() returns ImplMapping::Standalone(funcs) 
   but never pushes an ImplBlock into impl_blocks.
   Fix site: functions/mod.rs line 108, syn_conv.rs build_standalone().
   Defect class: missing emit."
```

The brief is produced entirely by code. The LLM only answers: given this brief, write the patch. That is a bounded, deterministic task with a verifiable output.

The intelligence of the system lives in `trace_root_causes`. That is where you invest engineering effort. The LLM is just the patch writer.
