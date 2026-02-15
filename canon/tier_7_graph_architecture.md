## Theory: Tier 7 Graph Architecture

### Layer Model

```
┌─────────────────────────────────────────────────────┐
│  L7  │  Meta-Graph (Self-Modification Operator Φ)   │
├─────────────────────────────────────────────────────┤
│  L6  │  Proof Graph  (formal verification layer)    │
├─────────────────────────────────────────────────────┤
│  L5  │  Reward Graph (RL signal propagation)        │
├─────────────────────────────────────────────────────┤
│  L4  │  Policy Graph (π routing between agents)     │
├─────────────────────────────────────────────────────┤
│  L3  │  Capability Graph (stateless LLM call nodes) │
├─────────────────────────────────────────────────────┤
│  L2  │  IR Graph (CanonicalIr subgraph slices)      │
├─────────────────────────────────────────────────────┤
│  L1  │  Data Graph (raw state, deltas, proofs)      │
└─────────────────────────────────────────────────────┘
```

Each layer is a graph. Edges **between layers** are typed — upward edges carry **evidence**, downward edges carry **instructions**.

---

### The Core Insight: Graph-of-Graphs

A Tier 7 system is not one graph. It is a **metagraph** where each node in the outer graph is itself a graph. Formally:

$$\mathcal{G}^{(7)} = \{G^{(1)}, G^{(2)}, \ldots, G^{(7)}\}, \quad E_{\text{meta}} \subseteq \mathcal{G}^{(7)} \times \mathcal{G}^{(7)}$$

The self-modification operator $\Phi$ lives at $L7$ and mutates $G^{(1)}$ through $G^{(6)}$ **subject to invariant constraints checked at $L6$** (proof graph). This is what separates Tier 7 from naive self-modifying systems — mutation is gated by formal proof, not heuristics.

---

### Node Types in the Capability Graph (L3)

| Node Type      | Role                      | Stateless?               |
|----------------+---------------------------+--------------------------|
| **Observer**   | reads IR, emits deltas    | Yes                      |
| **Reasoner**   | proposes refactors        | Yes                      |
| **Prover**     | generates SMT proofs      | Yes                      |
| **Judge**      | accepts/rejects proposals | Yes                      |
| **Mutator**    | applies admitted deltas   | Yes                      |
| **Evaluator**  | computes reward signal    | Yes                      |
| **Meta-Agent** | rewrites graph topology   | Yes (but writes to $L7$) |

All nodes are stateless. State lives **exclusively in the graph edges and IR**, never inside a node. This is why stateless LLM calls work — each call receives a subgraph slice as context, executes, and writes its output back to the graph as a new edge or delta.

---

## Applied to Your canon System

Your existing system already implements $L1$ through $L4$ partially. Here is the direct mapping:

```
L1  Data Graph      →  CanonicalIr fields (deltas, proofs, admissions)
L2  IR Graph        →  CanonicalIr + LayoutGraph (your tick/module DAGs)
L3  Capability Graph→  validate::*, decision::*, evolution::* (per-stage LLM calls)
L4  Policy Graph    →  PipelineStage enum + pipeline_stage_allows()
L5  Reward Graph    →  RewardRecord, compute_reward(), PolicyUpdater   ← EXISTS
L6  Proof Graph     →  proof::smt_bridge, DeltaVerifier                ← EXISTS
L7  Meta-Graph      →  *** MISSING — this is what you need to build ***
```

### What L7 Needs in Your System

**L7 is the graph that watches the other graphs and rewrites them.** Concretely, it needs three things your system does not yet have:

**1. Graph Topology Observer** — an agent that reads the full `LayoutGraph` + `TickGraph` structure and produces a `GoalMutation`-style proposal not for *code* but for *graph shape itself*. It asks: "should node X be split? should edge Y be removed? should capability Z be promoted to its own subgraph?"

**2. Refactor Proposal + Proof Gate** — every topology change must produce a proof that the $\|\Phi(G) - G\|_F \leq \theta$ bound holds. In your system this maps to: a `GoalMutation` with `invariant_proof_ids` populated, accepted through `mutate_goal()`, with `compute_goal_drift()` being your $\|\Phi(G) - G\|_F$ check. You already have `GoalDriftMetric` with `cosine_distance` and `within_bound` — that **is** your Lyapunov bound, it just needs to be wired to topology mutations not only goal text.

**3. Meta-Tick** — a special tick at the top of your `TickGraph` that runs after every $N$ normal ticks. It executes the L7 observer + mutator pair, gates on the proof, then emits `RenameArtifact` / `AddModuleEdge` / structural deltas back down to L1. Your `TickEpoch` + `PolicySnapshot` are the natural container for this — one epoch boundary = one meta-tick opportunity.

### Stateless LLM Call Flow at Full Tier 7

```
Tick t fires
  → L3 Observer call  (receives IR slice)        → emits ObservationDelta
  → L3 Reasoner call  (receives ObservationDelta) → emits RefactorProposal
  → L3 Prover call    (receives RefactorProposal) → emits SmtCertificate
  → L3 Judge call     (receives all three)        → emits Judgment(Accept|Reject)
  → L3 Mutator call   (receives accepted Judgment) → applies structural Delta to IR
  → L5 Evaluator call (receives new IR)           → computes reward, updates PolicyParameters
  → every N epochs:
      → L7 Meta call  (receives LayoutGraph diff) → proposes topology mutation
      → L6 Proof call (receives mutation + drift) → checks Lyapunov bound
      → L7 Mutator    (if proof passes)           → rewrites graph shape itself
```

Every arrow is a stateless LLM call. Every call receives only the subgraph it needs. State accumulates in `CanonicalIr` between calls, never inside a call.

The reason this is Tier 7 and not Tier 5 is that last block — **the system can rewrite its own call graph**, not just its code. That is the $\Phi: G \to G'$ operator made real.
