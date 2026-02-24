# Canon Refactor Pipeline (Consistent)

## State objects

| State    | Definition                                   |
|----------|----------------------------------------------|
| Source₀  | on-disk workspace                            |
| Model₀   | model_ir(Source₀)  [authoritative]           |
| Model₁   | Model₀ + Mutations      [authoritative]      |
| Model₁'  | Model₁ + Derivations   [authoritative]       |
| Plan₁    | projection(Model₁')      [text plan]         |
| Source₁  | emit(Plan₁)                                  |
| Model₂   | model_ir(Source₁)                            |

## Ownership (single-responsibility)

| Step | Description                                            | Project         | Input            | Output                                                                         |
|------+--------------------------------------------------------+-----------------+------------------+--------------------------------------------------------------------------------|
|    1 | Build Model₀ from Source₀                              | capture → model |                  | Model₀ (canonical IR; complete structural state)                               |
|    2 | Validate Model₀ invariants (structural + safety gates) | kernel          |                  | validated Model₀ or error                                                      |
|    3 | Apply refactor ops as pure graph mutations -> Model₁   | transform       | validated Model₀ | Model₁ (only "primary" edges mutated; NO text edits)                           |
|    4 | Recompute derived edges/relationships -> Model₁'       | analysis        | Model₁           | Model₁' (adds/refreshes derived edges: calls/imports/vis/order/contains, etc)  |
|    5 | Project to Plan₁ and emit Source₁                      | projection      | Model₁'          | Plan₁ + Source₁ (deterministic full regen)                                     |
|    6 | Rebuild Model₂ from Source₁                            | capture → model |                  | Model₂                                                                         |
|    7 | Fixpoint verify: Model₂ == Model₁' (NOT Model₁)        | analysis        |                  | equal => ok; else => diff report targeting projection/capture/derivation rules |
|    8 | Commit Source₁ if verified                             | kernel          |                  | commit gate + audit log                                                        |

## Corrected arrows

| Flow                                                                                                                                                                                                                 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| capture (read) -> model (canonical IR) -> kernel (invariants) -> transform (mutate) -> analysis (derive) -> projection (emit) -> capture (read) -> model (canonical IR) -> analysis (verify) -> kernel (commit gate) |
