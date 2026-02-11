# Canon Version Evolution Contract

Canonical IR now requires every document to declare an explicit version contract so
audit logs can explain how a given `meta.version` relates to prior releases.

## Artifacts

```rust
VersionContract {
  current: String,              // must match CanonicalMeta.version
  compatible_with: Vec<String>, // previous Canon versions this IR can replay
  migration_proofs: Vec<ProofId>, // proof obligations justifying the upgrade
}
```

Rules:

1. `current` must equal `meta.version`.
2. `compatible_with` entries must be unique. Empty lists are allowed when this is
   the first canonical release.
3. `migration_proofs` must contain at least one proof, and each referenced proof
   must exist and carry `ProofScope::Law`.
4. Proofs describing the upgrade must include their evidence (URI + hash) so
   replay can inspect the contract.

These invariants are enforced by `CanonRule::VersionEvolution`. Whenever the
canonical law revision changes, attach a new proof that cites the reasoning behind
the upgrade. The proof serves as an immutable boundary for future migrations.
