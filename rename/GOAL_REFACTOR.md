You need a **full module-graph rewrite pass**, not heuristics.

---

# Correct Complete Architecture

After structural move + rebuild:

```
MoveSet = { symbol_id → (old_module, new_module) }
```

Now perform **three deterministic passes**.

---

# PASS 1 — Canonical Reference Rewrite

For every file in project:

1. Traverse full AST:

   * ItemUse
   * TypePath
   * ExprPath
   * TraitBound
   * Impl blocks
   * Where clauses
   * Macro paths

2. Resolve every path via alias graph → canonical SymbolId.

3. If `symbol_id ∈ MoveSet`:

   * Replace path segments with `new_module_path + symbol_name`.

No string matching.
No path guessing.
Everything resolution-based.

This ensures:

* All call sites updated.
* All type references updated.
* All trait bounds updated.
* All impl headers updated.

---

# PASS 2 — Scope Rehydration (Dependency Injection)

For each destination module:

1. Parse moved AST.
2. Collect all referenced SymbolIds inside body.
3. For each referenced symbol:

   * If defined outside destination module
   * AND not fully-qualified
   * AND not already in scope
   * AND not brought in via glob
   * THEN inject canonical `use` statement.

Injection must:

* Respect module nesting.
* Preserve sorted import ordering.
* Deduplicate.
* Avoid shadowing conflicts.

---

# PASS 3 — Orphaned Import Cleanup

After rewrite + injection:

For every file:

1. Re-resolve every `use` tree.
2. If imported SymbolId has no remaining references → remove it.
3. If glob no longer necessary → reduce.

---

# Required Invariants

* All path resolution must be SymbolId-based.
* All edits must occur via syn AST transformation.
* After every major pass → full rebuild.
* No partial registry reuse.
* No stale alias graph usage.

---

# Full Deterministic Pipeline

```
1. Raw span structural move
2. Reparse
3. Rebuild registry + alias graph
4. Canonical reference rewrite
5. Reparse
6. Rebuild registry
7. Scope rehydration
8. Reparse
9. Rebuild registry
10. Orphan cleanup
11. Final reparse
12. Format
```

No shortcuts.

---

If you implement this as a single deterministic transformation pipeline, gaps disappear.

Gaps only exist when:

* resolution is skipped
* partial updates are applied
* registry state is reused
* or rewrite is string-based

You are now building a full module-graph transformer.

That is the correct long-term architecture.
