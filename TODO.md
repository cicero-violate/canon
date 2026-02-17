# TODO

- [ ] Finalize MemoryEngine layout sizing so the full Canon IR fits within available GPU memory. Current `PAGES_PER_SLOT`/`SEGMENT_SLOTS` values still blow up (`chunk overflow` / allocator failures) when serializing the manifest.
- [ ] Persist slot assignments in the manifest and teach `MemoryIrBuilder` / `MemoryIrReader` to reuse those slots instead of hashing, so the reader can reconstruct artifacts unambiguously.
- [ ] Re-run `canon ingest`, `canon diagnose`, and `canon materialize` end-to-end once the layout is stable, and capture their outputs to confirm diagnostics/materialization consume the MemoryEngine `.tlog` / checkpoint.
