Define Patch Proposal Protocol (PPP).
Restrict AI agents to suggest structured diffs only (no direct file writes).
Accept apply_patch outputs as proposals, not authoritative actions.
Create a Patch Queue for incoming AI proposals.
Disallow any client library from running apply_patch directly to disk.
Ensure structured patch format conforms to your DSL semantics.
Tag each patch with origin metadata (agent ID, prompt, timestamp).
Forward proposals to the Canonical Gate Logic.
The gate enforces rule sets and invariants.
Canon rejects any patch violating semantics.
Canon logs every decision deterministically.
Canon annotates accepted patches with proof of validation.
Upon acceptance, emit a verified patch object.
Verified patches enter the Approved Patch Registry.
Hand-off to a controlled patch applier process.
The applier runs tests before application.
If tests fail, send back to Canon with diagnostics.
On pass, persist patch in the versioned codebase.
Push changelog events into your event stream.
Publish metadata for replay safety and audit.
Optionally trigger CI/T step suites after apply.
Do not circumvent jobs (no direct shell apply).
Make all AI proposal responses auditable.
Maintain invariant: only Canon-sanctioned patches get applied.
Monitor for double patch or redundant diff strategies.
Implement rollback mechanisms for rejected apply events.
Log patch lineage for future traceability.
Integrate human review for sensitive deltas if required.
Periodically review gate rule logic for drift.
Enforce that structural semantics always originate from Canon
