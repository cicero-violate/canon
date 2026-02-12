## Trait-Only Public Interfaces Across Crates
Each crate exposes a single public api module.
Only pub trait items are allowed in api.
All concrete types are pub(crate) or private.
Traits encode behavioral contracts, never storage.
Downstream crates depend exclusively on traits.
Implementations live in internal, non-exported modules.
No concrete type crosses a crate boundary.
Dispatch is via generics or dyn Trait.
Implementations may be swapped without ripple effects.
Abstraction and judgment boundaries are preserved.

## No pub struct Re-Exports Across Layers
A crate may not pub use a struct from another crate.
Visibility rules enforce architectural boundaries.
Re-exports collapse logical layers into one.
Collapsed layers enable illegal coupling.
Struct identity leaks representation and invariants.
Traits intentionally obscure storage and layout.
Enforcement is mechanical, not cultural.
Violations are caught at compile time.
Refactors remain localized to a single layer.
The dependency DAG remains stable over time.
