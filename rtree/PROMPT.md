## 20-Line Prompt: Module Dependency Tree Visualizer

I'm building a Rust tool that visualizes module dependency graphs in two modes:

**Top-Down Mode**: Start from root modules (those with no incoming dependencies) and traverse downward following dependency edges. Show which modules each module depends on. This mode works correctly.

**Bottom-Up Mode**: Start from leaf modules (those with no outgoing dependencies) and traverse upward following reverse dependency edges. Show which modules depend on each leaf module, climbing up to root modules like `lib.rs`.

**Current Problem**: In bottom-up mode, shared dependency nodes (like `state/mod.rs` which is depended on by multiple leaf modules) appear multiple times in the output, but only the first occurrence shows its full subtree. Subsequent occurrences show the node with no children, creating a fragmented view.

**Desired Behavior**: 
- Each leaf module should show its complete path up to the root
- Shared nodes (like intermediate `mod.rs` files) should appear in multiple paths
- Only the first occurrence of a shared node should expand its subtree
- Subsequent occurrences should be marked as references (already expanded elsewhere)
- Tree should use proper indentation with `├──`, `└──`, and `│` characters
- Module names should be simplified (show last component only, e.g., `serialization` instead of `state::serialization`)
- Each node should display its file path (e.g., `<state/serialization.rs>`)

**Expected Output Structure**: Multiple trees, one per leaf, each showing the complete dependency chain from that leaf up to the root, with proper visual nesting and shared nodes handled gracefully.
