use rustc_session::declare_lint;

// Enforce that modules named `api` only expose `pub trait` items.
declare_lint! {
    pub API_TRAITS_ONLY,
    Warn,
    "modules named `api` may only expose public traits"
}
