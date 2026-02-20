//! Abstract interpretation framework.
//!
//! Variables:
//!   α: concrete values -> abstract domain D
//!   γ: abstract domain D -> P(concrete values)   (Galois connection)
//!
//! Equations:
//!   α(γ(d)) ⊑ d          (reductive)
//!   γ(α(S)) ⊇ S          (extensive)
//!   fixed point: X = f(X) computed via Kleene iteration with widening

pub trait AbstractDomain: Clone + PartialEq {
    fn bottom() -> Self;
    fn top() -> Self;
    fn join(&self, other: &Self) -> Self;
    fn widen(&self, other: &Self) -> Self {
        self.join(other)
    }
    fn is_bottom(&self) -> bool {
        *self == Self::bottom()
    }
}

/// Interval domain: [lo, hi] over i64. ⊥ = empty, ⊤ = [-∞, +∞].
#[derive(Clone, PartialEq, Debug)]
pub enum Interval {
    Bottom,
    Range(i64, i64),
}

impl AbstractDomain for Interval {
    fn bottom() -> Self {
        Interval::Bottom
    }
    fn top() -> Self {
        Interval::Range(i64::MIN, i64::MAX)
    }
    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, x) | (x, Interval::Bottom) => x.clone(),
            (Interval::Range(a, b), Interval::Range(c, d)) => Interval::Range((*a).min(*c), (*b).max(*d)),
        }
    }
    fn widen(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, x) | (x, Interval::Bottom) => x.clone(),
            (Interval::Range(a, b), Interval::Range(c, d)) => {
                let lo = if *c < *a { i64::MIN } else { *a };
                let hi = if *d > *b { i64::MAX } else { *b };
                Interval::Range(lo, hi)
            }
        }
    }
}
