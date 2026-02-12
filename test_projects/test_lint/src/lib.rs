// test_api/src/lib.rs
pub mod api {
    pub struct Bad; // should emit signal
    pub trait Good {} // allowed
}
