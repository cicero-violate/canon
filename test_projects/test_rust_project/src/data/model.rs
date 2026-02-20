#[derive(Debug)]
pub struct User {
    pub name: String,
    pub score: u32,
}

impl User {
    pub fn new(name: &str, score: u32) -> Self {
        Self {
            name: name.to_string(),
            score,
        }
    }
}
