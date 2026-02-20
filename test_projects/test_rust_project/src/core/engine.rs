use crate::data::model::User;

pub fn run() {
    let user = User::new("Cheese", 42);
    println!("User: {:?}", user);
}
