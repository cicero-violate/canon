pub type Result1231<T> = std::result::Result<T, String>;
pub fn compute_result(a: i32, b: i32) -> Result1231<i32> {
    if b == 0 { Err("division by zero".into()) } else { Ok(a / b) }
}
pub fn parse_result(input: &str) -> Result1231<i32> {
    input.parse::<i32>().map_err(|e| e.to_string())
}
pub fn nested_result(x: i32) -> Result1231<Result1231<i32>> {
    if x > 0 { Ok(Ok(x * 2)) } else { Ok(Err("negative value".into())) }
}
pub fn combine_results(a: &str, b: &str) -> Result1231<i32> {
    let x = parse_result(a)?;
    let y = parse_result(b)?;
    compute_result(x, y)
}
pub fn results_chain(a: &str, b: &str) -> Result1231<Result1231<i32>> {
    let combined = combine_results(a, b)?;
    nested_result(combined)
}
