use rename::rename::core::apply_rename_with_map;
use std::collections::HashMap;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_root =
        Path::new("/workspace/ai_sandbox/canon_workspace/canon");

    let mut map = HashMap::new();

    // Fully-qualified canonical paths
    map.insert(
        "crate::test_project_dir::out::src::Core".to_string(),
        "core".to_string(),
    );
    map.insert(
        "crate::test_project_dir::out::src::Delta".to_string(),
        "delta".to_string(),
    );
    map.insert(
        "crate::test_project_dir::out::src::Lint".to_string(),
        "lint".to_string(),
    );
    map.insert(
        "crate::test_project_dir::out::src::Parse".to_string(),
        "parse".to_string(),
    );
    map.insert(
        "crate::test_project_dir::out::src::Report".to_string(),
        "report".to_string(),
    );
    map.insert(
        "crate::test_project_dir::out::src::Test".to_string(),
        "test".to_string(),
    );

    apply_rename_with_map(
        project_root,
        &map,
        false,
        None,
    )?;

    println!("Renamed target modules.");
    Ok(())
}
