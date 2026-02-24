use anyhow::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    let root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_dir().expect("failed to resolve current directory")
        });

    // Use unified rustc-backed pipeline
    let model = capture::capture(&root)?;

    #[cfg(feature = "hir")]
    {
        // Run full rustc capture (HIR + MIR)
        capture_rustc::run(&root)
            .map_err(|e| anyhow::anyhow!("rustc capture failed: {e}"))?;
    }

    println!("Capture OK");
    println!("Root: {}", model.crate_.root);
    println!(
        "Counts => modules={} structs={} traits={} impls={} fns={}",
        model.modules.len(),
        model.structs.len(),
        model.traits.len(),
        model.impls.len(),
        model.functions.len()
    );

    #[cfg(feature = "hir")]
    {
        println!(
            "HIR => items={} exprs={} types={} paths={}",
            model.hir_items.len(),
            model.hir_exprs.len(),
            model.hir_types.len(),
            model.hir_paths.len(),
        );

        println!(
            "MIR => bodies={}",
            model.mir_bodies.len(),
        );
    }

    Ok(())
}
