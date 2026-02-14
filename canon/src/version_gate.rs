use crate::CanonicalIr;

pub fn enforce_version_gate(ir: &CanonicalIr) -> Result<(), Box<dyn std::error::Error>> {
    let runtime_version = env!("CARGO_PKG_VERSION");
    if ir.version_contract.current == runtime_version
        || ir
            .version_contract
            .compatible_with
            .iter()
            .any(|v| v == runtime_version)
    {
        Ok(())
    } else {
        Err(format!(
            "Canon version `{}` incompatible with runtime `{}`",
            ir.version_contract.current, runtime_version
        )
        .into())
    }
}
