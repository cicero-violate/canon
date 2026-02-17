mod ast;
mod functions;
mod impls;
mod module;
mod type_defs;

use super::error::Violation;
use super::helpers::Indexes;
use crate::ir::CanonicalIr;

pub fn check<'a>(ir: &'a CanonicalIr, idx: &Indexes<'a>, violations: &mut Vec<Violation>) {
    module::check_version_proofs(ir, idx, violations);
    module::check_module_edges(ir, violations);
    type_defs::check_structs(ir, idx, violations);
    type_defs::check_enums(ir, idx, violations);
    type_defs::check_traits(ir, idx, violations);
    impls::check_impls(ir, idx, violations);
    functions::check_functions(ir, idx, violations);
    ast::check_ast_node_kinds(ir, violations);
}
