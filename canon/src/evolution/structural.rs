use crate::ir::{
    CanonicalIr, Delta, DeltaPayload, ImplBlock, Module, ModuleEdge, Struct, StructKind, Trait,
    Visibility,
};

use super::EvolutionError;

pub(super) fn apply_structural_delta(
    ir: &mut CanonicalIr,
    delta: &Delta,
) -> Result<(), EvolutionError> {
    match &delta.payload {
        Some(DeltaPayload::AddModule {
            module_id,
            name,
            visibility,
            description,
        }) => {
            if ir.modules.iter().any(|m| m.id == *module_id) {
                return Err(EvolutionError::DuplicateArtifact(module_id.clone()));
            }
            ir.modules.push(Module {
                id: module_id.clone(),
                name: name.clone(),
                visibility: *visibility,
                description: description.clone(),
                pub_uses: Vec::new(),
                constants: Vec::new(),
                type_aliases: Vec::new(),
                statics: Vec::new(),
                attributes: Vec::new(),
            });
        }
        Some(DeltaPayload::AddStruct {
            module,
            struct_id,
            name,
        }) => {
            if ir.structs.iter().any(|s| s.id == *struct_id) {
                return Err(EvolutionError::DuplicateArtifact(struct_id.clone()));
            }
            ensure_module_exists(ir, module)?;
            ir.structs.push(Struct {
                id: struct_id.clone(),
                name: name.clone(),
                module: module.clone(),
                visibility: Visibility::Private,
                derives: Vec::new(),
                doc: None,
                kind: StructKind::Normal,
                fields: vec![],
                history: vec![],
            });
        }
        Some(DeltaPayload::AddField { struct_id, field }) => {
            let structure = ir
                .structs
                .iter_mut()
                .find(|s| s.id == *struct_id)
                .ok_or_else(|| EvolutionError::UnknownStruct(struct_id.clone()))?;
            if structure.fields.iter().any(|f| f.name == field.name) {
                return Err(EvolutionError::DuplicateArtifact(
                    field.name.as_str().to_string(),
                ));
            }
            structure.fields.push(field.clone());
        }
        Some(DeltaPayload::UpdateStructVisibility {
            struct_id,
            visibility,
        }) => {
            let structure = ir
                .structs
                .iter_mut()
                .find(|s| s.id == *struct_id)
                .ok_or_else(|| EvolutionError::UnknownStruct(struct_id.clone()))?;
            structure.visibility = *visibility;
        }
        Some(DeltaPayload::RemoveField {
            struct_id,
            field_name,
        }) => {
            ensure_field_exists(ir, struct_id, field_name.as_str())?;
            if let Some(structure) = ir.structs.iter_mut().find(|s| s.id == *struct_id) {
                structure
                    .fields
                    .retain(|field| field.name.as_str() != field_name.as_str());
            }
        }
        Some(DeltaPayload::RenameArtifact {
            kind,
            old_id,
            new_id,
        }) => match kind.as_str() {
            "module" => rename_module(ir, old_id, new_id)?,
            "struct" => rename_struct(ir, old_id, new_id)?,
            "function" => rename_function(ir, old_id, new_id)?,
            _ => return Err(EvolutionError::UnknownDelta(delta.id.clone())),
        },
        Some(DeltaPayload::AddTrait {
            module,
            trait_id,
            name,
        }) => {
            if ir.traits.iter().any(|t| t.id == *trait_id) {
                return Err(EvolutionError::DuplicateArtifact(trait_id.clone()));
            }
            ensure_module_exists(ir, module)?;
            ir.traits.push(Trait {
                id: trait_id.clone(),
                name: name.clone(),
                module: module.clone(),
                visibility: Visibility::Private,
                generic_params: Vec::new(),
                supertraits: Vec::new(),
                associated_types: Vec::new(),
                associated_consts: Vec::new(),
                functions: vec![],
            });
        }
        Some(DeltaPayload::AddTraitFunction { trait_id, function }) => {
            let target_trait = ir
                .traits
                .iter_mut()
                .find(|t| t.id == *trait_id)
                .ok_or_else(|| EvolutionError::UnknownTrait(trait_id.clone()))?;
            if target_trait.functions.iter().any(|f| f.id == function.id) {
                return Err(EvolutionError::DuplicateArtifact(function.id.clone()));
            }
            target_trait.functions.push(function.clone());
        }
        Some(DeltaPayload::AddImpl {
            module,
            impl_id,
            struct_id,
            trait_id,
        }) => {
            if ir.impl_blocks.iter().any(|i| i.id == *impl_id) {
                return Err(EvolutionError::DuplicateArtifact(impl_id.clone()));
            }
            ensure_module_exists(ir, module)?;
            ensure_struct_exists(ir, struct_id)?;
            ensure_trait_exists(ir, trait_id)?;
            ir.impl_blocks.push(ImplBlock {
                id: impl_id.clone(),
                module: module.clone(),
                struct_id: struct_id.clone(),
                trait_id: trait_id.clone(),
                functions: vec![],
            });
        }
        Some(DeltaPayload::AddFunction {
            function_id,
            impl_id,
            signature,
        }) => {
            if ir.functions.iter().any(|f| f.id == *function_id) {
                return Err(EvolutionError::DuplicateArtifact(function_id.clone()));
            }
            let block_index = ir
                .impl_blocks
                .iter()
                .position(|i| i.id == *impl_id)
                .ok_or_else(|| EvolutionError::UnknownImpl(impl_id.clone()))?;
            let trait_id = ir.impl_blocks[block_index].trait_id.clone();
            let module = ir.impl_blocks[block_index].module.clone();
            ensure_trait_function_exists(ir, &trait_id, &signature.trait_function)?;
            ir.functions.push(crate::ir::Function {
                receiver: crate::ir::Receiver::None,
                id: function_id.clone(),
                name: signature.name.clone(),
                module,
                impl_id: impl_id.clone(),
                trait_function: signature.trait_function.clone(),
                visibility: signature.visibility,
                doc: signature.doc.clone(),
                lifetime_params: signature.lifetime_params.clone(),
                is_async: signature.is_async,
                is_unsafe: signature.is_unsafe,
                generics: signature.generics.clone(),
                where_clauses: signature.where_clauses.clone(),
                inputs: signature.inputs.clone(),
                outputs: signature.outputs.clone(),
                deltas: vec![],
                contract: crate::ir::FunctionContract {
                    total: true,
                    deterministic: true,
                    explicit_inputs: true,
                    explicit_outputs: true,
                    effects_are_deltas: true,
                },
                metadata: crate::ir::FunctionMetadata::default(),
            });
            ir.impl_blocks[block_index]
                .functions
                .push(crate::ir::ImplFunctionBinding {
                    trait_fn: signature.trait_function.clone(),
                    function: function_id.clone(),
                });
        }
        Some(DeltaPayload::AddModuleEdge {
            from,
            to,
            rationale,
        }) => {
            ensure_module_exists(ir, from)?;
            ensure_module_exists(ir, to)?;
            if ir
                .module_edges
                .iter()
                .any(|edge| &edge.source == from && &edge.target == to)
            {
                return Err(EvolutionError::DuplicateArtifact(format!("{from}->{to}")));
            }
            ir.module_edges.push(ModuleEdge {
                source: from.clone(),
                target: to.clone(),
                rationale: rationale.clone(),
                imported_types: Vec::new(),
            });
        }
        Some(DeltaPayload::AddCallEdge { caller, callee }) => {
            ensure_function_exists(ir, caller)?;
            ensure_function_exists(ir, callee)?;
            if ir
                .call_edges
                .iter()
                .any(|edge| edge.caller == *caller && edge.callee == *callee)
            {
                return Err(EvolutionError::DuplicateArtifact(format!(
                    "{caller}->{callee}"
                )));
            }
            ir.call_edges.push(crate::ir::CallEdge {
                id: format!("call:{}:{}", caller, callee),
                caller: caller.clone(),
                callee: callee.clone(),
                rationale: "delta-applied".to_string(),
            });
        }
        Some(DeltaPayload::AttachExecutionEvent {
            execution_id,
            event,
        }) => {
            let record = ir
                .executions
                .iter_mut()
                .find(|e| e.id == *execution_id)
                .ok_or_else(|| EvolutionError::UnknownExecution(execution_id.clone()))?;
            record.events.push(event.clone());
        }
        Some(DeltaPayload::UpdateFunctionAst { function_id, ast }) => {
            let function = ir
                .functions
                .iter_mut()
                .find(|f| f.id == *function_id)
                .ok_or_else(|| EvolutionError::UnknownFunction(function_id.clone()))?;
            function.metadata.ast = Some(ast.clone());
        }
        Some(DeltaPayload::UpdateFunctionInputs {
            function_id,
            inputs,
        }) => {
            let function = ir
                .functions
                .iter_mut()
                .find(|f| f.id == *function_id)
                .ok_or_else(|| EvolutionError::UnknownFunction(function_id.clone()))?;
            function.inputs = inputs.clone();
        }
        Some(DeltaPayload::UpdateFunctionOutputs {
            function_id,
            outputs,
        }) => {
            let function = ir
                .functions
                .iter_mut()
                .find(|f| f.id == *function_id)
                .ok_or_else(|| EvolutionError::UnknownFunction(function_id.clone()))?;
            function.outputs = outputs.clone();
        }
        Some(DeltaPayload::AddEnum {
            module,
            enum_id,
            name,
            visibility,
        }) => {
            if ir.enums.iter().any(|e| e.id == *enum_id) {
                return Err(EvolutionError::DuplicateArtifact(enum_id.clone()));
            }
            ensure_module_exists(ir, module)?;
            ir.enums.push(crate::ir::EnumNode {
                id: enum_id.clone(),
                name: name.clone(),
                module: module.clone(),
                visibility: *visibility,
                variants: vec![],
                history: vec![],
            });
        }
        Some(DeltaPayload::AddEnumVariant { enum_id, variant }) => {
            let target = ir
                .enums
                .iter_mut()
                .find(|e| e.id == *enum_id)
                .ok_or_else(|| EvolutionError::UnknownEnum(enum_id.clone()))?;
            if target.variants.iter().any(|v| v.name == variant.name) {
                return Err(EvolutionError::DuplicateArtifact(
                    variant.name.as_str().to_string(),
                ));
            }
            target.variants.push(variant.clone());
        }
        Some(DeltaPayload::RecordReward { record }) => {
            if ir.reward_deltas.iter().any(|r| r.id == record.id) {
                return Err(EvolutionError::DuplicateArtifact(record.id.clone()));
            }
            ir.reward_deltas.push(record.clone());
        }
        _ => {}
    }
    Ok(())
}

fn rename_module(ir: &mut CanonicalIr, old_id: &str, new_id: &str) -> Result<(), EvolutionError> {
    let module = ir
        .modules
        .iter_mut()
        .find(|m| m.id == *old_id)
        .ok_or_else(|| EvolutionError::UnknownModule(old_id.to_string()))?;
    module.id = new_id.to_owned();
    for structure in &mut ir.structs {
        if structure.module == *old_id {
            structure.module = new_id.to_owned();
        }
    }
    for tr in &mut ir.traits {
        if tr.module == *old_id {
            tr.module = new_id.to_owned();
        }
    }
    for block in &mut ir.impl_blocks {
        if block.module == *old_id {
            block.module = new_id.to_owned();
        }
    }
    for function in &mut ir.functions {
        if function.module == *old_id {
            function.module = new_id.to_owned();
        }
    }
    for edge in &mut ir.module_edges {
        if edge.source == *old_id {
            edge.source = new_id.to_owned();
        }
        if edge.target == *old_id {
            edge.target = new_id.to_owned();
        }
    }
    Ok(())
}

fn rename_struct(ir: &mut CanonicalIr, old_id: &str, new_id: &str) -> Result<(), EvolutionError> {
    let structure = ir
        .structs
        .iter_mut()
        .find(|s| s.id == *old_id)
        .ok_or_else(|| EvolutionError::UnknownStruct(old_id.to_string()))?;
    structure.id = new_id.to_owned();
    for block in &mut ir.impl_blocks {
        if block.struct_id == *old_id {
            block.struct_id = new_id.to_owned();
        }
    }
    for function in &mut ir.functions {
        if function.module == *old_id {
            function.module = new_id.to_owned();
        }
    }
    Ok(())
}

fn rename_function(ir: &mut CanonicalIr, old_id: &str, new_id: &str) -> Result<(), EvolutionError> {
    let function = ir
        .functions
        .iter_mut()
        .find(|f| f.id == *old_id)
        .ok_or_else(|| EvolutionError::UnknownFunction(old_id.to_string()))?;
    function.id = new_id.to_owned();
    for block in &mut ir.impl_blocks {
        for binding in &mut block.functions {
            if binding.function == *old_id {
                binding.function = new_id.to_owned();
            }
        }
    }
    for edge in &mut ir.call_edges {
        if edge.caller == *old_id {
            edge.caller = new_id.to_owned();
        }
        if edge.callee == *old_id {
            edge.callee = new_id.to_owned();
        }
    }
    Ok(())
}

fn ensure_module_exists(ir: &CanonicalIr, module: &str) -> Result<(), EvolutionError> {
    if ir.modules.iter().any(|m| m.id == module) {
        Ok(())
    } else {
        Err(EvolutionError::UnknownModule(module.to_string()))
    }
}

fn ensure_struct_exists(ir: &CanonicalIr, struct_id: &str) -> Result<(), EvolutionError> {
    if ir.structs.iter().any(|s| s.id == struct_id) {
        Ok(())
    } else {
        Err(EvolutionError::UnknownStruct(struct_id.to_string()))
    }
}

fn ensure_field_exists(
    ir: &CanonicalIr,
    struct_id: &str,
    field_name: &str,
) -> Result<(), EvolutionError> {
    let structure = ir
        .structs
        .iter()
        .find(|s| s.id == struct_id)
        .ok_or_else(|| EvolutionError::UnknownStruct(struct_id.to_string()))?;
    if structure
        .fields
        .iter()
        .any(|f| f.name.as_str() == field_name)
    {
        Ok(())
    } else {
        Err(EvolutionError::UnknownField {
            struct_id: struct_id.to_string(),
            field: field_name.to_string(),
        })
    }
}

fn ensure_trait_exists(ir: &CanonicalIr, trait_id: &str) -> Result<(), EvolutionError> {
    if ir.traits.iter().any(|t| t.id == trait_id) {
        Ok(())
    } else {
        Err(EvolutionError::UnknownTrait(trait_id.to_string()))
    }
}

fn ensure_trait_function_exists(
    ir: &CanonicalIr,
    trait_id: &str,
    trait_fn: &str,
) -> Result<(), EvolutionError> {
    let tr = ir
        .traits
        .iter()
        .find(|t| t.id == trait_id)
        .ok_or_else(|| EvolutionError::UnknownTrait(trait_id.to_string()))?;
    if tr.functions.iter().any(|f| f.id == trait_fn) {
        Ok(())
    } else {
        Err(EvolutionError::UnknownTraitFunction(trait_fn.to_string()))
    }
}

fn ensure_function_exists(ir: &CanonicalIr, function_id: &str) -> Result<(), EvolutionError> {
    if ir.functions.iter().any(|f| f.id == function_id) {
        Ok(())
    } else {
        Err(EvolutionError::UnknownFunction(function_id.to_string()))
    }
}
