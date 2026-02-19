#![cfg(feature = "rustc_frontend")]
use super::frontend_context::FrontendMetadata;
use super::node_builder::ensure_node;
use crate::rename::core::symbol_id::normalize_symbol_id_with_crate;
use crate::state::builder::{EdgePayload, KernelGraphBuilder};
use crate::state::graph::EdgeKind;
use crate::state::ids::NodeId;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::mir::{self, BasicBlock, TerminatorKind};
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::Span;
use serde::Serialize;
use std::collections::HashMap;
/// Captures MIR-derived information for a single function (CFG + calls).
pub(super) fn capture_function<'tcx>(
    builder: &mut KernelGraphBuilder,
    tcx: TyCtxt<'tcx>,
    local_def: LocalDefId,
    function_node: NodeId,
    cache: &mut HashMap<DefId, NodeId>,
    metadata: &FrontendMetadata,
) {
    let body = tcx.optimized_mir(local_def);
    let mut bb_nodes: HashMap<BasicBlock, NodeId> = HashMap::new();
    let crate_name = tcx.crate_name(local_def.to_def_id().krate).to_string();
    let function_name = normalize_symbol_id_with_crate(
        &tcx.def_path_str(local_def.to_def_id()),
        Some(&crate_name),
    );
    let function_key = format!("{:?}", local_def.to_def_id());
    for (bb_idx, bb_data) in body.basic_blocks.iter_enumerated() {
        let bb_label = format!("{function_name}::bb{}", bb_idx.index());
        let bb_key = format!("{function_key}::bb{}", bb_idx.index());
        let statement_count = bb_data.statements.len();
        let mut payload = crate::state::builder::NodePayload::new(&bb_key, bb_label)
            .with_metadata("type", "basic_block")
            .with_metadata("index", bb_idx.index().to_string())
            .with_metadata("statement_count", statement_count.to_string())
            .with_metadata("terminator", format!("{:?}", bb_data.terminator().kind));
        if let Some(statements_json) = serialize_statements(bb_data) {
            payload = payload.with_metadata("statements", statements_json);
        }
        let bb_node = builder.add_node(payload).expect("bb node allocation");
        bb_nodes.insert(bb_idx, bb_node);
        let mut edge = EdgePayload::new(function_node, bb_node, EdgeKind::Contains);
        edge = edge.with_metadata("bb_index", bb_idx.index().to_string());
        let _ = builder.add_edge(edge);
    }
    for (bb_idx, bb_data) in body.basic_blocks.iter_enumerated() {
        let Some(&from_bb) = bb_nodes.get(&bb_idx) else {
            continue;
        };
        for successor in bb_data.terminator().successors() {
            if let Some(&to_bb) = bb_nodes.get(&successor) {
                let mut edge = EdgePayload::new(from_bb, to_bb, EdgeKind::ControlFlow);
                edge = edge.with_metadata("from_bb", bb_idx.index().to_string());
                edge = edge.with_metadata("to_bb", successor.index().to_string());
                let _ = builder.add_edge(edge);
            }
        }
    }
    collect_calls(builder, tcx, &body, function_node, local_def, cache, metadata);
    if let Some(cfg_json) = serialize_cfg_graph(tcx, &body) {
        let _ = builder
            .merge_node_metadata(function_node, [("cfg".to_string(), cfg_json)]);
    }
    if let Some(dfg_json) = serialize_dfg(tcx, &body) {
        let _ = builder
            .merge_node_metadata(function_node, [("dfg".to_string(), dfg_json)]);
    }
    if let Some(mir_dump) = serialize_mir_dump(&body) {
        let _ = builder
            .merge_node_metadata(function_node, [("mir".to_string(), mir_dump)]);
    }
    if let Some(metrics_json) = compute_metrics(tcx, local_def, &body) {
        let _ = builder
            .merge_node_metadata(function_node, [("metrics".to_string(), metrics_json)]);
    }
    if let Some(effects_json) = compute_effects(tcx, &body) {
        let _ = builder
            .merge_node_metadata(function_node, [("effects".to_string(), effects_json)]);
    }
}
fn collect_calls<'tcx>(
    builder: &mut KernelGraphBuilder,
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    caller_node: NodeId,
    local_def: LocalDefId,
    cache: &mut HashMap<DefId, NodeId>,
    metadata: &FrontendMetadata,
) {
    let source_map = tcx.sess.source_map();
    for (bb_idx, bb) in body.basic_blocks.iter_enumerated() {
        let terminator = bb.terminator();
        if let TerminatorKind::Call { func, target, .. } = &terminator.kind {
            let func_ty = func.ty(&body.local_decls, tcx);
            let span = call_span(tcx, terminator.source_info.span, local_def);
            if let ty::FnDef(callee_def, _) = *func_ty.kind() {
                let callee_node = ensure_node(builder, tcx, callee_def, cache, metadata);
                let loc = source_map.lookup_char_pos(span.lo());
                let mut edge = EdgePayload::new(
                    caller_node,
                    callee_node,
                    EdgeKind::Call,
                );
                let file_name = loc.file.name.prefer_local_unconditionally().to_string();
                edge = edge.with_metadata("call_site_file", file_name);
                edge = edge.with_metadata("call_site_line", loc.line.to_string());
                edge = edge.with_metadata("call_site_column", loc.col.0.to_string());
                edge = edge.with_metadata("call_site_bb", bb_idx.index().to_string());
                let dispatch = if target.is_some() { "static" } else { "unknown" };
                edge = edge.with_metadata("dispatch", dispatch);
                let hi = source_map.lookup_char_pos(span.hi());
                edge = edge.with_metadata("call_site_end_line", hi.line.to_string());
                edge = edge.with_metadata("call_site_end_column", hi.col.0.to_string());
                edge = edge.with_metadata("call_span", format!("{span:?}"));
                let _ = builder.add_edge(edge);
            }
        }
    }
}
fn call_span<'tcx>(_tcx: TyCtxt<'tcx>, span: Span, _local_def: LocalDefId) -> Span {
    span.with_hi(span.hi()).with_lo(span.lo())
}
fn serialize_statements<'tcx>(bb_data: &mir::BasicBlockData<'tcx>) -> Option<String> {
    #[derive(serde::Serialize)]
    struct StatementMetadata {
        index: usize,
        kind: String,
    }
    let statements: Vec<StatementMetadata> = bb_data
        .statements
        .iter()
        .enumerate()
        .map(|(idx, stmt)| StatementMetadata {
            index: idx,
            kind: format!("{:?}", stmt.kind),
        })
        .collect();
    serde_json::to_string(&statements).ok()
}
fn serialize_cfg_graph<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
) -> Option<String> {
    #[derive(Serialize)]
    struct SpanCapture {
        file: String,
        line: usize,
        column: usize,
    }
    #[derive(Serialize)]
    struct TerminatorCapture {
        kind: String,
        span: Option<SpanCapture>,
    }
    #[derive(Serialize)]
    struct BasicBlockCapture {
        id: String,
        statements: Vec<String>,
        terminator: TerminatorCapture,
        successors: Vec<String>,
    }
    #[derive(Serialize)]
    struct CfgCapture {
        blocks: Vec<BasicBlockCapture>,
        entry: String,
    }
    let source_map = tcx.sess.source_map();
    let mut blocks = Vec::new();
    for (bb_idx, bb_data) in body.basic_blocks.iter_enumerated() {
        let statements = bb_data
            .statements
            .iter()
            .map(|stmt| format!("{:?}", stmt.kind))
            .collect();
        let span = bb_data.terminator().source_info.span;
        let pos = source_map.lookup_char_pos(span.lo());
        let terminator = TerminatorCapture {
            kind: format!("{:?}", bb_data.terminator().kind),
            span: Some(SpanCapture {
                file: pos.file.name.prefer_local_unconditionally().to_string(),
                line: pos.line,
                column: pos.col.0,
            }),
        };
        let successors = bb_data
            .terminator()
            .successors()
            .map(|succ| format!("bb{}", succ.index()))
            .collect();
        blocks
            .push(BasicBlockCapture {
                id: format!("bb{}", bb_idx.index()),
                statements,
                terminator,
                successors,
            });
    }
    serde_json::to_string(
            &CfgCapture {
                blocks,
                entry: "bb0".into(),
            },
        )
        .ok()
}
fn serialize_dfg<'tcx>(_tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>) -> Option<String> {
    #[derive(Serialize)]
    struct LocalCapture {
        local: String,
        name: Option<String>,
        ty: String,
        kind: String,
    }
    #[derive(Serialize)]
    struct DfgCapture {
        locals: Vec<LocalCapture>,
        def_use_chains: Vec<serde_json::Value>,
    }
    let locals = body
        .local_decls
        .iter_enumerated()
        .map(|(local, decl)| LocalCapture {
            local: format!("{local:?}"),
            name: None,
            ty: format!("{:?}", decl.ty),
            kind: format!("{:?}", decl.mutability),
        })
        .collect();
    serde_json::to_string(
            &DfgCapture {
                locals,
                def_use_chains: Vec::new(),
            },
        )
        .ok()
}
fn serialize_mir_dump<'tcx>(body: &mir::Body<'tcx>) -> Option<String> {
    Some(format!("{body:?}"))
}
fn compute_metrics<'tcx>(
    tcx: TyCtxt<'tcx>,
    local_def: LocalDefId,
    body: &mir::Body<'tcx>,
) -> Option<String> {
    #[derive(Serialize)]
    struct MetricsCapture {
        cyclomatic_complexity: usize,
        lines_of_code: usize,
        statement_count: usize,
        basic_block_count: usize,
        call_count: usize,
        branch_count: usize,
    }
    let mut edge_count = 0;
    let mut call_count = 0;
    let mut branch_count = 0;
    let mut statement_count = 0;
    for bb in body.basic_blocks.iter() {
        statement_count += bb.statements.len();
        edge_count += bb.terminator().successors().count();
        if matches!(bb.terminator().kind, TerminatorKind::Call { .. }) {
            call_count += 1;
        }
        if matches!(
            bb.terminator().kind, TerminatorKind::SwitchInt { .. } | TerminatorKind::Goto
            { .. } | TerminatorKind::Assert { .. }
        ) {
            branch_count += 1;
        }
    }
    let bb_count = body.basic_blocks.len();
    let cyclomatic = edge_count.saturating_sub(bb_count).saturating_add(2);
    let source_map = tcx.sess.source_map();
    let span = tcx.def_span(local_def.to_def_id());
    let lo = source_map.lookup_char_pos(span.lo());
    let hi = source_map.lookup_char_pos(span.hi());
    let loc = hi.line.saturating_sub(lo.line) + 1;
    serde_json::to_string(
            &MetricsCapture {
                cyclomatic_complexity: cyclomatic,
                lines_of_code: loc,
                statement_count,
                basic_block_count: bb_count,
                call_count,
                branch_count,
            },
        )
        .ok()
}
fn compute_effects<'tcx>(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>) -> Option<String> {
    #[derive(Serialize)]
    struct EffectsCapture {
        pure: bool,
        has_side_effects: bool,
        performs_io: bool,
        panics: bool,
        unsafe_operations: Vec<String>,
    }
    let mut has_side_effects = false;
    let mut performs_io = false;
    let mut panics = false;
    let mut unsafe_ops = Vec::new();
    for bb in body.basic_blocks.iter() {
        match &bb.terminator().kind {
            TerminatorKind::Call { func, .. } => {
                has_side_effects = true;
                if let ty::FnDef(def_id, _) = *func.ty(&body.local_decls, tcx).kind() {
                    let path = normalize_symbol_id_with_crate(
                        &tcx.def_path_str(def_id),
                        Some(&tcx.crate_name(def_id.krate).to_string()),
                    );
                    if path.contains("::io::") {
                        performs_io = true;
                    }
                    if path.contains("panic") {
                        panics = true;
                    }
                }
            }
            TerminatorKind::Assert { .. } => {
                panics = true;
            }
            _ => {}
        }
        for stmt in &bb.statements {
            if matches!(stmt.kind, mir::StatementKind::Assign(_)) {
                has_side_effects = true;
            }
            if matches!(stmt.kind, mir::StatementKind::FakeRead(..)) {
                unsafe_ops.push("FakeRead".into());
            }
        }
    }
    serde_json::to_string(
            &EffectsCapture {
                pure: !(has_side_effects || performs_io || panics),
                has_side_effects,
                performs_io,
                panics,
                unsafe_operations: unsafe_ops,
            },
        )
        .ok()
}
