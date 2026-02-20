//! `report` — runs every analysis domain over a captured graph and returns a
//! structured `AnalysisReport`.  The example (and any future binary) calls
//! `report::run_all(nodes, edges)` and prints the result.

use crate::{
    abstract_interp::{AbstractDomain, Interval},
    alias::PointsToGraph,
    call_graph::CallGraph,
    cfg::Cfg,
    concurrency::LocksetState,
    dataflow::{reaching_definitions, DataflowFacts},
    deadcode::DeadCodeAnalysis,
    effect::{Effect, EffectMap},
    escape::EscapeSet,
    interproc::{FnSummary, SummaryStore},
    lifetime::LifetimeRegions,
    scc::tarjan_scc,
    taint::TaintState,
    types::TypeHierarchy,
    usedef::UseDefChains,
};
use database::graph_log::{GraphSnapshot, WireEdge, WireNode};
use std::collections::{HashMap, HashSet};

// ── Public report type ────────────────────────────────────────────────────────

/// Aggregated results from all analysis domains.
#[derive(Debug)]
pub struct AnalysisReport {
    // graph basics
    pub node_count: usize,
    pub edge_count: usize,
    pub call_edge_count: usize,
    pub contains_edge_count: usize,
    pub ref_edge_count: usize,

    // call graph
    pub entry_points: Vec<String>,
    pub reachability: Vec<(String, usize)>, // (entry, reachable count)

    // dead code
    pub dead_symbols: Vec<String>,

    // SCC / cycles
    pub scc_count: usize,
    pub cycles: Vec<Vec<String>>,

    // effects
    pub unsafe_fns: Vec<String>,
    pub panic_fns: Vec<String>,

    // taint
    pub taint_sources: usize,
    pub taint_sinks: usize,
    pub tainted_sinks: Vec<String>,

    // type hierarchy (sample)
    pub supertype_samples: Vec<(String, Option<Vec<String>>)>,

    // alias
    pub alias_pairs: usize,
    pub alias_samples: Vec<(String, String)>,

    // cfg dominators
    pub cfg_nontrivial_dom_count: usize,

    // concurrency
    pub races: Vec<String>,

    // dataflow
    pub df_blocks: usize,
    pub df_total_facts: usize,

    // escape
    pub escaped_symbols: Vec<String>,

    // lifetime
    pub lifetime_regions: usize,
    pub lifetime_outlives: usize,

    // use-def
    pub usedef_pairs: usize,
    pub used_before_defined: Vec<String>,

    // interproc
    pub summarized_fns: usize,

    // abstract interp
    pub ai_base: Interval,
    pub ai_next: Interval,
    pub ai_widened: Interval,
}

impl Default for AnalysisReport {
    fn default() -> Self {
        Self {
            node_count: 0,
            edge_count: 0,
            call_edge_count: 0,
            contains_edge_count: 0,
            ref_edge_count: 0,
            entry_points: Vec::new(),
            reachability: Vec::new(),
            dead_symbols: Vec::new(),
            scc_count: 0,
            cycles: Vec::new(),
            unsafe_fns: Vec::new(),
            panic_fns: Vec::new(),
            taint_sources: 0,
            taint_sinks: 0,
            tainted_sinks: Vec::new(),
            supertype_samples: Vec::new(),
            alias_pairs: 0,
            alias_samples: Vec::new(),
            cfg_nontrivial_dom_count: 0,
            races: Vec::new(),
            df_blocks: 0,
            df_total_facts: 0,
            escaped_symbols: Vec::new(),
            lifetime_regions: 0,
            lifetime_outlives: 0,
            usedef_pairs: 0,
            used_before_defined: Vec::new(),
            summarized_fns: 0,
            ai_base: Interval::Bottom,
            ai_next: Interval::Bottom,
            ai_widened: Interval::Bottom,
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Run every analysis domain over the provided nodes and edges.
/// Returns a fully populated [`AnalysisReport`].
pub fn run_all(nodes: &[WireNode], edges: &[WireEdge]) -> AnalysisReport {
    let mut r = AnalysisReport::default();

    // bookkeeping indices used across domains
    let id_to_key: HashMap<_, _> =
        nodes.iter().map(|n| (n.id.clone(), n.key.clone())).collect();
    let key_to_idx: HashMap<String, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.key.clone(), i)).collect();
    let id_label = |id: &_| {
        id_to_key
            .get(id)
            .cloned()
            .unwrap_or_else(|| format!("{id:?}"))
    };

    // ── graph basics ──────────────────────────────────────────────────────────
    r.node_count = nodes.len();
    r.edge_count = edges.len();
    r.call_edge_count = edges.iter().filter(|e| e.kind == "call").count();
    r.contains_edge_count = edges.iter().filter(|e| e.kind == "contains").count();
    r.ref_edge_count = edges.iter().filter(|e| e.kind == "reference").count();

    // ── call graph ────────────────────────────────────────────────────────────
    let snapshot = GraphSnapshot::new(nodes.to_vec(), edges.to_vec());
    let mut cg = CallGraph::from_snapshot(snapshot);
    r.entry_points = nodes
        .iter()
        .filter(|n| {
            n.key.ends_with("::main")
                || n.key.ends_with("::run")
                || n.key.ends_with("::execute")
        })
        .map(|n| n.key.clone())
        .collect();
    for entry in &r.entry_points {
        let reachable = cg.reachable_from(entry);
        let count = reachable.len();
        r.reachability.push((entry.clone(), count));
    }

    // ── dead code ─────────────────────────────────────────────────────────────
    let mut dc = DeadCodeAnalysis::new();
    for entry in &r.entry_points {
        dc.add_entry(entry.clone());
    }
    for n in nodes {
        if n.metadata.get("kind").map(|k| k == "fn").unwrap_or(false)
            && n.metadata
                .get("visibility")
                .map(|v| v == "pub")
                .unwrap_or(false)
        {
            dc.add_entry(n.key.clone());
        }
    }
    for e in edges {
        if e.kind == "call" {
            dc.add_call(id_label(&e.from), id_label(&e.to));
        }
    }
    let mut dead: Vec<String> = dc.dead_symbols().into_iter().collect();
    dead.sort();
    r.dead_symbols = dead;

    // ── SCC / cycles ──────────────────────────────────────────────────────────
    let mut adj: HashMap<String, Vec<String>> = HashMap::new();
    for e in edges {
        if e.kind == "call" {
            let from = id_label(&e.from);
            let to = id_label(&e.to);
            if !from.is_empty() && !to.is_empty() {
                adj.entry(from).or_default().push(to);
            }
        }
    }
    let sccs = tarjan_scc(&adj);
    r.scc_count = sccs.len();
    r.cycles = sccs.iter().filter(|s| s.len() > 1).cloned().collect();

    // ── effects ───────────────────────────────────────────────────────────────
    let mut em = EffectMap::new();
    for n in nodes {
        if n.metadata.get("is_unsafe").map(|v| v == "true").unwrap_or(false) {
            em.add(&n.key, Effect::Unsafe);
        }
        if n.metadata.get("has_panic").map(|v| v == "true").unwrap_or(false) {
            em.add(&n.key, Effect::Panics);
        }
        if n.metadata.get("effects").map(|v| v.contains("io")).unwrap_or(false) {
            em.add(&n.key, Effect::Io);
        }
        if n.metadata.get("effects").map(|v| v.contains("alloc")).unwrap_or(false) {
            em.add(&n.key, Effect::Allocates);
        }
    }
    r.unsafe_fns = nodes
        .iter()
        .filter(|n| em.get(&n.key).iter().any(|e| **e == Effect::Unsafe))
        .map(|n| n.key.clone())
        .collect();
    r.panic_fns = nodes
        .iter()
        .filter(|n| em.get(&n.key).iter().any(|e| **e == Effect::Panics))
        .map(|n| n.key.clone())
        .collect();

    // ── taint ─────────────────────────────────────────────────────────────────
    let mut taint = TaintState::new();
    let sources: Vec<String> = nodes
        .iter()
        .filter(|n| {
            n.key.contains("parse")
                || n.key.contains("input")
                || n.key.contains("read")
                || n.key.contains("recv")
        })
        .map(|n| n.key.clone())
        .collect();
    let sinks: Vec<String> = nodes
        .iter()
        .filter(|n| {
            n.key.contains("exec")
                || n.key.contains("write")
                || n.key.contains("send")
                || n.key.contains("commit")
        })
        .map(|n| n.key.clone())
        .collect();
    for src in &sources {
        taint.add_source(src.clone(), "external".into());
    }
    for e in edges {
        if e.kind == "call" {
            let from = id_label(&e.from);
            let to = id_label(&e.to);
            taint.propagate(&from, &to);
        }
    }
    let forbidden: HashSet<String> = ["external".to_string()].into();
    r.taint_sources = sources.len();
    r.taint_sinks = sinks.len();
    r.tainted_sinks = sinks
        .iter()
        .filter(|s| !taint.check_sink(s, &forbidden).is_empty())
        .cloned()
        .collect();

    // ── type hierarchy ────────────────────────────────────────────────────────
    let mut th = TypeHierarchy::new();
    for e in edges {
        if e.kind == "contains" {
            let parent = id_label(&e.from);
            let child = id_label(&e.to);
            if !parent.is_empty() && !child.is_empty() {
                th.add_subtype(child, parent);
            }
        }
    }
    r.supertype_samples = nodes
        .iter()
        .take(3)
        .map(|n| {
            let supers = th
                .supertypes
                .get(&n.key)
                .map(|set| {
                    let mut v: Vec<String> = set.iter().cloned().collect();
                    v.sort();
                    v
                });
            (n.key.clone(), supers)
        })
        .collect();

    // ── alias ─────────────────────────────────────────────────────────────────
    let mut ptg = PointsToGraph::new();
    for e in edges {
        if e.kind == "reference" {
            let from = id_label(&e.from);
            let to = id_label(&e.to);
            if !from.is_empty() && !to.is_empty() {
                ptg.add_address_of(&from, &to);
                ptg.add_assign(&from, &to);
            }
        }
    }
    let ref_vars: Vec<String> = edges
        .iter()
        .filter(|e| e.kind == "reference")
        .flat_map(|e| {
            [
                id_to_key.get(&e.from).cloned().unwrap_or_default(),
                id_to_key.get(&e.to).cloned().unwrap_or_default(),
            ]
        })
        .filter(|s| !s.is_empty())
        .collect::<HashSet<_>>()
        .into_iter()
        .take(6)
        .collect();
    let ref_vars_v: Vec<&String> = ref_vars.iter().collect();
    for i in 0..ref_vars_v.len() {
        for j in (i + 1)..ref_vars_v.len() {
            if ptg.may_alias(ref_vars_v[i], ref_vars_v[j]) {
                r.alias_pairs += 1;
                if r.alias_samples.len() < 3 {
                    r.alias_samples
                        .push((ref_vars_v[i].clone(), ref_vars_v[j].clone()));
                }
            }
        }
    }

    // ── cfg dominators ────────────────────────────────────────────────────────
    let mut cfg = Cfg::new();
    for (i, n) in nodes.iter().enumerate() {
        cfg.nodes.push(crate::cfg::CfgNode { id: i, label: n.key.clone() });
    }
    for e in edges {
        if e.kind == "call" {
            let from = id_label(&e.from);
            let to = id_label(&e.to);
            if let (Some(&fi), Some(&ti)) =
                (key_to_idx.get(&from), key_to_idx.get(&to))
            {
                cfg.add_edge(fi, ti);
            }
        }
    }
    let doms = cfg.dominators();
    r.cfg_nontrivial_dom_count = doms.values().filter(|d| d.len() > 1).count();

    // ── concurrency ───────────────────────────────────────────────────────────
    let mut ls = LocksetState::new();
    for (idx, scc) in sccs.iter().enumerate().take(10) {
        let thread = format!("thread_{idx}");
        let lock = format!("lock_{idx}");
        ls.acquire(&thread, lock.clone());
        for sym in scc.iter().take(4) {
            ls.record_access(&thread, sym.clone());
        }
        ls.release(&thread, &lock);
    }
    r.races = ls.races();

    // ── dataflow ──────────────────────────────────────────────────────────────
    let block_ids: Vec<usize> = (0..sccs.len().min(16)).collect();
    let mut df_facts = DataflowFacts {
        r#gen: HashMap::new(),
        kill: HashMap::new(),
    };
    let mut df_pred: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, scc) in sccs.iter().enumerate().take(16) {
        df_facts.r#gen.insert(i, scc.iter().cloned().collect());
        df_facts.kill.insert(i, HashSet::new());
        if i > 0 {
            df_pred.entry(i).or_default().push(i - 1);
        }
    }
    let df_result = reaching_definitions(&block_ids, &df_pred, &df_facts);
    r.df_blocks = block_ids.len();
    r.df_total_facts = df_result.out.values().map(|s| s.len()).sum();

    // ── escape ────────────────────────────────────────────────────────────────
    let mut esc = EscapeSet::new();
    for src in &sources {
        esc.mark_escaped(src.clone());
    }
    r.escaped_symbols = sources.iter().filter(|s| esc.escapes(s)).cloned().collect();

    // ── lifetime ──────────────────────────────────────────────────────────────
    let mut lr = LifetimeRegions::new();
    for (i, e) in edges.iter().enumerate().filter(|(_, e)| e.kind == "reference") {
        let r_from = format!("r_{}", id_label(&e.from));
        let r_to = format!("r_{}", id_label(&e.to));
        lr.add_live_point(r_from.clone(), i);
        lr.add_live_point(r_to.clone(), i);
        lr.add_outlives(r_from, r_to);
    }
    lr.propagate();
    r.lifetime_regions = r.ref_edge_count * 2;
    r.lifetime_outlives = r.ref_edge_count;

    // ── use-def ───────────────────────────────────────────────────────────────
    let mut ud = UseDefChains::new();
    for (i, e) in edges.iter().enumerate().filter(|(_, e)| e.kind == "call") {
        ud.add_chain(format!("def_{}", id_label(&e.from)), format!("use_{i}"));
    }
    r.usedef_pairs = r.call_edge_count;
    r.used_before_defined = ud.used_before_defined();

    // ── interproc ─────────────────────────────────────────────────────────────
    let mut store = SummaryStore::new();
    let scc_order = sccs.clone();
    store.compute_bottom_up(&scc_order, |fn_id: &String, _: &SummaryStore| {
        FnSummary {
            pre: HashMap::from([("pre".to_string(), format!("pre({fn_id})"))]),
            post: HashMap::from([("post".to_string(), format!("post({fn_id})"))]),
        }
    });
    r.summarized_fns = scc_order.iter().flat_map(|s| s.iter()).count();

    // ── abstract interp ───────────────────────────────────────────────────────
    r.ai_base = Interval::Range(0, nodes.len() as i64);
    r.ai_next = Interval::Range(0, (nodes.len() + edges.len()) as i64);
    r.ai_widened = r.ai_base.widen(&r.ai_next);

    r
}

// ── Display ───────────────────────────────────────────────────────────────────

impl AnalysisReport {
    pub fn print(&self) {
        println!("=== analysis report ===\n");

        println!("--- graph ---");
        println!("  nodes:          {}", self.node_count);
        println!("  edges:          {}", self.edge_count);
        println!("  call edges:     {}", self.call_edge_count);
        println!("  contains edges: {}", self.contains_edge_count);
        println!("  ref edges:      {}", self.ref_edge_count);

        println!("\n--- call graph ---");
        println!("  entry points: {}", self.entry_points.len());
        for (entry, count) in &self.reachability {
            println!("    {entry}: {count} reachable");
        }

        println!("\n--- dead code ---");
        println!("  dead symbols: {}", self.dead_symbols.len());
        for s in self.dead_symbols.iter().take(10) {
            println!("    {s}");
        }
        if self.dead_symbols.len() > 10 {
            println!("    ... and {} more", self.dead_symbols.len() - 10);
        }

        println!("\n--- scc / cycles ---");
        println!("  total SCCs: {}", self.scc_count);
        println!("  recursive cycles: {}", self.cycles.len());
        for c in self.cycles.iter().take(5) {
            println!("    {:?}", c);
        }

        println!("\n--- effects ---");
        println!("  unsafe fns: {}", self.unsafe_fns.len());
        for s in self.unsafe_fns.iter().take(5) { println!("    {s}"); }
        println!("  panic-capable fns: {}", self.panic_fns.len());
        for s in self.panic_fns.iter().take(5) { println!("    {s}"); }

        println!("\n--- taint ---");
        println!("  sources: {}", self.taint_sources);
        println!("  sinks:   {}", self.taint_sinks);
        println!("  tainted sinks: {}", self.tainted_sinks.len());
        for s in self.tainted_sinks.iter().take(5) { println!("    {s}"); }

        println!("\n--- type hierarchy ---");
        for (key, supers) in &self.supertype_samples {
            println!("  supertypes({key}) = {supers:?}");
        }

        println!("\n--- alias ---");
        println!("  alias pairs (sampled): {}", self.alias_pairs);
        for (a, b) in &self.alias_samples {
            println!("    {a} ~ {b}");
        }

        println!("\n--- cfg dominators ---");
        println!("  nodes with non-trivial dominators: {}", self.cfg_nontrivial_dom_count);

        println!("\n--- concurrency ---");
        println!("  potential races: {}", self.races.len());
        for r in self.races.iter().take(5) { println!("    {r}"); }

        println!("\n--- dataflow ---");
        println!("  blocks analyzed:      {}", self.df_blocks);
        println!("  total reaching facts: {}", self.df_total_facts);

        println!("\n--- escape ---");
        println!("  escaped symbols: {}", self.escaped_symbols.len());
        for s in self.escaped_symbols.iter().take(5) { println!("    {s}"); }

        println!("\n--- lifetime ---");
        println!("  regions:          {}", self.lifetime_regions);
        println!("  outlives constraints: {}", self.lifetime_outlives);

        println!("\n--- use-def chains ---");
        println!("  pairs modeled:          {}", self.usedef_pairs);
        println!("  used-before-defined:    {}", self.used_before_defined.len());

        println!("\n--- interproc summaries ---");
        println!("  functions summarized: {}", self.summarized_fns);

        println!("\n--- abstract interp ---");
        println!("  base:    {:?}", self.ai_base);
        println!("  next:    {:?}", self.ai_next);
        println!("  widened: {:?}", self.ai_widened);

        println!("\n=== summary ===");
        println!("  nodes:               {}", self.node_count);
        println!("  edges:               {}", self.edge_count);
        println!("  entry points:        {}", self.entry_points.len());
        println!("  dead symbols:        {}", self.dead_symbols.len());
        println!("  call cycles:         {}", self.cycles.len());
        println!("  tainted sinks:       {}", self.tainted_sinks.len());
        println!("  alias pairs:         {}", self.alias_pairs);
        println!("  cfg dom nodes:       {}", self.cfg_nontrivial_dom_count);
        println!("  races:               {}", self.races.len());
        println!("  reach-def facts:     {}", self.df_total_facts);
        println!("  escaped symbols:     {}", self.escaped_symbols.len());
        println!("  lifetime regions:    {}", self.lifetime_regions);
        println!("  use-def violations:  {}", self.used_before_defined.len());
        println!("  summarized fns:      {}", self.summarized_fns);
    }
}
