---

### Latent Constraint

[
\text{Minimum System} = {S_{min}, C_{min}, P, E}
]

---

# ✅ What To Have (Minimum Viable Substrate)

## 1) Five State Facts

[
S_{min} =
{
exists(path),;
file(path),;
dir(path),;
running(proc),;
installed(bin)
}
]

That’s it.

Nothing else.

---

## 2) Five Probes

```
test -e path
test -f path
test -d path
ps -C name
which name
```

Each maps 1→1 to a fact.

---

## 3) Capability Record Shape

[
c = \langle name,; pre,; exec,; post \rangle
]

Example:

```
name: build_rust
pre: exists(Cargo.toml)
exec: cargo build
post: exists(target/)
```

---

## 4) Initial Capability Set

```
mkdir(path)
rm(path)
git_clone(url, path)
cargo_build()
cargo_test()
```

5–10 total.

---

## 5) Planner Behavior

Input:

```
Goal: exists(target/)
```

Planner:

```
if not exists(Cargo.toml) -> git_clone
if not installed(cargo) -> install_rust
then cargo_build
```

No search explosion.
No ML.
Simple chaining.

---

## 6) Executor Loop

```
for action in plan:
    run exec
    re-probe facts
    update graph
    if postcondition false:
        replan
```

---

# ❌ What You Do NOT Have Yet

* Full filesystem mirror
* Package databases
* Hardware models
* Network models
* Ontologies

All of that comes later.

---

# 🧠 Mental Model

You are building:

> A planner that can compile software by itself.

Not:

> A digital twin of Linux.

---

# Success Criterion

If your system can:

* Detect missing file
* Create it
* Build project
* Detect failure
* Retry

You already crossed the threshold.

---

# Expansion Rule

Only add a new fact type when:

[
\text{System fails without it}
]

---

# Final Answer

[
\boxed{
Have 5 facts, 5 probes, 5–10 capabilities, a planner, and a closed-loop executor.
}
]

Everything else is noise right now.

---

### Optimization Objective

[
\max(I, E, C, A, R, P, S, D, T, K, X, B, L, F) = \text{Good}
]

Build the smallest loop that works.
Then grow.
