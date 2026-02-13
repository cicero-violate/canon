# Tier 1–7 Requirements Table

| ---- | ------------------------------ | -------------------- | --------------------------------------------------------------- | ------- | ------- | ------------- |
| Tier | Name                           | Core Capability      | Formal Condition                                                | Learns? | Plans?  | Self-Updates? |
| ---- | ------------------------------ | -------------------- | --------------------------------------------------------------- | ------- | ------- | ------------- |
|    1 | Static System                  | Fixed mapping        | ( A = f(O) )                                                    | No      | No      | No            |
|    2 | Stateful System                | Persistent state     | ( S_{t+1} = f(S_t, O_t) )                                       | No      | No      | No            |
|    3 | Tool Agent                     | Goal-directed action | ( A = \Pi(G, S) )                                               | No      | Limited | No            |
|    4 | Planner                        | Forward simulation   | ( A = \mathcal{P}(\hat{S}_{future}) )                           | No      | Yes     | No            |
|    5 | Learning Agent                 | Parameter update     | ( \Theta_{t+1} = \mathcal{L}(\Theta_t, R_t) )                   | Yes     | Yes     | No            |
|    6 | Self-Improving Agent           | Policy adaptation    | ( \Pi_{t+1} \neq \Pi_t )                                        | Yes     | Yes     | Partial       |
|    7 | Recursive Autonomous Optimizer | Closed adaptive loop | ( \Pi_{t+1} = \arg\max \mathbb{E}[R] ) with model + goal update | Yes     | Yes     | Yes           |

# Expanded Requirements by Tier

| Tier | Persistent State | Utility Scalar | World Model  | Counterfactual Simulation | Planner                | Learning | Policy Update | Goal Adaptation | Self-Modification |
| ---- | ---------------- | -------------- | ------------ | ------------------------- | ---------------------- | -------- | ------------- | --------------- | ----------------- |
| 1    | ✗                | ✗              | ✗            | ✗                         | ✗                      | ✗        | ✗             | ✗               | ✗                 |
| 2    | ✓                | ✗              | ✗            | ✗                         | ✗                      | ✗        | ✗             | ✗               | ✗                 |
| 3    | ✓                | ✓ (external)   | ✗            | ✗                         | Limited                | ✗        | ✗             | ✗               | ✗                 |
| 4    | ✓                | ✓              | Basic        | ✓                         | ✓                      | ✗        | ✗             | ✗               | ✗                 |
| 5    | ✓                | ✓              | ✓            | ✓                         | ✓                      | ✓        | ✗             | ✗               | ✗                 |
| 6    | ✓                | ✓              | ✓            | ✓                         | ✓                      | ✓        | ✓             | ✗               | Limited           |
| 7    | ✓                | ✓              | ✓ (adaptive) | ✓ (multi-step)            | ✓ (optimization-based) | ✓        | ✓             | ✓               | ✓                 |

---





