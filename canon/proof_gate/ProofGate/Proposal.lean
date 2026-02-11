import LearnGate.State

namespace LearnGate

/-- Minimal description of a delta proposed by the runtime. -/
structure ProposalDelta where
  deltaId : Nat
  payloadHash : String
  bytes : Nat
deriving Repr, DecidableEq

/-- Shell metadata bound into a proposal when the runtime executes a shell graph. -/
structure ShellMetadata where
  shellId : Nat
  epoch : Nat
  command : String
  stateHash : String
deriving Repr, DecidableEq

/--
A proposal is what the runtime suggests for the current graph.
We keep only the graph identifier and a list of deltas it intends to apply.
-/
structure Proposal where
  graphId : String
  deltas  : List ProposalDelta
  shell?  : Option ShellMetadata := none
deriving Repr, DecidableEq

/-- Semantics: proposals currently leave state unchanged (pure gate). -/
def applyRule (_ : Proposal) (s : State) : State :=
  s

/-- Enable condition (“guard”): in this phase every proposal is eligible. -/
def enabled (_ : Proposal) (_ : State) : Prop :=
  True

end LearnGate
