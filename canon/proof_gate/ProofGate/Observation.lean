import ProofGate.State
import ProofGate.Core

namespace ProofGate

/-- Observation is what the runtime records (already happened). -/
structure Observation where
  spent : Nat
deriving Repr, DecidableEq

/-- Extract a fact delta from an observation (pure, no proof). -/
def extractFact (o : Observation) : Delta :=
  Delta.fact s!"spent={o.spent}"

end ProofGate
