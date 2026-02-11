import LearnGate.Core
import LearnGate.Proposal
import LearnGate.Verify

namespace LearnGate

/-- Policy is the set of admitted rule names (toy model). -/
structure Policy where
  rules : List String
deriving Repr, DecidableEq

def empty : Policy := { rules := [] }

def addRule (pi : Policy) (ruleName : String) : Policy :=
  { rules := ruleName :: pi.rules }

/--
Update policy only if Lean produced a certificate.
This is the “only proven rules enter policy” rule.
-/
def applyLearning (pi : Policy) (p : Proposal) : Policy × List Delta :=
  match verify p with
  | Sum.inr cert =>
      (addRule pi cert.proposalName, [Delta.rule cert.proposalName])
  | Sum.inl rej =>
      (pi, [rej])

end LearnGate
