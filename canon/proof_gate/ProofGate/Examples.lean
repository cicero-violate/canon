import LearnGate.State
import LearnGate.Observation
import LearnGate.Proposal
import LearnGate.Policy

namespace LearnGate

def s0 : State := { rootHash := "demo-root" }
def o0 : Observation := { spent := 3 }
def p0 : Proposal :=
  { graphId := "system.demo"
  , deltas :=
      [ { deltaId := 1, payloadHash := "abc123", bytes := 3 } ]
  }

def demo : Policy × List Delta :=
  let pi0 := empty
  let fact := extractFact o0
  let (pi1, deltas) := applyLearning pi0 p0
  (pi1, fact :: deltas)

end LearnGate
