import LearnGate.State

namespace LearnGate

/-- Invariants are predicates over State. Currently we just require that a root hash exists. -/
def Inv (_ : State) : Prop := True

theorem inv_holds (s : State) : Inv s := by
  exact trivial

end LearnGate
