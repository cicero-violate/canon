namespace LearnGate

/-- World state at tick t. Keep it tiny and formal. -/
structure State where
  rootHash : String
deriving Repr, DecidableEq

end LearnGate
