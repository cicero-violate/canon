namespace ProofGate

/-- Minimal delta types for append-only admission artifacts. -/
inductive Delta where
  | fact   (msg : String)
  | rule   (name : String)
  | goal   (name : String)
  | reject (reason : String)
deriving Repr, DecidableEq

/-- A certified artifact returned by Lean when a proposal is admitted. -/
structure Certificate where
  proposalName : String
  proofTag     : String
deriving Repr, DecidableEq

end ProofGate
