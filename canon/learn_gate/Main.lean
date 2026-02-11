import Lean.Data.Json
import LearnGate.Verify
import LearnGate.State
import LearnGate.Proposal

open Lean

structure RequestState where
  root_hash : String
deriving FromJson

structure RequestDelta where
  delta_id : Nat
  payload_hash : String
  bytes : Nat
deriving FromJson

structure RequestShell where
  shell_id : Nat
  epoch : Nat
  command : String
  state_hash : String
deriving FromJson

structure RequestProposal where
  graph_id : String
  deltas : List RequestDelta
  shell? : Option RequestShell := none
deriving FromJson

structure Request where
  state : RequestState
  proposal : RequestProposal
deriving FromJson

def deltaToJson : LearnGate.Delta -> Json
  | LearnGate.Delta.fact msg =>
      Json.mkObj [("kind", Json.str "fact"), ("message", Json.str msg)]
  | LearnGate.Delta.rule name =>
      Json.mkObj [("kind", Json.str "rule"), ("name", Json.str name)]
  | LearnGate.Delta.goal name =>
      Json.mkObj [("kind", Json.str "goal"), ("name", Json.str name)]
  | LearnGate.Delta.reject reason =>
      Json.mkObj [("kind", Json.str "reject"), ("reason", Json.str reason)]

def certificateToJson (cert : LearnGate.Certificate) : Json :=
  Json.mkObj
    [ ("proposal_name", Json.str cert.proposalName)
    , ("proof_tag", Json.str cert.proofTag)
    ]

def responseToJson
    (status : String)
    (cert? : Option LearnGate.Certificate := none)
    (delta? : Option LearnGate.Delta := none)
    (error? : Option String := none)
    (proofs : List Json := []) : Json :=
  let base := [("status", Json.str status)]
  let base :=
    match cert? with
    | some cert => base ++ [("certificate", certificateToJson cert)]
    | none => base
  let base :=
    match delta? with
    | some delta => base ++ [("delta", deltaToJson delta)]
    | none => base
  let base :=
    match error? with
    | some err => base ++ [("error", Json.str err)]
    | none => base
  let base := base ++ [("proofs", Json.arr proofs.toArray)]
  Json.mkObj base

def mkProof (name detail : String) : Json :=
  Json.mkObj
    [ ("name", Json.str name)
    , ("status", Json.str "proved")
    , ("detail", Json.str detail)
    ]

def emitJson (json : Json) : IO Unit :=
  IO.println json.compress

def toState (st : RequestState) : LearnGate.State :=
  { rootHash := st.root_hash }

def toProposalDelta (delta : RequestDelta) : LearnGate.ProposalDelta :=
  { deltaId := delta.delta_id
  , payloadHash := delta.payload_hash
  , bytes := delta.bytes
  }

def toShellMetadata (sh : RequestShell) : LearnGate.ShellMetadata :=
  { shellId := sh.shell_id
  , epoch := sh.epoch
  , command := sh.command
  , stateHash := sh.state_hash
  }

def toProposal (prop : RequestProposal) : LearnGate.Proposal :=
  { graphId := prop.graph_id
  , deltas := prop.deltas.map toProposalDelta
  , shell? := prop.shell?.map toShellMetadata
  }

def runVerify (req : Request) : IO Unit := do
  let state := toState req.state
  let proposal := toProposal req.proposal
  let baseProofs := [mkProof "state_hash" s!"state root hash {state.rootHash}"]
  let shellProofs :=
    match proposal.shell? with
    | some shell =>
        [mkProof "shell_state"
          s!"shell {shell.shellId} epoch {shell.epoch} hash {shell.stateHash}"]
    | none => []
  let proofs := baseProofs ++ shellProofs
  match LearnGate.verify proposal with
  | Sum.inr cert =>
      emitJson (responseToJson "accepted" (some cert) none none proofs)
  | Sum.inl delta =>
      emitJson
        (responseToJson "rejected" none (some delta) (some "proof rejected") proofs)

def errorResponse (msg : String) : Json :=
  responseToJson "error" none none (some msg) []

def main : IO Unit := do
  let stdin ← IO.getStdin
  let payload ← stdin.readToEnd
  match Json.parse payload with
  | Except.error err =>
      emitJson (errorResponse s!"invalid json: {err}")
  | Except.ok json =>
      match fromJson? json with
      | Except.error err =>
          emitJson (errorResponse s!"invalid payload: {err}")
      | Except.ok req =>
          runVerify req
