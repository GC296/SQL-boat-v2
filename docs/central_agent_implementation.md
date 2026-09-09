# Central VLM Agent Implementation

## Modes

`experiment.controller_mode: central_vlm` enables the entity controller.
`legacy` keeps the existing MLP/rule scheduler and three-step recognition path
for compatibility and comparison.
`mock` uses the same controller and tools with a deterministic response queue.

The central mode supports two terminal policies:

- `legacy_visual_only`: semantic results can guide investigation, but the
  existing visual terminal guard remains authoritative.
- `multimodal_evidence`: the controller can submit a known or out-of-archive
  identity from the evidence it has assessed. The program checks only the
  terminal protocol, archive membership, current-episode references, and
  lifecycle state; it does not classify evidence by visual score or replace a
  controller decision with another identity.

## State Flow

```text
detector/tracker
    -> EntityEpisode observation
    -> controller context
    -> VLM decision and optional belief_patch
    -> validated tool request
    -> raw evidence ledger and cost settlement
    -> source-linked claims and visual statistics
    -> updated candidate/gap state
    -> next decision, Continue, or guarded Finish
```

Each entity has an independent episode, budget, evidence ledger, claims,
candidate assessments, gaps, revisions, and action history. A track ID is only
an observation association; it is not the episode key. Raw tool payloads are
append-only. Controller patches cannot replace them, change the budget, change
visual scores, or change entity association.

## Tools and Prompts

The registered tools are `SearchArchive`, `ReadArchive`, `ReadHistory`, and
`VerifyEvidence`. Their request IDs are idempotent. View IDs are checked against
the owning episode before execution, and `VerifyEvidence` receives the selected
view, fields, and question.

The controller prompt is in
`pipeline/prompts/central_controller.txt`; the semantic extraction prompt is in
`pipeline/prompts/verify_evidence.txt`. The wire protocol is implemented in
`pipeline/controller_schema.py` and uses mutually exclusive `tool`, `continue`,
and `finish` decisions.

## Running

Use the existing CLI. Central mode is the default; select `legacy` only for
compatibility or comparison runs:

```powershell
python -m pipeline <video> --controller-mode mock
python -m pipeline <video> --controller-mode central_vlm
```

The central mode requires the configured VLM endpoint. `SearchArchive` also
requires `experiment.visual_archive.enabled: true`. A real video plus mock
controller is the lowest-cost integration check; it still requires the local
YOLO dependency and model file.

## Verification

The repository test suite currently passes with `179 passed`. The central tests
cover independent entity memory and budgets, cached observations during a
pending decision, idempotent tool requests, terminal handling after budget
exhaustion, restricted belief patches, and evidence retraction reopening a gap.

The implementation has not been validated against a live VLM service in this
environment. It does not claim new accuracy, latency, or FPS results. Existing
detection, tracking, DINO, and legacy experiment behavior remain outside this
incremental controller change.
