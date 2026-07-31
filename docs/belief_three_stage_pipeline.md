# Three-stage MAS belief pipeline

The three-stage pipeline avoids giving early-discovered beliefs more evidence
opportunities than beliefs discovered near the end of the topic corpus.

## Stages

1. **Candidate discovery**
   - Processes each topic sequentially.
   - Uses `ADD`, `MATCH`, and `PROPOSE_REVISION`.
   - Revision proposals are stored without changing canonical candidate text.
   - No Bayesian update is performed.
2. **Global consolidation**
   - Processes the complete candidate bank in one pass.
   - Resolves proposed revisions and merges genuinely equivalent candidates.
   - Assigns stable belief IDs and freezes every final definition.
   - Has no application-level output-token cap because every candidate must be
     accounted for in the structured response.
   - Enforces atomic beliefs: one causal mechanism and one coherent strategy
     per posterior, with at most four source candidates per final belief.
   - Preserves rare or tool-specific reusable candidates as probationary
     beliefs; low frequency is not a valid exclusion reason.
   - Gives Gemini four explicit merge questions as prompt guidance: same
     problem, same causal mechanism, same strategy, and one interpretable
     effectiveness variable.
3. **Retrospective evidence evaluation**
   - Evaluates the same frozen bank against every topic.
   - Every belief–topic pair is recorded as `applicable`, `not_applicable`, or
     `insufficient`.
   - Applicable evidence uses categorical classifications which local code
     maps to deterministic Bayesian inputs.

## Pawsey execution

From the repository root:

```bash
sbatch slurm/run_belief_generation_random30.slurm discovery
```

To reuse a completed candidate checkpoint and begin with the uncapped global
consolidation stage:

```bash
sbatch slurm/run_belief_generation_random30.slurm consolidation
```

This requires `belief_candidates.json` to exist and have `"complete": true`.
The resume mode does not clear the output directory.

To preserve the existing frozen belief definitions and restart only the
retrospective evidence stage with the current Stage 3 prompt:

```bash
sbatch slurm/run_belief_generation_random30.slurm evidence
```

This loads `frozen_beliefs.json`, resets every belief to the common prior, and
re-evaluates all topics from topic 1 without rerunning discovery or consolidation.

The job starts with a fresh three-stage output directory by default. It does
not overwrite the legacy sequential belief experiment at the pipeline root.

## Outputs

Outputs are written under:

```text
mas_logs/pipeline_20260728_101242/belief_three_stage/
```

- `belief_candidates.json`: candidate origins, matches, and deferred revisions
- `belief_consolidation.json`: consolidation decisions and exclusions
- `frozen_beliefs.json`: definitions used throughout retrospective evaluation
- `belief_evidence_matrix.jsonl`: one row for every belief–topic pair
- `belief_library.json`: final Bayesian belief bank
- `belief_evidence.jsonl`: denormalised applicable evidence
- `belief_embeddings.npz`: persisted BGE vectors
- `belief_embedding_metadata.json`: embedding IDs and text hashes
- `belief_pipeline_analysis.json`: usage and call-level summary
- `belief_pipeline_progress.json`: current stage and progress
- `bbn_parameters.json`: applicability CPD configuration

Monitor progress with:

```bash
tail -f code2video-beliefs-random30-<job-id>.out
```

or:

```bash
cat mas_logs/pipeline_20260728_101242/belief_three_stage/belief_pipeline_progress.json
```
