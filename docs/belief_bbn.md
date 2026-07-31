# Bayesian belief lifecycle

The belief pipeline keeps three separate artifacts:

- `belief_evidence.jsonl`: episode-specific observations extracted from MAS logs.
- `belief_library.json`: reusable beliefs and their Beta effectiveness posteriors.
- `bbn_parameters.json`: the shared applicability model used for retrieval.

Gemini extracts semantic observations and writes candidate belief assessments.
It does not assign posterior parameters. Deterministic code computes:

```text
evidence weight =
  P(applicable)
  × P(strategy applied)
  × P(attributable)
  × P(reliable)
```

For improvement `y` in `[0, 1]`:

```text
alpha += weight × y
beta  += weight × (1 - y)
```

Evidence from one belief/topic pair is capped at a total weight of one so
dependent repair attempts do not appear to be independent replications.

## Topic-sequential reflection

The pipeline retains the original reflection structure:

```text
Topic 1 → one structured-JSON reflection call → update working bank
Topic 2 → one structured-JSON reflection call → update working bank
...
Topic N → one structured-JSON reflection call → update working bank
```

Each topic call receives its bounded topic evidence and the bank produced by
earlier topics. There is no function-call loop. The additions are structured
applicability/effectiveness evidence, strict action validation, automatic
Bayesian updates, malformed-JSON retry, separate evidence persistence, and
stored embeddings.

Every analysis records `reflection_calls` with prompt characters, actual API
input/output tokens, assessment count, and retry count.

## Build the first bank

After a belief-free pipeline has completed:

```bash
python src/mas_belief_reflection.py \
  --pipeline-dir mas_logs/pipeline_20260728_101242 \
  --write-library \
  --belief-embedding-model BAAI/bge-small-en-v1.5
```

This writes the three artifacts above into the pipeline directory.
It also writes:

```text
belief_embeddings.npz
belief_embedding_metadata.json
```

The NPZ contains one normalized vector per belief. Metadata records the model,
belief IDs, dimensions, and a hash of every embedded problem description.

## Use the bank in a later MAS run

```bash
python src/main.py \
  --runner mas \
  --run_pipeline \
  --knowledge_file json_files/mas_random_topics_30_seed_20260728.json \
  --inject_beliefs_into_prompts \
  --belief_library_path mas_logs/pipeline_20260728_101242/belief_library.json \
  --contextual_belief_selection \
  --belief_usefulness_threshold 0.60 \
  --top_specialized_beliefs 2
```

For local BGE similarity, add:

```bash
--belief_embedding_model BAAI/bge-small-en-v1.5
```

The model must already be available locally on a network-restricted compute
node. If a valid stored index exists beside the bank, it is loaded
automatically. If `--belief_embedding_model` is supplied and no valid index
exists, MAS builds and persists one before selection. Without an index or model,
the selector uses a deterministic lexical fallback.

Embeddings can also be rebuilt explicitly after editing a belief bank:

```bash
python src/build_belief_embeddings.py \
  --library mas_logs/pipeline_20260728_101242/belief_library.json \
  --model BAAI/bge-small-en-v1.5
```

At runtime, belief vectors are never recomputed when the stored text hashes
match. The current problem is embedded once per retrieval opportunity and
compared with the complete stored matrix.

Every runtime evaluation is appended to `belief_selections.jsonl` in the topic
run directory. Selection uses:

```text
usefulness =
  P(Applicable | RoleMatch, StageMatch, ProblemMatch, ContextMatch)
  × E[belief effectiveness]
```

At most `top_specialized_beliefs` above the threshold are injected. Selecting
none is valid.

## Versioning

Treat the generated bank and BBN parameters as frozen during a MAS batch:

```text
belief-free Run 1 → bank v1
Run 2 using bank v1 → bank v2
Run 3 using bank v2 → bank v3
```

The supplied `bbn_parameters.json` begins with documented expert-prior
coefficients. Calibrate those coefficients against a manually labelled
belief–situation relevance set before reporting them as learned parameters.

Each labelled JSONL row has:

```json
{"role_match": 1, "stage_match": 1, "problem_match": 0.87, "context_match": 0.75, "applicable": 1, "weight": 1}
```

Fit a new version with:

```bash
python src/calibrate_belief_bbn.py \
  --cases labelled_applicability.jsonl \
  --initial mas_logs/pipeline_20260728_101242/bbn_parameters.json \
  --output bbn_parameters_v2.json
```

The fitter shrinks coefficients toward the initial model, which limits
overfitting when the manually validated set is small.
