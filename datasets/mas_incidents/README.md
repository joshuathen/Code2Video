# MAS Incident Dataset

Generated from `mas_logs` on `2026-06-16T04:35:29Z`.

## Files

- `all_incidents.jsonl`: Rich master dataset with metadata, labels, code excerpts, and training input text.
- `incidents_train.csv`: Flattened training export for Model 1.
- `repair_actions_train.csv`: One row per incident with a derived repair-action label.
- `repair_actions_train_augmented.csv`: Multi-view repair-action export for ablation/augmentation experiments.
- `summary.json`: Aggregate counts by source, label, and exception type.
- `repair_actions_summary.json`: Aggregate counts for derived repair actions and view variants.

## Counts

- Total incidents: `62`
- Gold incidents: `62`

## Model 1 Input

Each training row includes an `input_text` field with this format:

```text
Topic: ...
Section: ...
Exception: ...
Render status: ...
Timed out: ...

Error:
...

Code excerpt:
...
```

## Coarse Labels

- `environment_error`
- `performance_or_timeout`
- `name_error`
- `type_error`

## Repair Action Labels

- `fix_environment`
- `scope_refine_repair`
- `retry_with_timeout_or_perf_fix`
- `full_regenerate`

## Repair Action Augmentation Views

- `full_context`: topic + exception + code excerpt
- `masked_exception`: hides the explicit exception field
- `error_only`: only render status/timed-out/error text
