# MAS Incident Dataset

Generated from `mas_logs` on `2026-06-09T02:22:49Z`.

## Files

- `all_incidents.jsonl`: Rich master dataset with metadata, labels, code excerpts, and training input text.
- `incidents_train.csv`: Flattened training export for Model 1.
- `summary.json`: Aggregate counts by source, label, and exception type.

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
