#!/usr/bin/env python3
"""Train a coarse MAS incident classifier with TF-IDF + Linear SVM."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency for training. Install requirements from "
        "`src/requirements.txt` so `scikit-learn` and `joblib` are available."
    ) from exc


def _default_dataset_path() -> Path:
    return Path(__file__).resolve().parent.parent / "datasets" / "mas_incidents" / "incidents_train.csv"


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "datasets" / "mas_incidents" / "model1_linear_svm"


def _default_repair_action_dataset_path() -> Path:
    return Path(__file__).resolve().parent.parent / "datasets" / "mas_incidents" / "repair_actions_train.csv"


def _default_repair_action_augmented_dataset_path() -> Path:
    return Path(__file__).resolve().parent.parent / "datasets" / "mas_incidents" / "repair_actions_train_augmented.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_default_dataset_path(),
        help="Path to incidents_train.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory to save the trained model and evaluation artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of rows to keep for the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test splitting.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=1,
        help="Minimum document frequency for TF-IDF features.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Upper bound of the n-gram range used by TF-IDF.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label_coarse",
        help="Column name to predict.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="input_text",
        help="Column name to use as model input.",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="",
        help="Optional column name for a grouped train/test split.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["incidents", "repair_actions", "repair_actions_augmented"],
        default="incidents",
        help="Convenience preset for dataset/label/output defaults.",
    )
    parser.add_argument(
        "--view-name",
        type=str,
        default="",
        help="Optional filter for datasets that include a view_name column, e.g. full_context, masked_exception, error_only.",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _validate_rows(rows: Sequence[Dict[str, str]], *, text_column: str, label_column: str) -> None:
    if not rows:
        raise ValueError("Dataset is empty.")

    missing_input = sum(1 for row in rows if not (row.get(text_column) or "").strip())
    missing_label = sum(1 for row in rows if not (row.get(label_column) or "").strip())
    if missing_input:
        raise ValueError(f"Dataset contains {missing_input} rows with empty {text_column}.")
    if missing_label:
        raise ValueError(f"Dataset contains {missing_label} rows with empty {label_column}.")


def _apply_preset_defaults(args: argparse.Namespace) -> None:
    if args.preset == "repair_actions":
        if args.dataset == _default_dataset_path():
            args.dataset = _default_repair_action_dataset_path()
        if args.output_dir == _default_output_dir():
            args.output_dir = _default_output_dir().with_name("repair_action_linear_svm")
        if args.label_column == "label_coarse":
            args.label_column = "repair_action"
    elif args.preset == "repair_actions_augmented":
        if args.dataset == _default_dataset_path():
            args.dataset = _default_repair_action_augmented_dataset_path()
        if args.output_dir == _default_output_dir():
            args.output_dir = _default_output_dir().with_name("repair_action_augmented_linear_svm")
        if args.label_column == "label_coarse":
            args.label_column = "repair_action"


def _build_pipeline(min_df: int, ngram_max: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, ngram_max),
                    min_df=min_df,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_predictions_csv(
    path: Path,
    *,
    test_rows: Sequence[Dict[str, str]],
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> None:
    fieldnames = [
        "incident_id",
        "label_true",
        "label_pred",
        "topic",
        "section_id",
        "section_title",
        "exception_type",
        "source_type",
        "input_text",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, true_label, pred_label in zip(test_rows, y_true, y_pred):
            writer.writerow(
                {
                    "incident_id": row.get("incident_id", ""),
                    "label_true": true_label,
                    "label_pred": pred_label,
                    "topic": row.get("topic", ""),
                    "section_id": row.get("section_id", ""),
                    "section_title": row.get("section_title", ""),
                    "exception_type": row.get("exception_type", ""),
                    "source_type": row.get("source_type", ""),
                    "input_text": row.get("input_text", ""),
                }
            )


def main() -> None:
    args = parse_args()
    _apply_preset_defaults(args)
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = _load_rows(dataset_path)
    if args.view_name:
        rows = [row for row in rows if str(row.get("view_name") or "").strip() == args.view_name]
        if not rows:
            raise ValueError(
                f"No rows matched --view-name {args.view_name!r} in dataset: {dataset_path}"
            )
        output_dir = output_dir.with_name(f"{output_dir.name}_{args.view_name}")
    _validate_rows(rows, text_column=args.text_column, label_column=args.label_column)

    labels = [row[args.label_column].strip() for row in rows]
    label_counts = Counter(labels)
    if min(label_counts.values()) < 2:
        raise ValueError(
            "At least one label has fewer than 2 examples, so a stratified train/test split is not safe: "
            f"{dict(label_counts)}"
        )

    input_texts = [row[args.text_column].strip() for row in rows]
    label_order = sorted(label_counts)
    groups = [str(row.get(args.group_column) or "").strip() for row in rows] if args.group_column else None

    if groups:
        non_empty_groups = [group for group in groups if group]
        if len(set(non_empty_groups)) < 2:
            raise ValueError(f"Need at least 2 distinct non-empty groups in column '{args.group_column}'.")
        grouped_items: Dict[str, Dict[str, object]] = {}
        for text, label, row, group in zip(input_texts, labels, rows, groups):
            group_key = group or f"__ungrouped__::{row.get('incident_id', len(grouped_items))}"
            payload = grouped_items.setdefault(
                group_key,
                {"label_counts": Counter(), "texts": [], "rows": []},
            )
            payload["label_counts"][label] += 1
            payload["texts"].append(text)
            payload["rows"].append(row)

        grouped_keys = list(grouped_items)
        grouped_labels = [
            grouped_items[key]["label_counts"].most_common(1)[0][0]
            for key in grouped_keys
        ]
        train_group_keys, test_group_keys = train_test_split(
            grouped_keys,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=grouped_labels,
        )
        train_group_set = set(train_group_keys)
        train_texts, test_texts, y_train, y_test, train_rows, test_rows = [], [], [], [], [], []
        for key in grouped_keys:
            target_lists = (
                (train_texts, y_train, train_rows)
                if key in train_group_set
                else (test_texts, y_test, test_rows)
            )
            bucket = grouped_items[key]
            for text, row in zip(bucket["texts"], bucket["rows"]):
                target_lists[0].append(text)
                target_lists[1].append(str(row[args.label_column]).strip())
                target_lists[2].append(row)
    else:
        (
            train_texts,
            test_texts,
            y_train,
            y_test,
            train_rows,
            test_rows,
        ) = train_test_split(
            input_texts,
            labels,
            rows,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=labels,
        )

    pipeline = _build_pipeline(args.min_df, args.ngram_max)
    pipeline.fit(train_texts, y_train)
    y_pred = pipeline.predict(test_texts)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=label_order, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=label_order)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metrics = {
        "generated_at_utc": timestamp,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "preset": args.preset,
        "view_name": args.view_name or None,
        "label_column": args.label_column,
        "text_column": args.text_column,
        "group_column": args.group_column or None,
        "row_count": len(rows),
        "train_count": len(train_texts),
        "test_count": len(test_texts),
        "labels": label_order,
        "label_counts": dict(label_counts),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "tfidf_min_df": args.min_df,
        "tfidf_ngram_range": [1, args.ngram_max],
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": {
            "labels": label_order,
            "matrix": matrix.tolist(),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "model.joblib")
    _write_json(output_dir / "metrics.json", metrics)
    _write_predictions_csv(output_dir / "test_predictions.csv", test_rows=test_rows, y_true=y_test, y_pred=y_pred)

    print(f"Saved model to: {output_dir / 'model.joblib'}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")
    print(f"Saved test predictions to: {output_dir / 'test_predictions.csv'}")
    print("")
    print(f"Rows: train={len(train_texts)} test={len(test_texts)} total={len(rows)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Label counts:", dict(label_counts))
    print("")
    print(classification_report(y_test, y_pred, labels=label_order, zero_division=0))


if __name__ == "__main__":
    main()
