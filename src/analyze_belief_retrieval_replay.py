#!/usr/bin/env python3
"""Replay recorded belief-selection queries under the current selector."""

import argparse
import json
from pathlib import Path

from belief_bbn import (
    BBNParameters,
    BeliefEmbeddingIndex,
    BeliefSelector,
    BeliefSituation,
    exact_error_match,
)


def _belief_text(belief):
    scope = belief.get("scope") if isinstance(belief.get("scope"), dict) else {}
    return "\n".join(
        [
            str(scope.get("problem_description") or ""),
            str(belief.get("instruction") or ""),
            " ".join(str(item) for item in scope.get("context_conditions", [])),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline_dir", type=Path)
    parser.add_argument("belief_dir", type=Path)
    args = parser.parse_args()

    payload = json.loads((args.belief_dir / "belief_library.json").read_text())
    beliefs = payload["beliefs"]
    index = BeliefEmbeddingIndex.load(
        embeddings_path=args.belief_dir / "belief_embeddings.npz",
        metadata_path=args.belief_dir / "belief_embedding_metadata.json",
    )
    selector = BeliefSelector(
        beliefs,
        parameters=BBNParameters.from_path(args.belief_dir / "bbn_parameters.json"),
        embedding_index=index,
    )

    rows = []
    paths = sorted(args.pipeline_dir.glob("*/coder_debugger/*/belief_selections.jsonl"))
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            old = json.loads(line)
            situation_payload = old["situation"]
            if situation_payload.get("timing") != "reactive":
                continue
            situation_payload["pipeline_stage"] = "fix"
            situation = BeliefSituation(**situation_payload)
            expected = {
                belief["belief_id"]
                for belief in beliefs
                if exact_error_match(situation.problem_text, _belief_text(belief)) == 1.0
            }
            if not expected:
                continue
            old_ids = list(old.get("selected_belief_ids") or [])
            new = selector.select(situation, top_k=5, threshold=0.0, candidate_limit=82)
            new_ids = [item["belief_id"] for item in new]
            rows.append(
                {
                    "query": situation.problem_text.split("Current exception or failure:\n")[-1],
                    "expected": sorted(expected),
                    "old": old_ids,
                    "new": new_ids,
                }
            )

    def hit(row, ids):
        return bool(set(row["expected"]) & set(ids))

    print(f"Reactive queries with an identifier-exact belief: {len(rows)}")
    for label in ("old", "new"):
        top1 = sum(hit(row, row[label][:1]) for row in rows)
        top5 = sum(hit(row, row[label][:5]) for row in rows)
        print(f"{label}: exact belief Recall@1={top1}/{len(rows)}; Recall@5={top5}/{len(rows)}")
    for row in rows:
        print("\nQuery:", row["query"])
        print("Expected:", row["expected"])
        print("Old top 5:", row["old"][:5])
        print("New top 5:", row["new"][:5])


if __name__ == "__main__":
    main()
