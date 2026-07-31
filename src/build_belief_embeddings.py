#!/usr/bin/env python3
"""Build a persisted BGE embedding index for a belief library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from belief_bbn import BeliefEmbeddingIndex


def load_beliefs(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    beliefs = payload.get("beliefs", payload.get("lessons", payload))
    if isinstance(beliefs, dict):
        return list(beliefs.values())
    if isinstance(beliefs, list):
        return beliefs
    raise ValueError(f"Unsupported belief library structure in {path}")


def build_index(
    library_path: Path,
    *,
    model_name_or_path: str,
    embeddings_path: Path,
    metadata_path: Path,
) -> BeliefEmbeddingIndex:
    return BeliefEmbeddingIndex.build(
        load_beliefs(library_path),
        model_name_or_path=model_name_or_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, help="Path to belief_library.json.")
    parser.add_argument("--model", required=True, help="Local path or sentence-transformers model.")
    parser.add_argument("--embeddings-output", default=None)
    parser.add_argument("--metadata-output", default=None)
    args = parser.parse_args()

    library_path = Path(args.library).expanduser().resolve()
    embeddings_path = (
        Path(args.embeddings_output).expanduser().resolve()
        if args.embeddings_output
        else library_path.with_name("belief_embeddings.npz")
    )
    metadata_path = (
        Path(args.metadata_output).expanduser().resolve()
        if args.metadata_output
        else library_path.with_name("belief_embedding_metadata.json")
    )
    index = build_index(
        library_path,
        model_name_or_path=args.model,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )
    print(
        f"Stored {len(index.belief_ids)} belief embeddings in {embeddings_path} "
        f"using {index.model_name}"
    )
    print(f"Embedding metadata written to: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
