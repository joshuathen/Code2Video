#!/usr/bin/env python3
"""Fit BBN applicability parameters from manually validated JSONL cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from belief_bbn import BBNParameters, fit_bbn_parameters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="Labelled applicability JSONL.")
    parser.add_argument("--output", required=True, help="Output bbn_parameters.json.")
    parser.add_argument("--initial", default=None, help="Optional existing parameters.")
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    cases = []
    with Path(args.cases).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))

    initial = BBNParameters.from_path(Path(args.initial)) if args.initial else BBNParameters()
    fitted = fit_bbn_parameters(cases, initial=initial, epochs=args.epochs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(fitted.to_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote BBN parameters fitted from {len(cases)} cases to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
