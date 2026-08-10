#!/usr/bin/env python3
"""Minimal smoke test for Gemini 2.5 Pro using the primary API key."""

from mas_interactions import request_interaction_text, response_usage_dict


MODEL = "gemini-2.5-pro"
PROMPT = "hellow world"


def main() -> int:
    try:
        response = request_interaction_text(
            prompt=PROMPT,
            model_name=MODEL,
            max_tokens=64,
            max_retries=0,
            use_eval_credentials=False,
        )
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("SUCCESS")
    print(f"Requested model: {MODEL}")
    print(f"Resolved model: {response.model or '<not returned>'}")
    print(f"Response: {response.text.strip()}")
    print(f"Usage: {response_usage_dict(response)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
