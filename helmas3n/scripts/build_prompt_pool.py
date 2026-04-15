#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _arith_prompt(rng: random.Random, idx: int) -> dict[str, str]:
    a = rng.randint(10, 999)
    b = rng.randint(10, 999)
    c = rng.randint(2, 25)
    d = rng.randint(2, 25)
    return {
        "id": f"arith_{idx:04d}",
        "category": "arithmetic",
        "prompt": (
            "Solve carefully and show brief reasoning.\n"
            f"Compute (({a} + {b}) * {c}) - ({a} // {d}).\n"
            "Return the final integer only on the last line."
        ),
    }


def _logic_prompt(rng: random.Random, idx: int) -> dict[str, str]:
    names = ["Ari", "Bo", "Cy", "Dee", "Eli", "Fox", "Gia", "Hal"]
    x, y, z = rng.sample(names, 3)
    return {
        "id": f"logic_{idx:04d}",
        "category": "logic",
        "prompt": (
            "Determine whether the conclusion is entailed, contradicted, or unknown.\n"
            f"Premise 1: If {x} studies, then {y} passes.\n"
            f"Premise 2: If {y} passes, then {z} celebrates.\n"
            f"Premise 3: {x} studies.\n"
            f"Conclusion: {z} celebrates.\n"
            "Answer with one token: entailed, contradicted, or unknown."
        ),
    }


def _sequence_prompt(rng: random.Random, idx: int) -> dict[str, str]:
    start = rng.randint(1, 9)
    step = rng.randint(2, 8)
    length = rng.randint(6, 8)
    seq = [start + i * step for i in range(length)]
    return {
        "id": f"seq_{idx:04d}",
        "category": "sequence",
        "prompt": (
            "Infer the pattern and provide the next two numbers.\n"
            f"Sequence: {', '.join(str(x) for x in seq)}\n"
            "Return exactly: <n1>, <n2>"
        ),
    }


def _symbolic_prompt(rng: random.Random, idx: int) -> dict[str, str]:
    letters = list("abcdefxyz")
    p, q, r = rng.sample(letters, 3)
    return {
        "id": f"sym_{idx:04d}",
        "category": "symbolic",
        "prompt": (
            "Simplify the boolean expression and provide a compact equivalent.\n"
            f"Expression: ({p} AND {q}) OR ({p} AND NOT {q}) OR ({r} AND FALSE)\n"
            "Use operators AND, OR, NOT and minimal form."
        ),
    }


def _constraint_prompt(rng: random.Random, idx: int) -> dict[str, str]:
    words = ["amber", "brisk", "cinder", "delta", "ember", "fable", "glint", "harbor"]
    a, b, c = rng.sample(words, 3)
    return {
        "id": f"con_{idx:04d}",
        "category": "constraint",
        "prompt": (
            "Create one sentence that satisfies all constraints:\n"
            f"1) Contains the words '{a}', '{b}', and '{c}'.\n"
            "2) Exactly 12 words total.\n"
            "3) Ends with a period.\n"
            "Return only the sentence."
        ),
    }


def build_pool(total: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    gens = [
        _arith_prompt,
        _logic_prompt,
        _sequence_prompt,
        _symbolic_prompt,
        _constraint_prompt,
    ]
    items: list[dict[str, str]] = []
    for i in range(total):
        fn = gens[i % len(gens)]
        items.append(fn(rng, i))
    rng.shuffle(items)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a synthetic prompt pool for RegimeLift kill-tests")
    parser.add_argument("--out", type=str, required=True, help="Output .jsonl path")
    parser.add_argument("--total", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_pool(total=args.total, seed=args.seed)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(json.dumps({"out": str(out_path), "total": len(rows), "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
