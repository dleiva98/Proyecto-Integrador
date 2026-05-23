"""Divide corpus_v0.jsonl en train/val/test estratificado por (domain, confidence).

Salida en data/splits/{train,val,test}.jsonl.

Uso:
    python scripts/make_splits.py [--val 0.1] [--test 0.1] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "corpus_v0.jsonl"
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def load_corpus(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stratified_split(
    pairs: list[dict], val_frac: float, test_frac: float, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in pairs:
        buckets[(p["domain"], p["confidence"])].append(p)

    train, val, test = [], [], []
    for key, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_frac))) if n - n_test >= 2 else 0
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--out", type=Path, default=SPLITS_DIR)
    args = parser.parse_args()

    pairs = load_corpus(args.corpus)
    train, val, test = stratified_split(pairs, args.val, args.test, args.seed)

    write_jsonl(train, args.out / "train.jsonl")
    write_jsonl(val, args.out / "val.jsonl")
    write_jsonl(test, args.out / "test.jsonl")

    total = len(pairs)
    print(f"Corpus total: {total} pares")
    print(f"  train: {len(train)} ({len(train) / total:.1%})")
    print(f"  val:   {len(val)} ({len(val) / total:.1%})")
    print(f"  test:  {len(test)} ({len(test) / total:.1%})")
    print(f"Escrito en {args.out}/")


if __name__ == "__main__":
    main()
