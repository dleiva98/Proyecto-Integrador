"""Cálculo de métricas para traducción bri<->es.

Usa sacreBLEU (tokenize=flores200) + chrF y chrF++. Pensado para invocarse
desde el bucle de entrenamiento y también como CLI sobre un JSONL de
predicciones generadas a posteriori.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sacrebleu


def compute_translation_metrics(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Devuelve un dict con spBLEU (flores200), chrF y chrF++."""
    refs = [references]
    spbleu = sacrebleu.corpus_bleu(predictions, refs, tokenize="flores200").score
    chrf = sacrebleu.corpus_chrf(predictions, refs).score
    chrfpp = sacrebleu.corpus_chrf(predictions, refs, word_order=2).score
    return {"spbleu": spbleu, "chrf": chrf, "chrfpp": chrfpp}


def _load_pairs(path: Path) -> tuple[list[str], list[str]]:
    preds: list[str] = []
    refs: list[str] = []
    with path.open() as fh:
        for line in fh:
            obj = json.loads(line)
            preds.append(obj["prediction"])
            refs.append(obj["reference"])
    return preds, refs


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions_jsonl",
        type=Path,
        help="JSONL con campos {prediction, reference}.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Escribe JSON con resultados.")
    args = parser.parse_args()

    preds, refs = _load_pairs(args.predictions_jsonl)
    scores: dict[str, Any] = compute_translation_metrics(preds, refs)
    scores["n"] = len(preds)
    print(json.dumps(scores, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(scores, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
