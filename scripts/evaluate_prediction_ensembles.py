"""Evalúa ensambles ligeros sobre predicciones NMT ya generadas.

El proyecto es traducción automática, no clasificación. Por eso los ensambles
se implementan como blending/selección de candidatos a nivel de salida:

- Homogéneo: combina corridas NLLB de la misma familia (600M original y H100).
- Heterogéneo: combina NLLB y M2M-100 con una regla conservadora anti-degeneración.
- Oracle: cota superior no desplegable; usa la referencia para estimar cuánto
  margen habría si un reranker perfecto eligiera entre candidatos.

Uso:
    PYTHONPATH=src python scripts/evaluate_prediction_ensembles.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from voces_corpus.training.metrics import compute_translation_metrics


@dataclass(frozen=True)
class RunSpec:
    name: str
    kind: str
    predictions_path: Path
    metrics_path: Path | None = None
    hardware: str = "no registrado"
    train_time: str = "no registrado"


RUNS = [
    RunSpec(
        name="NLLB orig · 600M · 3ep lr5e-4",
        kind="individual",
        predictions_path=Path("outputs/test_predictions.jsonl"),
        metrics_path=Path("outputs/test_metrics.json"),
        hardware="Colab T4",
        train_time="30-45 min reportado",
    ),
    RunSpec(
        name="NLLB H100 · 600M · 8ep lr2e-4",
        kind="individual",
        predictions_path=Path("outputs_nllb_h100/test_predictions.jsonl"),
        metrics_path=Path("outputs_nllb_h100/test_metrics.json"),
        hardware="H100",
        train_time="no registrado",
    ),
    RunSpec(
        name="M2M-100 · 418M · 3ep lr5e-4",
        kind="individual",
        predictions_path=Path("outputs_m2m100/test_predictions.jsonl"),
        metrics_path=Path("outputs_m2m100/test_metrics.json"),
        hardware="H100",
        train_time="no registrado",
    ),
]


def load_predictions(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def validate_alignment(runs: dict[str, list[dict[str, str]]]) -> list[str]:
    first_name = next(iter(runs))
    first = runs[first_name]
    warnings: list[str] = []
    for name, records in runs.items():
        if len(records) != len(first):
            raise ValueError(f"{name} tiene {len(records)} registros; {first_name} tiene {len(first)}")
        for idx, (a, b) in enumerate(zip(first, records)):
            if a["direction"] != b["direction"]:
                raise ValueError(f"{name} no está alineado con {first_name} en índice {idx}")
            if a["reference"] != b["reference"]:
                warnings.append(
                    f"{name}: referencia distinta a {first_name} en índice {idx}; "
                    "los ensambles se evaluarán con la referencia base."
                )
                break
    return warnings


def repetition_ratio(text: str) -> float:
    tokens = re.findall(r"\S+", text.lower())
    if not tokens:
        return 1.0
    if len(tokens) == 1:
        return 0.0
    repeated_bigrams = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    unique_ratio = len(set(tokens)) / len(tokens)
    return max(repeated_bigrams / (len(tokens) - 1), 1.0 - unique_ratio)


def char_loop_score(text: str) -> float:
    compact = re.sub(r"\s+", "", text.lower())
    if not compact:
        return 1.0
    # Detecta secuencias cortas repetidas, comunes en colapsos seq2seq.
    score = 0.0
    for width in range(1, 5):
        for match in re.finditer(rf"(.{{{width}}})\1{{3,}}", compact):
            score = max(score, len(match.group(0)) / max(len(compact), 1))
    return score


def degeneracy_score(text: str) -> float:
    tokens = re.findall(r"\S+", text)
    length_penalty = max(0.0, (len(tokens) - 80) / 80)
    return repetition_ratio(text) + char_loop_score(text) + length_penalty


def is_degenerate(text: str) -> bool:
    return degeneracy_score(text) >= 0.45


def candidate_quality_without_reference(text: str, candidate_lengths: list[int]) -> float:
    length = len(re.findall(r"\S+", text))
    median_length = sorted(candidate_lengths)[len(candidate_lengths) // 2]
    if median_length == 0:
        length_distance = 1.0 if length else 0.0
    else:
        length_distance = abs(length - median_length) / median_length
    return -(2.5 * degeneracy_score(text) + 0.25 * length_distance)


def corpus_metrics(records: list[dict[str, str]]) -> dict[str, float]:
    preds = [r["prediction"] for r in records]
    refs = [r["reference"] for r in records]
    return compute_translation_metrics(preds, refs)


def average_directional_metrics(records: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for direction in ["es->bri", "bri->es"]:
        subset = [r for r in records if r["direction"] == direction]
        result[direction] = corpus_metrics(subset)
    result["avg"] = {
        key: (result["es->bri"][key] + result["bri->es"][key]) / 2
        for key in ["spbleu", "chrf", "chrfpp"]
    }
    return result


def sentence_chrfpp(prediction: str, reference: str) -> float:
    return compute_translation_metrics([prediction], [reference])["chrfpp"]


def build_selection(
    base_records: list[dict[str, str]],
    candidate_runs: list[list[dict[str, str]]],
    selector: Callable[[int, list[dict[str, str]]], int],
) -> list[dict[str, str]]:
    selected = []
    for idx, base in enumerate(base_records):
        candidates = [run[idx] for run in candidate_runs]
        choice = selector(idx, candidates)
        selected.append(
            {
                "direction": base["direction"],
                "prediction": candidates[choice]["prediction"],
                # La referencia base evita mezclar pequeñas diferencias de
                # normalización entre corridas al evaluar un mismo candidato.
                "reference": base["reference"],
            }
        )
    return selected


def write_jsonl(records: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def markdown_table(rows: list[dict[str, object]]) -> str:
    headers = ["Modelo", "Tipo", "Hardware", "Tiempo", "spBLEU", "chrF", "chrF++"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {kind} | {hardware} | {train_time} | {spbleu:.2f} | {chrf:.2f} | {chrfpp:.2f} |".format(
                **row
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_ensembles"))
    args = parser.parse_args()

    available_specs = [spec for spec in RUNS if spec.predictions_path.exists()]
    if len(available_specs) < 2:
        raise FileNotFoundError("Se requieren al menos dos archivos test_predictions.jsonl para evaluar ensambles.")

    loaded = {spec.name: load_predictions(spec.predictions_path) for spec in available_specs}
    alignment_warnings = validate_alignment(loaded)

    nllb_orig = loaded["NLLB orig · 600M · 3ep lr5e-4"]
    nllb_h100 = loaded["NLLB H100 · 600M · 8ep lr2e-4"]
    m2m100 = loaded.get("M2M-100 · 418M · 3ep lr5e-4")

    ensembles: dict[str, tuple[str, list[dict[str, str]], str, str]] = {}

    homogeneous = build_selection(
        nllb_h100,
        [nllb_h100, nllb_orig],
        lambda _idx, cands: 1 if is_degenerate(cands[0]["prediction"]) and not is_degenerate(cands[1]["prediction"]) else 0,
    )
    ensembles["Blend homogéneo NLLB anti-degeneración"] = (
        "ensamble homogéneo",
        homogeneous,
        "H100 + T4",
        "post-proceso; sin reentrenar",
    )

    if m2m100 is not None:
        def hetero_selector(_idx: int, candidates: list[dict[str, str]]) -> int:
            lengths = [len(re.findall(r"\S+", c["prediction"])) for c in candidates]
            scored = [candidate_quality_without_reference(c["prediction"], lengths) for c in candidates]
            return max(range(len(candidates)), key=lambda i: scored[i])

        heterogeneous = build_selection(nllb_h100, [nllb_h100, nllb_orig, m2m100], hetero_selector)
        ensembles["Blend heterogéneo NLLB+M2M anti-degeneración"] = (
            "ensamble heterogéneo",
            heterogeneous,
            "H100 + T4",
            "post-proceso; sin reentrenar",
        )

        oracle = build_selection(
            nllb_h100,
            [nllb_h100, nllb_orig, m2m100],
            lambda _idx, cands: max(
                range(len(cands)),
                key=lambda i: sentence_chrfpp(cands[i]["prediction"], cands[i]["reference"]),
            ),
        )
        ensembles["Oracle de candidatos NLLB+M2M (cota superior no desplegable)"] = (
            "oracle / upper bound",
            oracle,
            "H100 + T4",
            "usa referencia; no desplegable",
        )

    rows: list[dict[str, object]] = []
    all_results: dict[str, object] = {"individual": {}, "ensembles": {}}

    for spec in available_specs:
        metrics = average_directional_metrics(loaded[spec.name])
        all_results["individual"][spec.name] = metrics
        rows.append(
            {
                "model": spec.name,
                "kind": spec.kind,
                "hardware": spec.hardware,
                "train_time": spec.train_time,
                **metrics["avg"],
            }
        )

    for name, (kind, records, hardware, train_time) in ensembles.items():
        metrics = average_directional_metrics(records)
        write_jsonl(records, args.out_dir / f"{slugify(name)}.jsonl")
        all_results["ensembles"][name] = metrics
        rows.append(
            {
                "model": name,
                "kind": kind,
                "hardware": hardware,
                "train_time": train_time,
                **metrics["avg"],
            }
        )

    rows.sort(key=lambda row: float(row["chrfpp"]), reverse=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ensemble_metrics.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    (args.out_dir / "comparison_table.md").write_text(markdown_table(rows) + "\n")
    if alignment_warnings:
        (args.out_dir / "alignment_warnings.txt").write_text("\n".join(alignment_warnings) + "\n")
        for warning in alignment_warnings:
            print(f"ADVERTENCIA: {warning}")
    print(markdown_table(rows))
    print(f"\nGuardado en {args.out_dir}")


def slugify(text: str) -> str:
    text = text.lower().replace("+", " plus ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


if __name__ == "__main__":
    main()
