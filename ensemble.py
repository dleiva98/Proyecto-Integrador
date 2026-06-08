"""Avance 5 — Modelos de ensamble para NMT bribri<->español.

La rúbrica de ensamble (stacking, blending, voting) está formulada para
clasificación/regresión tabular. En traducción automática neuronal —tarea de
generación de secuencias— las estrategias de ensamble existen pero se
materializan de otra forma. Este script implementa las dos familias pedidas:

  1. ENSAMBLE HOMOGÉNEO  ·  checkpoint averaging
     Promedia los pesos de los N mejores checkpoints de la MISMA arquitectura
     (NLLB-200 H100). Equivale al "bagging" de modelos: reduce la varianza del
     punto final de entrenamiento sin coste de inferencia adicional (un solo
     modelo resultante). Sólo es válido entre checkpoints del mismo modelo y
     vocabulario.

  2. ENSAMBLE HETEROGÉNEO  ·  system combination por MBR (Minimum Bayes Risk)
     NLLB-200 y M2M-100 tienen vocabularios distintos, así que NO se pueden
     promediar logits token a token. La combinación se hace a nivel de
     hipótesis: cada sistema genera su n-best, se juntan en un pool y se elige
     por sentencia la hipótesis de "consenso" (máxima chrF promedio contra el
     resto del pool). Es la analogía en NMT del *voting*/*blending*: no requiere
     referencia y aprovecha el mejor modelo individual (NLLB) como ancla.

Los pesos fine-tuneados NO están en el repo (~2.4 GB, en .gitignore). Este
script se corre en el servidor H100 donde viven los checkpoints
(`outputs_nllb_h100/`, `outputs_m2m100/`). Reusa las MISMAS métricas que el
resto del proyecto (`voces_corpus.training.metrics`) para que la tabla
comparativa del Avance 5 sea consistente con las fases previas.

Uso:
    export PYTHONPATH="$PWD/src:$PYTHONPATH"

    # 1. Ensamble homogéneo: promedia los checkpoints best_* de NLLB H100
    python ensemble.py avg --ckpt-dir outputs_nllb_h100 --eval

    # 2. Ensamble heterogéneo: combina NLLB H100 + M2M-100
    python ensemble.py combine \
        --nllb outputs_nllb_h100/final_nllb \
        --m2m  outputs_m2m100/final_m2m100 --eval
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from voces_corpus.training.metrics import compute_translation_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NLLB_TOK = {"es": "spa_Latn", "bri": "quy_Latn"}   # tokens NLLB
M2M_CODE = {"es": "es", "bri": "br"}               # códigos M2M (br = proxy)
OUT = Path("outputs_ensemble")


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


# ============================================================= GENERACIÓN
@torch.no_grad()
def generate(model, tok, sentences, src, tgt, is_m2m, num_beams=5,
             n_best=1, max_new_tokens=256):
    """Traduce una lista de oraciones. Devuelve list[str] (n_best=1) o
    list[list[str]] (n_best>1, para MBR)."""
    model.eval()
    out_all = []
    for s in sentences:
        if is_m2m:
            tok.src_lang = M2M_CODE[src]
            forced = tok.get_lang_id(M2M_CODE[tgt])
        else:
            tok.src_lang = NLLB_TOK[src]
            forced = tok.convert_tokens_to_ids(NLLB_TOK[tgt])
        enc = tok(s, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        gen = model.generate(
            **enc, forced_bos_token_id=forced, num_beams=num_beams,
            num_return_sequences=n_best, max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
        )
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        out_all.append(dec[0] if n_best == 1 else dec)
    return out_all


# ============================================== 1. ENSAMBLE HOMOGÉNEO
def average_checkpoints(ckpt_dir: Path, out_dir: Path, topk: int):
    """Promedia los pesos de los `topk` checkpoints best_nllb_spbleu=* (mayor
    spBLEU). Guarda un único modelo en out_dir."""
    ckpts = sorted(ckpt_dir.glob("best_nllb_spbleu=*"),
                   key=lambda p: float(p.name.split("=")[-1]), reverse=True)[:topk]
    if not ckpts:
        raise SystemExit(f"No hay checkpoints best_* en {ckpt_dir}")
    print(f"Promediando {len(ckpts)} checkpoints:")
    for c in ckpts:
        print("  ", c.name)

    base = AutoModelForSeq2SeqLM.from_pretrained(ckpts[0])
    avg = {k: v.clone().float() for k, v in base.state_dict().items()}
    for c in ckpts[1:]:
        sd = AutoModelForSeq2SeqLM.from_pretrained(c).state_dict()
        for k in avg:
            avg[k] += sd[k].float()
    for k in avg:
        avg[k] /= len(ckpts)
    base.load_state_dict({k: v.to(base.state_dict()[k].dtype) for k, v in avg.items()})

    out_dir.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(ckpts[0]).save_pretrained(out_dir)
    print(f"Modelo promediado guardado en {out_dir}")
    return out_dir


# ============================================ 2. ENSAMBLE HETEROGÉNEO
def mbr_select(pool: list[str]) -> str:
    """Minimum Bayes Risk con chrF como utilidad: elige la hipótesis con mayor
    chrF promedio contra todas las demás del pool (consenso, sin referencia)."""
    if len(pool) == 1:
        return pool[0]
    best, best_score = pool[0], -1.0
    for i, cand in enumerate(pool):
        others = [pool[j] for j in range(len(pool)) if j != i]
        score = sum(sacrebleu.sentence_chrf(cand, [o]).score for o in others) / len(others)
        if score > best_score:
            best, best_score = cand, score
    return best


def combine_systems(nllb, m2m, sentences, src, tgt, n_best=5):
    """Combina NLLB + M2M por MBR a nivel de hipótesis."""
    model_n, tok_n = nllb
    model_m, tok_m = m2m
    nbest_n = generate(model_n, tok_n, sentences, src, tgt, False, n_best=n_best)
    nbest_m = generate(model_m, tok_m, sentences, src, tgt, True, n_best=n_best)
    return [mbr_select(list(a) + list(b)) for a, b in zip(nbest_n, nbest_m)]


# ===================================================== EVALUACIÓN
def evaluate(predictions, references, name, train_time=None):
    scores = compute_translation_metrics(predictions, references)
    row = {"model": name, **scores}
    if train_time is not None:
        row["train_time_s"] = train_time
    print(f"\n=== {name} ===")
    print(json.dumps(scores, indent=2))
    return row


def eval_on_test(translate_fn, label):
    """translate_fn(sentences, src, tgt) -> list[str]. Evalúa ambas direcciones."""
    test = read_jsonl("data/splits/test.jsonl")
    es = [r["es"] for r in test]
    bri = [r["bri"] for r in test]
    s2t = evaluate(translate_fn(es, "es", "bri"), bri, f"{label} · es->bri")
    t2s = evaluate(translate_fn(bri, "bri", "es"), es, f"{label} · bri->es")
    avg = {k: (s2t[k] + t2s[k]) / 2 for k in ("spbleu", "chrf", "chrfpp")}
    print(f"\n{label} · AVG: chrF={avg['chrf']:.2f} spBLEU={avg['spbleu']:.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{label.replace(' ', '_')}_test.json").write_text(
        json.dumps({"es->bri": s2t, "bri->es": t2s, "avg": avg}, indent=2, ensure_ascii=False))
    return avg


# ===================================================== CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("avg", help="ensamble homogéneo (checkpoint averaging NLLB)")
    a.add_argument("--ckpt-dir", default="outputs_nllb_h100")
    a.add_argument("--out", default="outputs_ensemble/nllb_avg")
    a.add_argument("--topk", type=int, default=3)
    a.add_argument("--eval", action="store_true")

    c = sub.add_parser("combine", help="ensamble heterogéneo (NLLB + M2M, MBR)")
    c.add_argument("--nllb", default="outputs_nllb_h100/final_nllb")
    c.add_argument("--m2m", default="outputs_m2m100/final_m2m100")
    c.add_argument("--n-best", type=int, default=5)
    c.add_argument("--eval", action="store_true")

    args = ap.parse_args()

    if args.cmd == "avg":
        t0 = time.time()
        out = average_checkpoints(Path(args.ckpt_dir), Path(args.out), args.topk)
        print(f"averaging tomó {time.time() - t0:.1f}s")
        if args.eval:
            model = AutoModelForSeq2SeqLM.from_pretrained(out).to(DEVICE)
            tok = AutoTokenizer.from_pretrained(out)
            eval_on_test(lambda s, src, tgt: generate(model, tok, s, src, tgt, False),
                         "NLLB_avg")

    elif args.cmd == "combine":
        nllb = (AutoModelForSeq2SeqLM.from_pretrained(args.nllb).to(DEVICE),
                AutoTokenizer.from_pretrained(args.nllb))
        m2m = (AutoModelForSeq2SeqLM.from_pretrained(args.m2m).to(DEVICE),
               AutoTokenizer.from_pretrained(args.m2m))
        if args.eval:
            eval_on_test(
                lambda s, src, tgt: combine_systems(nllb, m2m, s, src, tgt, args.n_best),
                "NLLB+M2M_MBR")


if __name__ == "__main__":
    main()
