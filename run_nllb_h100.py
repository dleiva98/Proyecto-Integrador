"""Launcher para re-entrenar NLLB-200 con más épocas y lr más bajo.

Experimento APARTE para intentar mejorar el baseline NLLB. NO toca
nllb_train.py: importa train() y TrainConfig() y solo cambia los
hiperparámetros vía el dataclass.

Guarda en outputs_nllb_h100/ para no pisar la corrida original (outputs/).

Cambios vs corrida original:
  epochs: 3 -> 8
  lr:     5e-4 -> 2e-4   (menos overfitting con 1201 pares)
  todo lo demás idéntico (seed, splits, max_length, batch_size, métricas)

Uso:
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
    python run_nllb_h100.py
"""
import json
from pathlib import Path

from voces_corpus.training.nllb_train import TrainConfig, train, evaluate_split, _read_jsonl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from voces_corpus.training.nllb_train import TranslationDataset
from torch.utils.data import DataLoader
import torch

OUT = Path("outputs_nllb_h100")

cfg = TrainConfig(
    epochs=8,
    lr=2e-4,
    output_dir=OUT,
    # batch_size, max_length, seed, tokens de idioma: se quedan por defecto
)

print(f"=== NLLB-200 H100 · epochs={cfg.epochs} lr={cfg.lr} ===")
history = train(cfg, repo_root=Path.cwd())

# --- Evaluación sobre TEST (espejo del notebook de NLLB) ---
print("\n=== Evaluación sobre TEST ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = Path.cwd() / OUT
model = AutoModelForSeq2SeqLM.from_pretrained(out_dir / "final_nllb").to(device)
tokenizer = AutoTokenizer.from_pretrained(out_dir / "final_nllb")

test_data = _read_jsonl(Path("data/splits/test.jsonl"))
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                  padding="longest", max_length=cfg.max_length,
                                  pad_to_multiple_of=8)

def loader(src_lang, tgt_lang, src_tok, tgt_tok):
    ds = TranslationDataset(test_data, src_lang, tgt_lang, src_tok, tgt_tok,
                            False, tokenizer, cfg.max_length)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)

s2t = evaluate_split(model, tokenizer,
                     loader(cfg.src_lang, cfg.tgt_lang, cfg.src_lang_token, cfg.tgt_lang_token),
                     cfg.tgt_lang_token, device, cfg.max_length)
t2s = evaluate_split(model, tokenizer,
                     loader(cfg.tgt_lang, cfg.src_lang, cfg.tgt_lang_token, cfg.src_lang_token),
                     cfg.src_lang_token, device, cfg.max_length)

summary = {
    "model": cfg.model_str,
    "run": "nllb_h100_8ep_lr2e-4",
    "es->bri": {k: s2t[k] for k in ("eval_loss", "spbleu", "chrf", "chrfpp")},
    "bri->es": {k: t2s[k] for k in ("eval_loss", "spbleu", "chrf", "chrfpp")},
    "avg": {
        "eval_loss": (s2t["eval_loss"] + t2s["eval_loss"]) / 2,
        "spbleu": (s2t["spbleu"] + t2s["spbleu"]) / 2,
        "chrf": (s2t["chrf"] + t2s["chrf"]) / 2,
        "chrfpp": (s2t["chrfpp"] + t2s["chrfpp"]) / 2,
    },
    "n_test": len(test_data),
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
(out_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
with (out_dir / "test_predictions.jsonl").open("w") as fh:
    for direction, r in (("es->bri", s2t), ("bri->es", t2s)):
        for pred, ref in zip(r["predictions"], r["references"]):
            fh.write(json.dumps({"direction": direction, "prediction": pred,
                                 "reference": ref}, ensure_ascii=False) + "\n")
print(f"\nGuardado en {out_dir}/")
