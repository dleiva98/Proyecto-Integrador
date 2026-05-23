"""Fine-tuning de NLLB-200-distilled-600M sobre el corpus bribri<->español.

Adaptado del template oficial del Proyecto Integrador (a9c6fdf2-nllb_training.py).

Cambios respecto al template:
- Par `bri`/`es` con `quy_Latn` (Quechua Ayacucho) como proxy para bribri y
  `spa_Latn` para español. La elección de proxy es discutible; ver README,
  sección "Avance 2".
- Carga splits desde `data/splits/{train,val,test}.jsonl`.
- Loguea train_losses, val_losses (src2tgt, tgt2src, avg), spBLEU, chrF y
  chrF++ a `outputs/metrics.json` cada `EVAL_EVERY` pasos.
- `tqdm` no-notebook por defecto.
- Pensado para ejecutarse en Colab con GPU; ver `notebooks/nllb_finetune_colab.ipynb`.
"""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
)

from voces_corpus.training.metrics import compute_translation_metrics


# =========== CONFIG ===========
@dataclass
class TrainConfig:
    model_str: str = "facebook/nllb-200-distilled-600M"
    tokenizer_str: str = "facebook/nllb-200-distilled-600M"

    lr: float = 5e-4
    eps: float = 1e-8
    weight_decay: float = 0.01
    batch_size: int = 8
    epochs: int = 3

    bidirectional: bool = True
    max_length: int = 256

    log_every: int = 10
    eval_every: int = 100
    save_model_on_evaluation: bool = True

    use_float16: bool = True
    seed: int = 42

    # Par del proyecto: bri <-> es
    src_lang: str = "es"
    tgt_lang: str = "bri"
    src_lang_token: str = "spa_Latn"
    tgt_lang_token: str = "quy_Latn"  # proxy para bribri (sin token propio en NLLB)

    # Rutas (relativas al repo root)
    train_data_path: Path = field(default=Path("data/splits/train.jsonl"))
    val_data_path: Path = field(default=Path("data/splits/val.jsonl"))
    test_data_path: Path = field(default=Path("data/splits/test.jsonl"))
    output_dir: Path = field(default=Path("outputs"))


# =========== DATASET ===========
class TranslationDataset(Dataset):
    def __init__(
        self,
        data: list[dict],
        src_lang: str,
        tgt_lang: str,
        src_lang_token: str,
        tgt_lang_token: str,
        bidirectional: bool,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ):
        self.data = data
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_lang_token = src_lang_token
        self.tgt_lang_token = tgt_lang_token
        self.bidirectional = bidirectional
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.data[idx % len(self.data)]
        if idx < len(self.data):
            src_lang, tgt_lang = self.src_lang, self.tgt_lang
            src_tok, tgt_tok = self.src_lang_token, self.tgt_lang_token
        else:
            src_lang, tgt_lang = self.tgt_lang, self.src_lang
            src_tok, tgt_tok = self.tgt_lang_token, self.src_lang_token

        self.tokenizer.src_lang = src_tok
        self.tokenizer.tgt_lang = tgt_tok

        inputs = self.tokenizer(
            item[src_lang],
            text_target=item[tgt_lang],
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        for v in inputs.values():
            v.squeeze_(0)
        return inputs

    def __len__(self):
        return len(self.data) * (2 if self.bidirectional else 1)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


# =========== EVAL ===========
@torch.no_grad()
def evaluate_split(
    model,
    tokenizer,
    dataloader,
    tgt_token: str,
    device: str,
    max_length: int,
) -> dict[str, float]:
    model.eval()
    val_losses: list[float] = []
    predictions: list[str] = []
    references: list[str] = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        val_inputs = batch.to(device)
        val_outputs = model(**val_inputs, use_cache=False)
        val_losses.append(val_outputs.loss.item())

        labels = val_inputs.pop("labels")
        labels[labels == -100] = tokenizer.pad_token_id
        batch_pred = model.generate(
            **val_inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_token),
            max_new_tokens=max_length,
        )
        predictions.extend(
            tokenizer.batch_decode(batch_pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        )
        references.extend(
            tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        )

    model.train()
    eval_loss = sum(val_losses) / max(len(val_losses), 1)
    scores = compute_translation_metrics(predictions, references)
    return {"eval_loss": eval_loss, **scores, "predictions": predictions, "references": references}


# =========== TRAIN ===========
def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def train(cfg: TrainConfig, repo_root: Optional[Path] = None) -> dict:
    repo_root = Path(repo_root) if repo_root else Path.cwd()

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else (repo_root / p)

    train_path = _resolve(cfg.train_data_path)
    val_path = _resolve(cfg.val_data_path)
    output_dir = _resolve(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_str).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_str)

    train_data = _read_jsonl(train_path)
    val_data = _read_jsonl(val_path)
    print(f"train pairs: {len(train_data)} | val pairs: {len(val_data)}")

    ds_kwargs = {"tokenizer": tokenizer, "max_length": cfg.max_length}
    train_dataset = TranslationDataset(
        data=train_data,
        src_lang=cfg.src_lang,
        tgt_lang=cfg.tgt_lang,
        src_lang_token=cfg.src_lang_token,
        tgt_lang_token=cfg.tgt_lang_token,
        bidirectional=cfg.bidirectional,
        **ds_kwargs,
    )
    val_src2tgt = TranslationDataset(
        data=val_data,
        src_lang=cfg.src_lang,
        tgt_lang=cfg.tgt_lang,
        src_lang_token=cfg.src_lang_token,
        tgt_lang_token=cfg.tgt_lang_token,
        bidirectional=False,
        **ds_kwargs,
    )
    val_tgt2src = TranslationDataset(
        data=val_data,
        src_lang=cfg.tgt_lang,
        tgt_lang=cfg.src_lang,
        src_lang_token=cfg.tgt_lang_token,
        tgt_lang_token=cfg.src_lang_token,
        bidirectional=False,
        **ds_kwargs,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding="longest",
        max_length=cfg.max_length, pad_to_multiple_of=8,
    )
    dl_kwargs = {"batch_size": cfg.batch_size, "pin_memory": device == "cuda", "collate_fn": collator}
    train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    val_loader_s2t = DataLoader(val_src2tgt, shuffle=False, **dl_kwargs)
    val_loader_t2s = DataLoader(val_tgt2src, shuffle=False, **dl_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, eps=cfg.eps, weight_decay=cfg.weight_decay)
    use_amp = device == "cuda" and cfg.use_float16
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history = {
        "config": asdict(cfg),
        "steps": [],
        "train_loss": [],
        "src2tgt": {"loss": [], "spbleu": [], "chrf": [], "chrfpp": []},
        "tgt2src": {"loss": [], "spbleu": [], "chrf": [], "chrfpp": []},
        "avg": {"loss": [], "spbleu": [], "chrf": [], "chrfpp": []},
    }
    previous_best = -1.0
    step = 0

    model.train()
    for epoch in range(cfg.epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}"):
            optimizer.zero_grad()
            inputs = batch.to(device)
            ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
            with ctx:
                outputs = model(**inputs, use_cache=False)
                loss = outputs.loss

            if step % cfg.log_every == 0:
                print(f"step {step:5d} | loss {loss.item():.4f}")

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if (step + 1) % cfg.eval_every == 0:
                s2t = evaluate_split(model, tokenizer, val_loader_s2t, cfg.tgt_lang_token, device, cfg.max_length)
                t2s = evaluate_split(model, tokenizer, val_loader_t2s, cfg.src_lang_token, device, cfg.max_length)

                avg_loss = (s2t["eval_loss"] + t2s["eval_loss"]) / 2
                avg_spbleu = (s2t["spbleu"] + t2s["spbleu"]) / 2
                avg_chrf = (s2t["chrf"] + t2s["chrf"]) / 2
                avg_chrfpp = (s2t["chrfpp"] + t2s["chrfpp"]) / 2

                history["steps"].append(step + 1)
                history["train_loss"].append(loss.item())
                for key, src in (("src2tgt", s2t), ("tgt2src", t2s)):
                    history[key]["loss"].append(src["eval_loss"])
                    history[key]["spbleu"].append(src["spbleu"])
                    history[key]["chrf"].append(src["chrf"])
                    history[key]["chrfpp"].append(src["chrfpp"])
                history["avg"]["loss"].append(avg_loss)
                history["avg"]["spbleu"].append(avg_spbleu)
                history["avg"]["chrf"].append(avg_chrf)
                history["avg"]["chrfpp"].append(avg_chrfpp)

                print(
                    f"  eval @ step {step + 1}: "
                    f"avg_loss={avg_loss:.3f} | avg_spBLEU={avg_spbleu:.2f} | "
                    f"avg_chrF={avg_chrf:.2f} | avg_chrF++={avg_chrfpp:.2f}"
                )

                (output_dir / "metrics.json").write_text(
                    json.dumps({k: v for k, v in history.items() if k != "config"} | {"config": history["config"]},
                               indent=2, ensure_ascii=False, default=_json_default)
                )

                if cfg.save_model_on_evaluation and avg_spbleu >= previous_best:
                    previous_best = avg_spbleu
                    save_path = output_dir / f"best_nllb_spbleu={avg_spbleu:.2f}"
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

            step += 1

    final_path = output_dir / "final_nllb"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2, ensure_ascii=False, default=_json_default))
    print(f"Modelo final guardado en {final_path}")
    return history


if __name__ == "__main__":
    train(TrainConfig())
