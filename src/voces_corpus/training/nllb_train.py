"""Fine-tuning de NLLB-200 sobre el corpus bribri<->español.

Adaptado del template oficial del Proyecto Integrador (a9c6fdf2-nllb_training.py).

Cambios respecto al template:
- Par `bri`/`es` con `quy_Latn` (Quechua Ayacucho) como proxy para bribri y
  `spa_Latn` para español. La elección de proxy es discutible; ver README,
  sección "Avance 2".
- Carga splits desde `data/splits/{train,val,test}.jsonl`.
- Loguea train_losses, val_losses (src2tgt, tgt2src, avg), spBLEU, chrF y
  chrF++ a `outputs/metrics.json` cada `EVAL_EVERY` pasos.
- Guarda checkpoints reanudables con estado de modelo, tokenizador,
  optimizador, scaler AMP, historial y posición del entrenamiento.
- `tqdm` no-notebook por defecto.
- Pensado para ejecutarse en Colab con GPU; ver `notebooks/nllb_finetune_colab.ipynb`.
"""
from __future__ import annotations

import json
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
    best_metric: str = "chrfpp"
    checkpoint_every: int = 100
    resume_from_checkpoint: Optional[Path] = None
    resume_if_checkpoint_exists: bool = False
    gradient_accumulation_steps: int = 1

    use_float16: bool = True
    use_bfloat16: bool = False
    seed: int = 42
    optimizer_name: str = "adamw_torch"  # adamw_torch | adamw_bnb_8bit
    required_gpu_name: Optional[str] = None
    min_cuda_memory_gb: Optional[float] = None
    min_system_memory_gb: Optional[float] = None
    load_in_4bit: bool = False
    lora_r: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

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


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _make_optimizer(model, cfg: TrainConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No hay parámetros entrenables. Revisa la configuración LoRA/full fine-tuning.")
    if cfg.optimizer_name == "adamw_torch":
        return torch.optim.AdamW(params, lr=cfg.lr, eps=cfg.eps, weight_decay=cfg.weight_decay)
    if cfg.optimizer_name == "adamw_bnb_8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "optimizer_name='adamw_bnb_8bit' requiere instalar bitsandbytes. "
                "En Colab: %pip install -q bitsandbytes"
            ) from exc
        return bnb.optim.AdamW8bit(params, lr=cfg.lr, eps=cfg.eps, weight_decay=cfg.weight_decay)
    raise ValueError(f"optimizer_name no soportado: {cfg.optimizer_name}")


def _validate_hardware_requirements(cfg: TrainConfig, device: str) -> None:
    if cfg.required_gpu_name is None and cfg.min_cuda_memory_gb is None and cfg.min_system_memory_gb is None:
        return
    if cfg.min_system_memory_gb is not None:
        try:
            import psutil
        except ImportError as exc:
            raise ImportError("min_system_memory_gb requiere psutil. En Colab: %pip install -q psutil") from exc
        system_gb = psutil.virtual_memory().total / 1024**3
        print(f"RAM del sistema: {system_gb:.1f} GiB")
        if system_gb < cfg.min_system_memory_gb:
            raise RuntimeError(
                f"RAM insuficiente: se requieren al menos {cfg.min_system_memory_gb:.1f} GiB, "
                f"pero el runtime actual tiene {system_gb:.1f} GiB."
            )

    if device != "cuda":
        raise RuntimeError(
            "Esta corrida requiere GPU CUDA, pero el runtime actual no tiene CUDA disponible."
        )

    gpu_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU detectada: {gpu_name} | VRAM total: {total_gb:.1f} GiB")

    if cfg.required_gpu_name and cfg.required_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(
            f"GPU insuficiente: se requiere una GPU que contenga '{cfg.required_gpu_name}' "
            f"en el nombre, pero Colab asignó '{gpu_name}'. "
            "Cambia el runtime de Colab a A100 antes de ejecutar esta celda."
        )

    if cfg.min_cuda_memory_gb is not None and total_gb < cfg.min_cuda_memory_gb:
        raise RuntimeError(
            f"VRAM insuficiente: se requieren al menos {cfg.min_cuda_memory_gb:.1f} GiB, "
            f"pero la GPU actual tiene {total_gb:.1f} GiB. "
            "No se continuará para evitar OOM."
        )


def _torch_dtype_for_model(cfg: TrainConfig, device: str):
    if device != "cuda":
        return None
    if cfg.use_bfloat16:
        return torch.bfloat16
    if cfg.use_float16:
        return torch.float16
    return None


def _load_seq2seq_model(model_path: str | Path, cfg: TrainConfig, device: str):
    if cfg.load_in_4bit:
        if device != "cuda":
            raise RuntimeError("load_in_4bit requiere GPU CUDA.")
        compute_dtype = torch.bfloat16 if cfg.use_bfloat16 else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        return AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
        )

    kwargs = {}
    torch_dtype = _torch_dtype_for_model(cfg, device)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs).to(device)


def _enable_input_grads(model) -> None:
    """Necesario para LoRA + gradient checkpointing en modelos cuantizados."""
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    input_embeddings = model.get_input_embeddings()

    def make_inputs_require_grad(_module, _inputs, output):
        output.requires_grad_(True)

    input_embeddings.register_forward_hook(make_inputs_require_grad)


def _attach_lora_if_needed(model, cfg: TrainConfig, adapter_dir: Optional[Path] = None):
    if cfg.lora_r <= 0:
        return model

    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("LoRA/QLoRA requiere peft. En Colab: %pip install -q peft") from exc

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        _enable_input_grads(model)

    if adapter_dir is not None and (adapter_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
        _enable_input_grads(model)
        print(f"Adaptadores LoRA restaurados desde: {adapter_dir}")
        return model

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=cfg.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    _enable_input_grads(model)
    model.print_trainable_parameters()
    return model


def load_trained_model(model_dir: Path, cfg: TrainConfig, device: str):
    """Carga un checkpoint final/best para evaluación, incluyendo adaptadores LoRA."""
    model_dir = Path(model_dir)
    if cfg.lora_r > 0 and (model_dir / "adapter_config.json").exists():
        model = _load_seq2seq_model(cfg.model_str, cfg, device)
        model = _attach_lora_if_needed(model, cfg, model_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir if (model_dir / "tokenizer_config.json").exists() else cfg.tokenizer_str
        )
        return model, tokenizer

    model = _load_seq2seq_model(model_dir, cfg, device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def _checkpoint_state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "training_state.pt"


def _checkpoint_meta_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "trainer_state.json"


def _save_metrics(output_dir: Path, history: dict[str, Any]) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False, default=_json_default)
    )


def _save_training_checkpoint(
    checkpoint_dir: Path,
    model,
    tokenizer,
    optimizer,
    scaler,
    cfg: TrainConfig,
    history: dict[str, Any],
    *,
    next_epoch: int,
    next_batch_idx: int,
    step: int,
    best_score: float,
    started_at: float,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    state = {
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "history": history,
        "next_epoch": next_epoch,
        "next_batch_idx": next_batch_idx,
        "step": step,
        "best_score": best_score,
        "started_at": started_at,
        "elapsed_seconds": time.time() - started_at,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, _checkpoint_state_path(checkpoint_dir))
    meta = {
        "next_epoch": next_epoch,
        "next_batch_idx": next_batch_idx,
        "step": step,
        "best_metric": cfg.best_metric,
        "best_score": best_score,
        "elapsed_seconds": state["elapsed_seconds"],
        "model_str": cfg.model_str,
        "tokenizer_str": cfg.tokenizer_str,
    }
    _checkpoint_meta_path(checkpoint_dir).write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def _load_training_state(checkpoint_dir: Path, device: str) -> dict[str, Any]:
    return torch.load(_checkpoint_state_path(checkpoint_dir), map_location=device)


def _resolve_resume_checkpoint(cfg: TrainConfig, output_dir: Path, repo_root: Path) -> Optional[Path]:
    if cfg.resume_from_checkpoint is not None:
        p = cfg.resume_from_checkpoint
        return p if p.is_absolute() else (repo_root / p)
    candidate = output_dir / "checkpoint-last"
    if cfg.resume_if_checkpoint_exists and _checkpoint_state_path(candidate).exists():
        return candidate
    return None


def train(cfg: TrainConfig, repo_root: Optional[Path] = None) -> dict:
    repo_root = Path(repo_root) if repo_root else Path.cwd()

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else (repo_root / p)

    train_path = _resolve(cfg.train_data_path)
    val_path = _resolve(cfg.val_data_path)
    output_dir = _resolve(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps debe ser >= 1")

    _set_reproducible_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    _validate_hardware_requirements(cfg, device)

    resume_checkpoint = _resolve_resume_checkpoint(cfg, output_dir, repo_root)
    if resume_checkpoint is not None:
        print(f"Reanudando desde checkpoint: {resume_checkpoint}")
        if cfg.lora_r > 0:
            model = _load_seq2seq_model(cfg.model_str, cfg, device)
            tokenizer = AutoTokenizer.from_pretrained(
                resume_checkpoint if (resume_checkpoint / "tokenizer_config.json").exists() else cfg.tokenizer_str
            )
            model = _attach_lora_if_needed(model, cfg, resume_checkpoint)
        else:
            model = _load_seq2seq_model(resume_checkpoint, cfg, device)
            tokenizer = AutoTokenizer.from_pretrained(resume_checkpoint)
    else:
        model = _load_seq2seq_model(cfg.model_str, cfg, device)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_str)
        model = _attach_lora_if_needed(model, cfg)

    peft_kbit_training = cfg.load_in_4bit and cfg.lora_r > 0
    if hasattr(model, "gradient_checkpointing_enable") and not peft_kbit_training:
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

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
    val_loader_s2t = DataLoader(val_src2tgt, shuffle=False, **dl_kwargs)
    val_loader_t2s = DataLoader(val_tgt2src, shuffle=False, **dl_kwargs)

    def make_train_loader(epoch: int) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(cfg.seed + epoch)
        return DataLoader(train_dataset, shuffle=True, generator=generator, **dl_kwargs)

    optimizer = _make_optimizer(model, cfg)
    use_fp16 = device == "cuda" and cfg.use_float16 and not cfg.use_bfloat16
    use_bf16 = device == "cuda" and cfg.use_bfloat16
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    history = {
        "config": asdict(cfg),
        "steps": [],
        "train_loss": [],
        "src2tgt": {"loss": [], "spbleu": [], "chrf": [], "chrfpp": []},
        "tgt2src": {"loss": [], "spbleu": [], "chrf": [], "chrfpp": []},
        "avg": {"loss": [], "spbleu": [], "chrf": [], "chrfpp": []},
    }
    best_score = float("-inf")
    step = 0
    start_epoch = 0
    start_batch_idx = 0
    started_at = time.time()

    if resume_checkpoint is not None:
        state = _load_training_state(resume_checkpoint, device)
        optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        history = state.get("history", history)
        step = int(state.get("step", 0))
        best_score = float(state.get("best_score", float("-inf")))
        start_epoch = int(state.get("next_epoch", 0))
        start_batch_idx = int(state.get("next_batch_idx", 0))
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        started_at = time.time() - float(state.get("elapsed_seconds", 0.0))
        print(f"Estado restaurado: epoch={start_epoch}, batch={start_batch_idx}, step={step}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, cfg.epochs):
        train_loader = make_train_loader(epoch)
        total_batches = len(train_loader)
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")):
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue

            inputs = batch.to(device)
            if use_fp16:
                ctx = torch.amp.autocast("cuda", dtype=torch.float16)
            elif use_bf16:
                ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
            else:
                ctx = nullcontext()
            with ctx:
                outputs = model(**inputs, use_cache=False)
                raw_loss = outputs.loss
                loss = raw_loss / cfg.gradient_accumulation_steps

            if step % cfg.log_every == 0:
                print(f"step {step:5d} | loss {raw_loss.item():.4f}")

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (
                (batch_idx + 1) % cfg.gradient_accumulation_steps == 0
                or batch_idx + 1 == total_batches
            )
            if not should_step:
                continue

            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            should_evaluate = cfg.eval_every > 0 and step % cfg.eval_every == 0
            should_checkpoint = cfg.checkpoint_every > 0 and step % cfg.checkpoint_every == 0

            if should_evaluate:
                s2t = evaluate_split(model, tokenizer, val_loader_s2t, cfg.tgt_lang_token, device, cfg.max_length)
                t2s = evaluate_split(model, tokenizer, val_loader_t2s, cfg.src_lang_token, device, cfg.max_length)

                avg_loss = (s2t["eval_loss"] + t2s["eval_loss"]) / 2
                avg_spbleu = (s2t["spbleu"] + t2s["spbleu"]) / 2
                avg_chrf = (s2t["chrf"] + t2s["chrf"]) / 2
                avg_chrfpp = (s2t["chrfpp"] + t2s["chrfpp"]) / 2

                history["steps"].append(step)
                history["train_loss"].append(raw_loss.item())
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
                    f"  eval @ step {step}: "
                    f"avg_loss={avg_loss:.3f} | avg_spBLEU={avg_spbleu:.2f} | "
                    f"avg_chrF={avg_chrf:.2f} | avg_chrF++={avg_chrfpp:.2f}"
                )

                history["wall_time_seconds"] = time.time() - started_at
                _save_metrics(output_dir, history)

                metric_values = {
                    "loss": -avg_loss,
                    "spbleu": avg_spbleu,
                    "chrf": avg_chrf,
                    "chrfpp": avg_chrfpp,
                }
                if cfg.best_metric not in metric_values:
                    raise ValueError(f"best_metric no soportada: {cfg.best_metric}")
                current_score = metric_values[cfg.best_metric]
                if cfg.save_model_on_evaluation and current_score >= best_score:
                    best_score = current_score
                    printable_score = -current_score if cfg.best_metric == "loss" else current_score
                    save_path = output_dir / f"best_nllb_{cfg.best_metric}={printable_score:.2f}"
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

            if should_checkpoint or should_evaluate:
                next_epoch = epoch + 1 if batch_idx + 1 == total_batches else epoch
                next_batch_idx = 0 if batch_idx + 1 == total_batches else batch_idx + 1
                _save_training_checkpoint(
                    output_dir / "checkpoint-last",
                    model,
                    tokenizer,
                    optimizer,
                    scaler,
                    cfg,
                    history,
                    next_epoch=next_epoch,
                    next_batch_idx=next_batch_idx,
                    step=step,
                    best_score=best_score,
                    started_at=started_at,
                )

    final_path = output_dir / "final_nllb"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    history["wall_time_seconds"] = time.time() - started_at
    _save_metrics(output_dir, history)
    print(f"Modelo final guardado en {final_path}")
    return history


if __name__ == "__main__":
    train(TrainConfig())
