"""Genera las gráficas comparativas NLLB-200 vs M2M-100.

Lee:
  outputs/metrics.json            (NLLB - curvas)
  outputs/test_metrics.json       (NLLB - test final)
  outputs_m2m100/metrics.json     (M2M-100 - curvas)
  outputs_m2m100/test_metrics.json(M2M-100 - test final)

Produce en outputs_m2m100/:
  1. training_curves_m2m100.png  - curvas de M2M-100 solo (espejo de NLLB)
  2. comparison_curves.png       - NLLB vs M2M-100 superpuestos (loss, spBLEU, chrF++)
  3. comparison_bars.png         - barras de test final (val_loss, chrF, spBLEU)
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NLLB_DIR = Path("outputs")
M2M_DIR = Path("outputs_m2m100")
M2M_DIR.mkdir(exist_ok=True)

nllb = json.loads((NLLB_DIR / "metrics.json").read_text())
m2m = json.loads((M2M_DIR / "metrics.json").read_text())
nllb_test = json.loads((NLLB_DIR / "test_metrics.json").read_text())
m2m_test = json.loads((M2M_DIR / "test_metrics.json").read_text())

# ----------------------------------------------------------------------- #
# 1. Curvas de M2M-100 solo (espejo exacto de la gráfica de NLLB)
# ----------------------------------------------------------------------- #
steps = m2m["steps"]
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].plot(steps, m2m["src2tgt"]["loss"], label="es→bri")
axes[0].plot(steps, m2m["tgt2src"]["loss"], label="bri→es")
axes[0].plot(steps, m2m["avg"]["loss"], label="avg", linestyle="--")
axes[0].set_title("M2M-100 · Val loss"); axes[0].set_xlabel("step"); axes[0].legend(); axes[0].grid(True)

axes[1].plot(steps, m2m["src2tgt"]["spbleu"], label="es→bri")
axes[1].plot(steps, m2m["tgt2src"]["spbleu"], label="bri→es")
axes[1].plot(steps, m2m["avg"]["spbleu"], label="avg", linestyle="--")
axes[1].set_title("M2M-100 · spBLEU"); axes[1].set_xlabel("step"); axes[1].legend(); axes[1].grid(True)

axes[2].plot(steps, m2m["src2tgt"]["chrfpp"], label="es→bri")
axes[2].plot(steps, m2m["tgt2src"]["chrfpp"], label="bri→es")
axes[2].plot(steps, m2m["avg"]["chrfpp"], label="avg", linestyle="--")
axes[2].set_title("M2M-100 · chrF++"); axes[2].set_xlabel("step"); axes[2].legend(); axes[2].grid(True)
plt.tight_layout()
plt.savefig(M2M_DIR / "training_curves_m2m100.png", dpi=120)
plt.close()
print("OK training_curves_m2m100.png")

# ----------------------------------------------------------------------- #
# 2. Curvas comparativas (avg de cada modelo superpuesto)
#    NOTA: si los steps difieren, se grafica cada uno con su propio eje x.
# ----------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for metric, ax, title in (("loss", axes[0], "Val loss (avg)"),
                          ("spbleu", axes[1], "spBLEU (avg)"),
                          ("chrfpp", axes[2], "chrF++ (avg)")):
    ax.plot(nllb["steps"], nllb["avg"][metric], label="NLLB-200", marker="o")
    ax.plot(m2m["steps"], m2m["avg"][metric], label="M2M-100", marker="s")
    ax.set_title(title); ax.set_xlabel("step"); ax.legend(); ax.grid(True)
plt.suptitle("NLLB-200 vs M2M-100 · curvas de validación", y=1.02)
plt.tight_layout()
plt.savefig(M2M_DIR / "comparison_curves.png", dpi=120, bbox_inches="tight")
plt.close()
print("OK comparison_curves.png")

# ----------------------------------------------------------------------- #
# 3. Barras comparativas de test final (val_loss, chrF, spBLEU)
# ----------------------------------------------------------------------- #
import numpy as np
metrics = [("eval_loss", "Val loss\n(menor=mejor)"),
           ("chrf", "chrF\n(mayor=mejor)"),
           ("spbleu", "spBLEU\n(mayor=mejor)")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for (key, label), ax in zip(metrics, axes):
    vals = [nllb_test["avg"][key], m2m_test["avg"][key]]
    bars = ax.bar(["NLLB-200", "M2M-100"], vals, color=["#2563eb", "#f59e0b"])
    ax.set_title(label)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}",
                ha="center", va="bottom", fontweight="bold")
plt.suptitle("NLLB-200 vs M2M-100 · métricas finales sobre TEST", y=1.02)
plt.tight_layout()
plt.savefig(M2M_DIR / "comparison_bars.png", dpi=120, bbox_inches="tight")
plt.close()
print("OK comparison_bars.png")

# ----------------------------------------------------------------------- #
# Tabla resumen en texto (útil para el informe)
# ----------------------------------------------------------------------- #
print("\n=== RESUMEN TEST (avg ambas direcciones) ===")
print(f"{'métrica':<12} {'NLLB-200':>12} {'M2M-100':>12} {'Δ (NLLB-M2M)':>14}")
for key in ("eval_loss", "spbleu", "chrf", "chrfpp"):
    n, m = nllb_test["avg"][key], m2m_test["avg"][key]
    print(f"{key:<12} {n:>12.3f} {m:>12.3f} {n-m:>14.3f}")

summary = {"nllb_test": nllb_test["avg"], "m2m100_test": m2m_test["avg"]}
(M2M_DIR / "comparison_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False))
print("\nOK comparison_summary.json")
print("\nTodo en outputs_m2m100/")
