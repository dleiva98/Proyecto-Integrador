"""Gráficas para la corrida NLLB-H100 + comparación de las 3 corridas.

Lee:
  outputs/test_metrics.json            (NLLB original, 3ep lr5e-4)
  outputs_nllb_h100/metrics.json       (NLLB H100, 8ep lr2e-4 - curvas)
  outputs_nllb_h100/test_metrics.json  (NLLB H100 - test)
  outputs_m2m100/test_metrics.json     (M2M-100)

Produce en outputs_nllb_h100/:
  1. training_curves_nllb_h100.png  - curvas de la corrida H100
  2. comparison_bars_3way.png       - barras de las 3 corridas
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H100 = Path("outputs_nllb_h100")
h = json.loads((H100 / "metrics.json").read_text())
h_test = json.loads((H100 / "test_metrics.json").read_text())
nllb_orig = json.loads((Path("outputs") / "test_metrics.json").read_text())
m2m = json.loads((Path("outputs_m2m100") / "test_metrics.json").read_text())

# 1. Curvas H100
steps = h["steps"]
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for metric, ax, title in (("loss", axes[0], "Val loss"),
                          ("spbleu", axes[1], "spBLEU"),
                          ("chrfpp", axes[2], "chrF++")):
    ax.plot(steps, h["src2tgt"][metric], label="es→bri")
    ax.plot(steps, h["tgt2src"][metric], label="bri→es")
    ax.plot(steps, h["avg"][metric], label="avg", linestyle="--")
    ax.set_title(f"NLLB-H100 · {title}"); ax.set_xlabel("step"); ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(H100 / "training_curves_nllb_h100.png", dpi=120)
plt.close()
print("OK training_curves_nllb_h100.png")

# 2. Barras 3-way
import numpy as np
runs = [("NLLB orig\n3ep lr5e-4", nllb_orig["avg"], "#2563eb"),
        ("NLLB H100\n8ep lr2e-4", h_test["avg"], "#16a34a"),
        ("M2M-100\n3ep lr5e-4", m2m["avg"], "#f59e0b")]
metrics = [("eval_loss", "Val loss (menor=mejor)"),
           ("chrf", "chrF (mayor=mejor)"),
           ("spbleu", "spBLEU (mayor=mejor)")]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for (key, label), ax in zip(metrics, axes):
    names = [r[0] for r in runs]
    vals = [r[1][key] for r in runs]
    colors = [r[2] for r in runs]
    bars = ax.bar(names, vals, color=colors)
    ax.set_title(label); ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}",
                ha="center", va="bottom", fontweight="bold", fontsize=9)
plt.suptitle("Comparación de las 3 corridas · TEST final", y=1.03)
plt.tight_layout()
plt.savefig(H100 / "comparison_bars_3way.png", dpi=120, bbox_inches="tight")
plt.close()
print("OK comparison_bars_3way.png")

# Tabla
print("\n=== 3 CORRIDAS · TEST (avg) ===")
print(f"{'métrica':<11}{'NLLB orig':>11}{'NLLB H100':>11}{'M2M-100':>11}")
for key in ("eval_loss", "spbleu", "chrf", "chrfpp"):
    print(f"{key:<11}{nllb_orig['avg'][key]:>11.3f}{h_test['avg'][key]:>11.3f}{m2m['avg'][key]:>11.3f}")

improved = h_test["avg"]["spbleu"] > nllb_orig["avg"]["spbleu"]
print(f"\n¿NLLB-H100 mejoró el spBLEU? {'SÍ' if improved else 'NO'} "
      f"({nllb_orig['avg']['spbleu']:.2f} -> {h_test['avg']['spbleu']:.2f})")
