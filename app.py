"""Demo interactiva de traducción bribri <-> español (Streamlit).

Carga el mejor modelo de la fase de modelado —NLLB-200-distilled-600M
reentrenado en H100 (Avance 4, 8 épocas, lr 2e-4)— y permite traducir texto
en ambas direcciones desde el navegador.

El modelo fine-tuneado pesa ~2.4 GB y NO está versionado en el repo
(`outputs/` y `*.safetensors` están en `.gitignore`). La app busca el
checkpoint en `--model-path` (por defecto `outputs_nllb_h100/final_nllb`) y,
si no lo encuentra, cae al modelo base `facebook/nllb-200-distilled-600M`
para que la interfaz sea ejecutable aun sin los pesos entrenados (avisando
que las traducciones serán las del modelo sin fine-tuning).

Tokens de idioma (idénticos al entrenamiento, ver nllb_train.py):
    español -> spa_Latn
    bribri  -> quy_Latn   (proxy Quechua Ayacucho; bribri no tiene token propio)

Uso:
    pip install -r requirements.txt
    streamlit run app.py
    # con un checkpoint específico:
    MODEL_PATH=outputs_nllb_h100/final_nllb streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL = "facebook/nllb-200-distilled-600M"
DEFAULT_CKPT = os.environ.get("MODEL_PATH", "outputs_nllb_h100/final_nllb")

# español <-> bribri  (bribri usa el proxy quechua quy_Latn)
LANG_TOKEN = {"es": "spa_Latn", "bri": "quy_Latn"}

# Ejemplos tomados del corpus real (data/splits), no inventados.
EJEMPLOS = {
    "es → bri": [
        "¿Cómo amaneció usted?",
        "Muy bien, ¿y usted?",
        "vieron y creyeron.",
    ],
    "bri → es": [
        "Ìs be' shkẽ̀na?",
        "Ye' na bua'ë, ìs be'?",
        "bua', ta̱ i-bikéíts irir.",
    ],
}


@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    """Carga modelo + tokenizer una sola vez. Devuelve (model, tok, fuente)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(model_path)
    if path.exists() and any(path.iterdir()):
        source = f"fine-tuned: {model_path}"
        load_from = model_path
    else:
        source = f"base (sin fine-tuning): {BASE_MODEL}"
        load_from = BASE_MODEL
    model = AutoModelForSeq2SeqLM.from_pretrained(load_from).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    return model, tokenizer, source, device


@torch.no_grad()
def translate(text, src, tgt, model, tokenizer, device, num_beams, max_new_tokens, length_penalty):
    tokenizer.src_lang = LANG_TOKEN[src]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    out = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_TOKEN[tgt]),
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        length_penalty=length_penalty,
        no_repeat_ngram_size=3,  # mitiga los bucles de repetición del bajo recurso
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Voces · Traductor Bribri-Español", page_icon="🌎", layout="centered")

st.title("🌎 Voces — Traductor Bribri ↔ Español")
st.caption(
    "Sistema NMT del Programa Voces (CENIA · Tec de Monterrey). "
    "Modelo: NLLB-200-distilled-600M fine-tuneado en H100 (chrF 31.43 · spBLEU 21.16)."
)

with st.sidebar:
    st.header("Configuración")
    model_path = st.text_input("Ruta del checkpoint", DEFAULT_CKPT,
                               help="Carpeta del modelo fine-tuneado. Si no existe, se usa el modelo base.")
    num_beams = st.slider("num_beams", 1, 8, 4, help="Más beams = búsqueda más amplia, más lento.")
    max_new_tokens = st.slider("max_new_tokens", 16, 256, 128)
    length_penalty = st.slider("length_penalty", 0.5, 2.0, 1.0, 0.1)
    st.divider()
    st.markdown(
        "**Par lingüístico:** bribri ↔ español  \n"
        "**Proxy de idioma:** `quy_Latn` (Quechua) para bribri  \n"
        "**Métrica primaria:** chrF / chrF++"
    )

model, tokenizer, source, device = load_model(model_path)

if source.startswith("base"):
    st.warning(
        f"⚠️ No se encontró el checkpoint fine-tuneado en `{model_path}`. "
        f"Usando el modelo base sin fine-tuning — las traducciones de bribri serán de baja calidad. "
        f"Coloque los pesos entrenados (carpeta `final_nllb/`) en esa ruta para resultados reales."
    )
else:
    st.success(f"Modelo cargado · {source} · device={device}")

direccion = st.radio("Dirección de traducción", ["es → bri", "bri → es"], horizontal=True)
src, tgt = ("es", "bri") if direccion == "es → bri" else ("bri", "es")

st.markdown("**Ejemplos rápidos:**")
cols = st.columns(len(EJEMPLOS[direccion]))
if "texto" not in st.session_state:
    st.session_state.texto = EJEMPLOS[direccion][0]
for col, ej in zip(cols, EJEMPLOS[direccion]):
    if col.button(ej, use_container_width=True):
        st.session_state.texto = ej

texto = st.text_area(
    f"Texto en {'español' if src == 'es' else 'bribri'}",
    key="texto",
    height=120,
)

if st.button("Traducir", type="primary", use_container_width=True):
    if texto.strip():
        with st.spinner("Traduciendo…"):
            resultado = translate(texto, src, tgt, model, tokenizer, device,
                                  num_beams, max_new_tokens, length_penalty)
        st.markdown(f"**Traducción ({'bribri' if tgt == 'bri' else 'español'}):**")
        st.success(resultado)
    else:
        st.info("Escriba algo de texto para traducir.")

st.divider()
st.caption(
    "Nota: el bribri no tiene token propio en NLLB-200; se usa el proxy `quy_Latn`. "
    "El modelo se entrenó con ~1.2k pares paralelos, así que las traducciones son un "
    "indicador de progreso de investigación, no un sistema de producción."
)
