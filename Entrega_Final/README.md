# Documento Final de Entrega — Proyecto Integrador (Equipo 66)

Informe académico consolidado de la Maestría en Inteligencia Artificial
Aplicada (Tecnológico de Monterrey). Integra las Entregas 1–2 y los Avances
3–6 del sistema NMT bribri–español del **Programa Voces (Componente B)** en un
único documento, anclado en la fase de Evaluación y Despliegue de CRISP-ML(Q).

## Contenido de la carpeta

| Archivo | Descripción |
|---|---|
| `Documento_Final_Entrega_Equipo66.docx` | **Entregable principal** (editable). Portada institucional, resumen ejecutivo/abstract, índice automático, 12 secciones, 13 tablas, figuras embebidas y referencias APA. |
| `build_documento_final.py` | Script reproducible que genera el `.docx` con `python-docx`. |
| `assets/logo_tec.png` | Membrete institucional de la portada. |
| `assets/make_logo.py` | Script que genera el membrete. |

## Notas

- **Asesor:** Carlos Villaseñor. **Patrocinador:** Carlos Aspillaga (CENIA).
- El documento tiene activado *update fields on load*: al abrirlo en Microsoft
  Word, el **índice** y los **números de página** se completan automáticamente
  (o con `Ctrl+E` ▸ F9 / clic derecho ▸ *Actualizar campos*).
- **Exportar a PDF:** abrir en Word ▸ *Archivo ▸ Guardar como / Exportar ▸ PDF*.
- Reproducir el `.docx`:
  ```bash
  pip install python-docx Pillow
  python Entrega_Final/assets/make_logo.py
  python Entrega_Final/build_documento_final.py
  ```
