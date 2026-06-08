| Modelo | Tipo | Hardware | Tiempo | spBLEU | chrF | chrF++ |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Oracle de candidatos NLLB+M2M (cota superior no desplegable) | oracle / upper bound | H100 + T4 | usa referencia; no desplegable | 22.15 | 34.30 | 33.30 |
| Blend homogéneo NLLB anti-degeneración | ensamble homogéneo | H100 + T4 | post-proceso; sin reentrenar | 21.25 | 31.61 | 30.66 |
| NLLB H100 · 600M · 8ep lr2e-4 | individual | H100 | no registrado | 21.16 | 31.43 | 30.47 |
| Blend heterogéneo NLLB+M2M anti-degeneración | ensamble heterogéneo | H100 + T4 | post-proceso; sin reentrenar | 17.13 | 28.14 | 26.98 |
| NLLB orig · 600M · 3ep lr5e-4 | individual | Colab T4 | 30-45 min reportado | 14.43 | 27.16 | 26.09 |
| M2M-100 · 418M · 3ep lr5e-4 | individual | H100 | no registrado | 1.33 | 9.38 | 8.33 |
