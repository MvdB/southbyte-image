# southbyte-image — Bild-Modell-Harness

Familien-agnostischer Betrieb + Evaluation von Text-zu-Bild-Modellen auf dem DGX Spark.
Ein Modell hinzufügen = **Profil/Config-Eintrag + Loader-Namen** — kein Code.

## Schichten

| # | Teil | Datei |
|---|------|-------|
| ① | Serving-Image (alle Deps) | `serving/Dockerfile.image` (v1) · `serving/Dockerfile.image.v2` (+ flash-attn/mage_flow/nunchaku) |
| ② | Loader-Registry | `serving/loaders.py` |
| ③ | Serving-Adapter (OpenAI-Images-API) | `serving/server_image.py` · Start: `serving/run_image.sh` |
| ③ | Profile (pro Modell) | `southbyte-spark-profiles/image/<dir>/image_profile.conf` |
| ④ | Orchestrator (Feldlauf + Scoring) | `eval/orchestrate_images.py` · Registry: `config/image_models.yaml` |
| ④ | Metriken | `eval/metrics/adherence.py` (Treue) · `eval/metrics/ocr_text.py` (Text-CER, containment) |
| ④ | Vergleichsseite | `eval/make_docs.py` |

## Loader-Strategien (`PROFILE_LOADER` → `IMG_LOADER`)

| Loader | Familie | Deps |
|--------|---------|------|
| `auto` | FLUX.1, Qwen-Image (Standard-Diffusers) | v1 |
| `diffusion` | ERNIE u.a. Custom-Klassen **in** diffusers (`_class_name`) | v1 |
| `mage_flow` | Mage-Flow (ext. Lib `mage_flow`) | **v2** (flash-attn) |
| `flux2_singlefile` | FLUX.2-klein/-dev (NVFP4-Single-File + geteilte Komponenten) | **v2** (nunchaku/torchao) |

Ohne `PROFILE_LOADER` rät `loaders._autodetect()` aus `model_index.json`.

## Neues Modell hinzufügen

1. Eintrag in `config/image_models.yaml` (`dir`, `loader`, `steps`, `guidance`, ggf. `components_dir`).
2. `active: true`, sobald die Serving-Deps im Image sind.
3. `python eval/orchestrate_images.py --models <Name>` — generiert + bewertet + baut die Seite.

## Scoring

Ein Judge über alles: **`qwen/qwen3.7-plus`** via LiteLLM-Proxy (`JUDGE_ENDPOINT`/`JUDGE_MODEL`,
`VISION_API_KEY`). Treue = VLM-as-Judge (0..1); Textrendering = **Containment-CER**
(bestraft fehlenden/falschen Soll-Text, nicht zusätzlich korrekt gerenderten).

## Offen (v2-Build)

`docker build -f serving/Dockerfile.image.v2 -t spark-southbyte-image:v2 .` — flash-attn
und NVFP4-Backend auf sm_120 beim Bauen verifizieren; danach Mage-Flow / FLUX.2-klein
in der Config auf `active: true`.
