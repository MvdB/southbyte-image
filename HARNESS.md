# southbyte-image — Bild-Modell-Harness

Familien-agnostischer Betrieb + Evaluation von Text-zu-Bild-Modellen auf dem DGX Spark.
Ein Modell hinzufügen = **Profil/Config-Eintrag + Loader-Namen** — kein Code.

## Schichten

| # | Teil | Datei |
|---|------|-------|
| ① | Serving-Image (alle Deps) | `serving/Dockerfile.image` (v1) · `serving/Dockerfile.image.v2` (+ flash-attn/mage_flow/nunchaku) · `serving/Dockerfile.image.v3` (Qwen-Image-2.1) |
| ② | Loader-Registry | `serving/loaders.py` |
| ③ | Serving-Adapter (OpenAI-Images-API) | `serving/server_image.py` · Start: `serving/run_image.sh` |
| ③ | Profile (pro Modell) | `southbyte-spark-profiles/image/<dir>/image_profile.conf` |
| ④ | Orchestrator (Feldlauf + Scoring) | `eval/orchestrate_images.py` · Registry: `config/image_models.yaml` |
| ④ | Metriken | `eval/metrics/adherence.py` (Treue) · `eval/metrics/ocr_text.py` (Text-CER, containment) |
| ④ | Vergleichsseite | `eval/make_docs.py` |

## Warum drei Images

v1 und v2 tragen alles bis FLUX.2. **v3 (2026-09-22) gibt es nur wegen
Qwen-Image-2.1:** `QwenImage21Pipeline` steckt in keinem diffusers-Release —
auch 0.40.0 nicht —, sondern erst im Hauptzweig seit dem 18.09. (PR #14804);
das Dockerfile pinnt deshalb den Commit. Dazu verlangt die Modellkarte
transformers >=5.17, und das schliesst sich mit v2 aus: dort gilt `<5.6`, weil
5.6 den `input_embeds`-kwarg von `create_causal_mask` entfernt hat und
mage_flow daran crasht. Mage-Flow bleibt auf v2, die Registry waehlt das Image
je Modell (`image:`).

**Das v1-Image liegt seit 22.09. nicht mehr lokal vor.** Die vier Modelle, die
darauf zeigen, sind gemessen und veroeffentlicht; ein erneuter Lauf mit ihnen
braucht vorher einen Neubau.

## Loader-Strategien (`PROFILE_LOADER` → `IMG_LOADER`)

| Loader | Familie | Deps |
|--------|---------|------|
| `auto` | FLUX.1, Qwen-Image (Standard-Diffusers) | v1 |
| `diffusion` | ERNIE u.a. Custom-Klassen **in** diffusers (`_class_name`) | v1 |
| `mage_flow` | Mage-Flow (ext. Lib `mage_flow`) | **v2** (flash-attn) |
| `flux2_singlefile` | FLUX.2-klein/-dev (NVFP4-Single-File + geteilte Komponenten) | **v2** (nunchaku/torchao) |

Ohne `PROFILE_LOADER` rät `loaders._autodetect()` aus `model_index.json`.

**`auto` heisst `AutoPipelineForText2Image` — und dessen Mapping ist kleiner als
diffusers.** Qwen-Image-2.1 steht dort nicht (geprueft 22.09.: nur `qwenimage`
und `qwenimage-controlnet`), obwohl `QwenImage21Pipeline` existiert. Fuer solche
Faelle ist `diffusion` richtig: `DiffusionPipeline` liest `_class_name` aus
`model_index.json`. Im Zweifel vor dem Lauf nachsehen, sonst scheitert erst der
Modellstart.

## Neues Modell hinzufügen

1. Eintrag in `config/image_models.yaml` (`dir`, `loader`, `steps`, `guidance`, ggf. `components_dir`).
2. `active: true`, sobald die Serving-Deps im Image sind.
3. `python eval/orchestrate_images.py --models <Name>` — generiert + bewertet + baut die Seite.

## Guidance: bei den Qwen-Modellen wirkungslos

Der Server waehlt den Parameternamen anhand der Signatur: `guidance_scale`
(FLUX, ERNIE) oder `true_cfg_scale` (QwenImage, Qwen-Image-2.1). **Echte CFG
schaltet sich bei den Qwen-Pipelines aber nur mit Negativ-Prompt ein** — ohne
ihn meldet der Log `classifier-free guidance is not enabled` und der Wert aus
der Registry bleibt folgenlos. Im Testsatz hat **1 von 22** Faellen einen
Negativ-Prompt, die Qwen-Modelle laufen also praktisch guidance-frei, waehrend
FLUX und ERNIE ihren Wert anwenden. Das ist kein Fehler eines einzelnen Laufs,
aber ein Unterschied, der beim Vergleich der Zahlen mitzudenken ist
(festgestellt 2026-09-22).

## Lizenzen der Modelle

`config/models.yaml` in southbyte-vllm fuehrt Release und Lizenz je Modell; die
Vergleichsseite baut daraus einen sichtbaren Hinweis fuer alle Modelle mit
nicht-kommerzieller Lizenz (`_NICHT_KOMMERZIELL` in `eval/make_docs.py`).
Betroffen sind aktuell **FLUX.2-dev** und **Qwen-Image-2.1** (Qwen Research
License, 2026-09-20: Nutzung nur fuer Forschung und Evaluation). Die
Einschraenkung gilt auch fuer die erzeugten Bilder — dass die Messbilder
trotzdem auf der Seite stehen, ist eine bewusste Entscheidung vom 22.09.2026.

Beim Nachtragen in `models.yaml`: **kein Kommentar hinter dem Wert.** Die
Parser in `eval/make_docs.py` und `southbyte-results/feeds.py` lesen die Zeile
mit einem regulaeren Ausdruck, der an `#` abbricht — die Lizenz erscheint dann
als „—" auf der Seite.

## Scoring

Ein Judge über alles: **`qwen/qwen3.7-plus`** via LiteLLM-Proxy (`JUDGE_ENDPOINT`/`JUDGE_MODEL`,
`VISION_API_KEY`). Treue = VLM-as-Judge (0..1); Textrendering = **Containment-CER**
(bestraft fehlenden/falschen Soll-Text, nicht zusätzlich korrekt gerenderten).

## Offen (v2-Build)

`docker build -f serving/Dockerfile.image.v2 -t spark-southbyte-image:v2 .` — flash-attn
und NVFP4-Backend auf sm_120 beim Bauen verifizieren; danach Mage-Flow / FLUX.2-klein
in der Config auf `active: true`.
