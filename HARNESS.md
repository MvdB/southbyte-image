# southbyte-image — Bild-Modell-Harness

Familien-agnostischer Betrieb + Evaluation von Text-zu-Bild-Modellen auf dem DGX Spark.
Ein Modell hinzufügen = **Profil/Config-Eintrag + Loader-Namen** — kein Code.

## Schichten

| # | Teil | Datei |
|---|------|-------|
| ① | Serving-Image (alle Deps) | `serving/Dockerfile.image` (v1) · `.v2` (+ mage_flow/nunchaku/bnb) · `.v2.1` (= v2 + aktueller Servercode) · `.v3` (Qwen-Image-2.1) |
| ② | Loader-Registry | `serving/loaders.py` |
| ③ | Serving-Adapter (OpenAI-Images-API) | `serving/server_image.py` · Start: `serving/run_image.sh` |
| ③ | Profile (pro Modell) | `southbyte-spark-profiles/image/<dir>/image_profile.conf` |
| ④ | Orchestrator (Feldlauf + Scoring) | `eval/orchestrate_images.py` · Registry: `config/image_models.yaml` |
| ④ | Metriken | `eval/metrics/adherence.py` (Treue) · `eval/metrics/ocr_text.py` (Text-CER, containment) |
| ④ | Vergleichsseite | `eval/make_docs.py` |

## Testsatz: v1 und v2 sind nicht vergleichbar

`testset/image_de_v2.jsonl` (seit 2026-09-22) ist v1 plus **einen generischen
Negativ-Prompt je Fall**. Grund: `true_cfg_scale` schaltet bei den
Qwen-Pipelines erst mit Negativ-Prompt echte CFG ein, und in v1 hatte genau
1 von 22 Faellen einen — die Guidance war dort also wirkungslos, waehrend FLUX
und ERNIE ihren `guidance_scale` anwenden.

Der Text ist fuer alle Faelle derselbe. Auf die Pruefkriterien hin formulierte
Negativ-Prompts wuerden die Testdaten messen statt das Modell.

`summary.json` schreibt den Testsatz mit (`"testset"`), und **Zahlen aus v1 und
v2 gehoeren nicht in dieselbe Tabelle** — weder die Qualitaet noch die Zeiten.
Bei der Umstellung wechselte fuer vier Modelle auch das Image, deshalb sind die
Zeiten doppelt unvergleichbar (ERNIE 122 → 51 s/Bild allein durch das Image).

**Rauschmass:** FLUX.1-schnell und FLUX.2-dev bekommen den Negativ-Prompt
gar nicht (guidance-distilliert, der Server verwirft ihn). Ihre Differenz
zwischen zwei Laeufen — 2026-09-27 **+0,055** und **−0,009** in der
Prompt-Treue — ist die Schwelle, ab der eine Aenderung bei den anderen
ueberhaupt etwas bedeutet.

## Warum drei Images

v1 und v2 tragen alles bis FLUX.2. **v3 (2026-09-22) gibt es nur wegen
Qwen-Image-2.1:** `QwenImage21Pipeline` steckt in keinem diffusers-Release —
auch 0.40.0 nicht —, sondern erst im Hauptzweig seit dem 18.09. (PR #14804);
das Dockerfile pinnt deshalb den Commit. Dazu verlangt die Modellkarte
transformers >=5.17, und das schliesst sich mit v2 aus: dort gilt `<5.6`, weil
5.6 den `input_embeds`-kwarg von `create_causal_mask` entfernt hat und
mage_flow daran crasht. Mage-Flow bleibt auf v2, die Registry waehlt das Image
je Modell (`image:`).

**Das v1-Image liegt seit 22.09. nicht mehr lokal vor.** Seit 27.09. steht in
den defaults deshalb v2: laut Aufbau ist es die Obermenge von v1 (gleiche
diffusers-Basis plus die schweren Deps), und mit FLUX.1-schnell verifiziert.

**v2.1 = v2 plus die zwei aktuellen Python-Dateien, sonst nichts.** Am 27.09.
scheiterten alle 22 FLUX.2-Faelle mit
`Flux2Pipeline.__call__() got an unexpected keyword argument 'negative_prompt'`.
Ursache war nicht der Code, sondern das Image: **v2 (gebaut 11.08.) trug eine
alte `server_image.py` ohne den Fix, der nicht unterstuetzte Argumente
verwirft** — der Fix stand seit 11.08. im Repo (Commit 49b0160), war aber nie
ausgeliefert. Im August fiel das nicht auf, weil nur ein einziger Fall einen
Negativ-Prompt trug. Ein voller Neubau von v2 wuerde torchao/nunchaku/
bitsandbytes unpinned neu ziehen und damit die empfindliche FLUX.2-Kette
anfassen; der Aufsatz tauscht nur `server_image.py` und `loaders.py`.

**Lehre:** nach einer Aenderung an `serving/` gehoert das Image neu gebaut,
sonst laeuft der Fix nur im Repo. Pruefen laesst sich das in einer Zeile:

```bash
docker exec southbyte-image md5sum /opt/southbyte-image/server_image.py
md5sum serving/server_image.py
```

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

## Ein Lauf ohne Bild ist ein Fehler

`orchestrate_images.py` prueft seit 27.09. nach jedem Modell, ob ueberhaupt ein
Bild entstanden ist, und endet sonst mit rc=1. Vorher meldete der Feldlauf
rc=0, obwohl alle 22 Faelle mit HTTP 500 gescheitert waren — der Fehllauf sah
von aussen aus wie ein Erfolg.

## Scoring

Ein Judge über alles: **`qwen/qwen3.7-plus`** via LiteLLM-Proxy (`JUDGE_ENDPOINT`/`JUDGE_MODEL`,
`VISION_API_KEY`). Treue = VLM-as-Judge (0..1); Textrendering = **Containment-CER**
(bestraft fehlenden/falschen Soll-Text, nicht zusätzlich korrekt gerenderten).

## Offen (v2-Build)

`docker build -f serving/Dockerfile.image.v2 -t spark-southbyte-image:v2 .` — flash-attn
und NVFP4-Backend auf sm_120 beim Bauen verifizieren; danach Mage-Flow / FLUX.2-klein
in der Config auf `active: true`.
