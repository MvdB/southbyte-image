# southbyte-image — Bild-Modell-Harness

Familien-agnostischer Betrieb + Evaluation von Text-zu-Bild-Modellen auf dem DGX Spark.
Ein Modell hinzufügen = **Profil/Config-Eintrag + Loader-Namen** — kein Code.

## Schichten

| # | Teil | Datei |
|---|------|-------|
| ① | Serving-Image (alle Deps) | `serving/Dockerfile.image` (v1) · `.v2` (+ mage_flow/nunchaku/bnb) · `.v2.1` (= v2 + aktueller Servercode) · `.v3` (Qwen-Image-2.1) · `.v3.1` (Qwen-Image-2.1-Turbo) · `.v4` (FLUX.3 [klein]) · `.v4.1` (+ bitsandbytes/outlines: ideogram-4, Krea-2) · `.hidream` (HiDream-O1: transformers 4.57.1 + Modellcode) |
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

## Warum mehrere Images

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

**v4 (2026-10-20) gibt es nur wegen FLUX.3 [klein]:** FLUX.3 steckt in keinem
diffusers-Release, sondern nur im Hauptzweig, und dort ausschliesslich als
ModularPipeline; den Prompt-Upsampler als Block der Pipeline gibt es erst seit
PR #15058. v4 ist v3.1 mit dem diffusers-Pin `4df4a5b35748` (19.10.) und prueft
beim Bau, dass `Flux3DistilledAutoBlocks` den Block `prompt_upsampler` enthaelt
und die Loader-Registry `flux3_modular` kennt. Gebaut wird mit Commit-Stempel,
`/health` zeigt ihn als `version`:

```bash
docker build -f serving/Dockerfile.image.v4 \
  --build-arg SOUTHBYTE_COMMIT="$(git rev-parse --short HEAD)" -t spark-southbyte-image:v4 .
```

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
| `flux3_modular` | FLUX.3 [klein] (ModularPipeline + Prompt-Upsampler aus `components_dir`) | **v4** |
| `ideogram4` | ideogram-4 (NF4 + lokales Prompt-Upsampling, LM-Kopf aus `components_dir`) | **v4.1** |
| `hidream_o1` | HiDream-O1-Image (eigener Modellcode, `/opt/hidream-o1`) | **hidream** |

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

## FLUX.3: der Prompt, den das Bildmodell sieht, ist ein anderer

FLUX.3 [klein] ist laut Modellkarte nicht ohne Prompt-Upsampler zu betreiben:
ein eigenes 9B-Modell (Qwen3.5-Finetune, `black-forest-labs--flux-3-image-klein-pup`)
schreibt die Anfrage vor dem Text-Encoder zu einer vollstaendigen
Szenenbeschreibung um — auf Englisch, zitierter Text behaelt seine Sprache.

- **Laden:** `modular_model_index.json` verweist per Hub-ID auf das Upsampler-Repo,
  und das ist offline nicht aufloesbar. Der Loader laedt ihn deshalb ausdruecklich
  aus `components_dir` und setzt ihn per `update_components` ein; alles andere
  kommt aus dem Modellverzeichnis, die VAE in fp32.
- **Nachvollziehbar:** Der Server gibt den umgeschriebenen Prompt je Bild als
  `revised_prompt` zurueck (Feldname wie in der OpenAI-Images-API), dazu
  `upsample_seconds` und `upsample_tokens`. `image_eval.py` schreibt alle drei
  in `results_raw.jsonl`; `summary.json` bekommt `upsample_seconds_mean` und
  `upsample_tokens_max`. `gen_seconds` bleibt die Gesamtzeit inklusive Upsampler.
- **Bewertet wird gegen den Original-Prompt** des Testsatzes, nicht gegen den
  umgeschriebenen — gemessen wird, ob das Bild die Anfrage trifft.
- **Ohne Upsampler** nur fuer Vergleichslaeufe: `prompt_upsampler: false` in der
  Registry entfernt den Block aus der Pipeline.
- **Offen:** diffusers dekodiert den Upsampler greedy, die BFL-Referenz (vLLM)
  setzt fuer Text-zu-Bild zusaetzlich eine presence penalty von 1,5. Stoesst
  `upsample_tokens_max` an 3584, ist die Beschreibung abgeschnitten.

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

## EU-Kennzeichnung KI-erzeugter Inhalte

Jedes Bild in `docs/img/` traegt das Icon der Europaeischen Kommission
(Variante **AI GENERATED**), unten links. Der Seitenbau macht das selbst —
**kein manueller Zwischenschritt**, und `make_docs.py` endet mit rc=1, wenn
auch nur ein Bild ungekennzeichnet veroeffentlicht wuerde.

**Nur auf den veroeffentlichten Kopien, nie auf den gemessenen Bildern.** Das
ist keine Bequemlichkeit, sondern Pflicht: unsere Bewertung laeuft ueber ein
Bildmodell als Judge und misst gerenderten Text per OCR. Ein eingebranntes
Label mit den Woertern "AI GENERATED" liefe in die Text-CER und in die
Prompt-Treue — wir wuerden unser eigenes Wasserzeichen mitmessen. `results/`
bleibt die Beweisebene, `docs/img/` ist die Veroeffentlichung.

**Position unten links**, weil unsere Szenen das Motiv meist mittig bis oben
tragen und Provenienzangaben konventionell unten gesucht werden. Die Vorgabe
der Kommission ("bei erstem Kontakt klar wahrnehmbar, sichtbar auch beim
Weiterteilen oder Herunterladen") ist erfuellt, weil das Label IM Bild steckt
und nicht nur in der Seite drumherum. Das Raster-Thumbnail wird aus der
gekennzeichneten Kopie gezogen, traegt es also ebenfalls.

**Schwarz oder weiss entscheidet das Bild.** Die Kommission liefert beide
Fassungen mit; der Seitenbau misst die Helligkeit der Zielflaeche und nimmt die
passende (ueber die aktuelle Kohorte: 80x schwarz, 74x weiss). Eine erste
Fassung mit selbstgebauter dunkler Hinterlegung wurde verworfen — grauer Kasten
mit harten Kanten, und ein Eingriff in fremdes Gestaltungsmaterial.

Assets in `assets/eu-ki-kennzeichnung/` (SVG als Quelle, PNG auf 1200 px
Breite fuer die Bildkennzeichnung). Quelle:
[digital-strategy.ec.europa.eu](https://digital-strategy.ec.europa.eu/en/policies/eu-icons-labelling-ai-generated-content).
Die Icons stuetzen Artikel 50(4) KI-Verordnung (Deepfakes und KI-Texte zu
Themen oeffentlichen Interesses). Unsere Testbilder sind weder das eine noch
das andere — die Kennzeichnung ist hier **freiwillig**; die rechtliche
Bewertung liegt bei Michael, nicht im Harness.

**Noch offen:** `southbyte-tts` veroeffentlicht KI-erzeugte Audiodateien. Ein
Bild-Icon laesst sich dort nicht einbrennen; wenn das gekennzeichnet werden
soll, braucht es einen eigenen Weg (Hinweis auf der Seite, Metadaten in der
Datei).

## Lizenzen der Modelle

`config/models.yaml` in southbyte-vllm fuehrt Release und Lizenz je Modell; die
Vergleichsseite baut daraus einen sichtbaren Hinweis fuer alle Modelle mit
nicht-kommerzieller Lizenz (`_NICHT_KOMMERZIELL` in `eval/make_docs.py`).
Betroffen sind aktuell **FLUX.2-dev** und **Qwen-Image-2.1** (Qwen Research
License, 2026-09-20: Nutzung nur fuer Forschung und Evaluation), dazu **ideogram-4**. Seit 25.10.
hat jede Lizenz ihren eigenen, am Lizenztext geprueften Satz, denn die
Lizenzen regeln die Ausgaben verschieden:

- **Qwen Research:** keine eigene Regel fuer die Ausgaben, die Einschraenkung
  gilt deshalb auch fuer die Bilder. Dass die Messbilder trotzdem auf der Seite
  stehen, ist eine bewusste Entscheidung vom 22.09.2026.
- **FLUX.2-dev:** das Modell nur nicht-kommerziell, die Ausgaben ausdruecklich
  "for any purpose (including for commercial purposes)" — ausser zum Training
  konkurrierender Modelle, und mit Inhaltsfilter oder Pruefung sowie
  KI-Kennzeichnung. Der fruehere Satz "gilt auch fuer die Bilder" war hier falsch.
- **ideogram-4:** nur nicht-kommerziell, Unternehmen nur Test und Evaluation in
  nicht-produktiver Umgebung, und ausdruecklich keine Ausgaben, um umsatzbringende
  Produkte oder Dienste zu bewerben. Die Bilder stehen mit diesem Hinweis in der
  Galerie (Entscheidung vom 25.10.2026: Vergleich aktueller Modelle, kein
  kommerzieller Zweck). Der Schalter `galerie: false` in der Registry zeigte nur
  die Messwerte, die Bilder weder als Raster noch als Datei.

Eine Lizenzkennung, die weder in `_OHNE_HINWEIS` noch in einer der beiden
Klassen steht, bricht den Seitenbau ab (rc=1, keine Datei) — eine neue Lizenz
faellt sonst stumm durch alle Hinweise. Dasselbe prueft `pruefe_metadaten` in
southbyte-results fuer die Bildtabelle.

**FLUX.3 [klein]** steht unter der FLUX Kommunity License v1.0 (`flux-kommunity`)
und bekommt einen eigenen Hinweis (`_MIT_BEDINGUNGEN`): nicht-kommerzielle
Nutzung erlaubt, kommerzielle Nutzung der Bilder nur fuer Anwender unter
5 Mio. US-Dollar Jahresumsatz und nur mit Inhaltsfilter oder Pruefung vor der
Veroeffentlichung (Abschnitt 2.e) sowie KI-Kennzeichnung, soweit gesetzlich
verlangt. Einen Inhaltsfilter hat die diffusers-Pipeline nicht — die
Messbilder also vor dem Commit von `docs/img/` ansehen. Upsampler und
Text-Encoder stehen unter Apache-2.0.

Beim Nachtragen in `models.yaml`: **kein Kommentar hinter dem Wert.** Die
Parser in `eval/make_docs.py` und `southbyte-results/feeds.py` lesen die Zeile
mit einem regulaeren Ausdruck, der an `#` abbricht — die Lizenz erscheint dann
als „—" auf der Seite.

## Vier Modelle vom 25.10.2026: was jeweils anders ist

**Krea-2-Turbo** (`diffusion`, v4.1): Destillat, 8 Schritte, `guidance 0.0` —
CFG aus, den Negativ-Prompt ignoriert die Pipeline dann. Krea-Lizenz mit eigenem
Hinweis (`_MIT_BEDINGUNGEN`): kommerziell nur unter 1 Mio. US-Dollar Umsatz,
Inhaltsfilter bei jedem Einsatz (Pruefung durch Menschen genuegt) — die Bilder
also vor dem Commit von `docs/img/` ansehen.

**ideogram-4** (`ideogram4`, v4.1): erwartet seine strukturierte JSON-Beschreibung.
Die gehostete Magic-Prompt-API ist ausgeschlossen, die Testprompts verlassen die
Maschine nicht; lokal schreibt der Text-Encoder mit dem LM-Kopf
`diffusers/qwen3-vl-8b-instruct-lm-head` um, JSON-gebunden ueber `outlines`. Die
Guidance ist ein Plan je Schritt (45 × 7,0, 3 × 3,0) — `guidance: none` in der
Registry, damit der Server keinen `guidance_scale` daneben setzt (die Pipeline
lehnt beides zusammen ab), und `steps: 48`, weil die Planlaenge daran haengt.
Fehlt der LM-Kopf, gibt es die Ersatzzeile `ideogram-4-ohne-Upsampling`
(`active: false`); ihre Zahlen gehoeren nicht neben die mit Upsampling.

Falle: `ideogram-ai/ideogram-4-nf4` ist NICHT die diffusers-Fassung, obwohl es eine
`model_index.json` traegt. Seine Transformer speichern die Attention fusioniert
(`attention.qkv`, `attention.o`); `Ideogram4Transformer2DModel` erwartet `to_q`, `to_k`,
`to_v`, `to_out` — 136 Parameter bleiben auf `meta`, der Start bricht mit "Cannot copy
out of meta tensor" ab (25.10.). Die diffusers-Fassung ist
`ideogram-ai/ideogram-4-nf4-diffusers` (so auch im diffusers-Beispiel der Karte);
nachrechnen geht nicht, die Gewichte sind NF4-quantisiert.

**Ming-Image-0.1-Design** (vLLM-Omni, `vllm/vllm-omni:v0.30.0`): nach dem Rezept
im vLLM-Omni-Repo zweistufig mit `--deploy-config`, die Registry reicht sie ueber
`omni_args` an `serving/run_omni.sh`; beide Stufen auf der einen GPU
(`serving/omni/ming_image_einzel.yaml`, Anteile 0,35 und 0,30). Bilder kommen ueber
chat/completions mit `modalities: ["image"]` (`api: chat`, `image_eval.py --api chat`),
nicht ueber `/v1/images/generations`.

Falle: Das Modell-Repo bekam am 2026-10-14 eine `config.json` im Wurzelverzeichnis
("Hub download tracking"). Mit ihr haelt vLLM-Omni die Sprachstufe fuer das Modell
und laedt die Diffusionsstufe ueber die Z-Image-Pipeline — Abbruch beim Laden mit
`Attempted to load weight (384) into parameter (64)` (VAE-Gewichte), in v0.30.0 und
v0.32.0rc1 gleich. `ausblenden: ["config.json"]` in der Registry laesst
`run_omni.sh` eine Ansicht aus Verweisen ohne diese Datei einhaengen
(`results/_ansichten/`); dann greift `MingImageDiffusionPipeline`. Der Modellordner
bleibt, wie ihn der Sync liefert. Gemessen am 25.10. mit v0.30.0, dem Stand des
Rezepts (vLLM-Omni-Commit 6daf5b30f vom 23.09.).

**HiDream-O1-Image-Dev** (`hidream_o1`, Image `hidream`): eigener Pixel-DiT auf
Qwen3-VL, Code von HiDream (MIT, gepinnt). Rendert immer **2048x2048** — der Code
rastet auf seine Formatliste ein, ein 1024er-Format gibt es nicht; die Zeiten
sind deshalb nicht mit den 1024er-Laeufen vergleichbar. Der Prompt-Refiner der
Referenz (Prompt-Refine, Gemma-4-31B, ~59 GB) laeuft **vorab**: der Orchestrator
startet ihn (`refiner:` in der Registry), `eval/prompts_vorab.py` schreibt im
HiDream-Image mit dem Agenten der Referenz alle Faelle um, dann stoppt der
Refiner und erst danach laedt das Bildmodell. `image_eval.py --vorab` nimmt die
Texte, rechnet die Refiner-Zeit in `gen_seconds` ein und vermerkt Rueckfaelle
auf den Originalprompt (`vorab_fallback`).

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
