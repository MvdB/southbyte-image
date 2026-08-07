# southbyte-image — Umsetzungsplan

Text-to-Image Serving + Evaluation auf dem NVIDIA DGX Spark (GB10, sm_120, 128 GB
Unified Memory, aarch64). Eigenständiges Repo nach dem Muster von `southbyte-tts`:
diffusers-basierter Serving-Adapter hinter einer OpenAI-kompatiblen API, eigener
Container, eigene Metriken. Teilt Modelle (`~/hf_models` via `southbyte-sync`) und
das Familien-Muster, aber keinen Code mit vLLM.

## Grundsatzentscheidungen (fix)

- **Backend: diffusers** (nicht vLLM, nicht ComfyUI). Alle Zielmodelle liegen
  diffusers-nativ vor; diffusers 0.38 / torch 2.11 / cu130 ist vorhanden.
- **Ein Modell zur Zeit** — je ~54 GB, passt nur einzeln in die 128 GB.
  Der Feldvergleich rotiert die Modelle (Container-Neustart dazwischen).
- **Profile** in `southbyte-spark-profiles/image/<modell>/image_profile.conf`
  (Schritte, Guidance, Scheduler, Auflösung, dtype), gelesen via `$SPARK_PROFILES_DIR`.
- **Objektive Metrik Textrendering**: OCR-Gegenprobe mit lokalem
  `baidu--Unlimited-OCR` bzw. `ibm-granite--granite-docling-258M`.
- **Bild-Moderation aufgeschoben** (konsistent mit den text-only Guards).

## Zielmodelle (lokal vorhanden)

| Modell (`~/hf_models`) | Pipeline | Start-Defaults (zu validieren) |
|---|---|---|
| `black-forest-labs--FLUX.1-schnell` | FluxPipeline | 4 Schritte, guidance 0.0 (distilliert), 1024² |
| `Qwen--Qwen-Image-2512` | QwenImagePipeline | ~50 Schritte, true_cfg 4.0, 1328² |
| `nvidia--Qwen-Image-Flash` | QwenImagePipeline (distilliert) | ~8–15 Schritte, cfg niedrig |

## Repo-Struktur

```
serving/   server_image.py (diffusers FastAPI-Adapter) · Dockerfile.image · run_image.sh
eval/      image_eval.py · metrics/{adherence,ocr_text,perf}.py · make_docs.py
testset/   image_de_v1.jsonl (deutsche Prompts inkl. Textrendering + Fallen)
docs/      publizierte Vergleichsseite (GitHub Pages)
# GB10-Tuning → southbyte-spark-profiles/image/<modell>/image_profile.conf
```

## API (OpenAI-Images-kompatibel)

- `POST /v1/images/generations` — `{prompt, negative_prompt?, size, steps, guidance_scale,
  seed, n, response_format}` → `{created, data:[{b64_json|url}]}`, Header `X-Gen-Time`
- `GET /health`, `GET /v1/models`

## Metriken

**Ein VLM für beide Rollen: Qwen3.6-VL** (lokal `Qwen--Qwen3.6-27B-FP8` bzw.
`-35B-A3B-FP8`, beide multimodal), serviert via vLLM auf **Spark B** — Transkription
(Textrendering) *und* Prompt-Treue-Judge. Kein separates OCR-Modell nötig.

| Metrik | Verfahren | Judge? |
|---|---|---|
| Prompt-Treue | VLM-as-Judge (Qwen3.6-VL, konfigurierbar via `--judge-model`) | VLM |
| **Textrendering** | VLM transkribiert **buchstabengetreu** → CER/exakt vs. erwarteter Text | nein |
| Ästhetik/Qualität | VLM-Judge | VLM |
| Performance | Latenz/Durchsatz je Auflösung × Schritte | nein |
| Safety/NSFW | **aufgeschoben** (TODO) | — |

Generieren und Bewerten laufen **in zwei Phasen** (nur ein großes Modell passt):
erst alle Bilder erzeugen (Bildmodell geladen, Roh-Ausgabe zuerst auf Platte),
dann das Qwen3.6-VL laden und die gespeicherten Bilder bewerten.

**Wichtig — Auto-Korrektur-Falle:** instruktionsgetunte VLMs „reparieren" gern die
Schreibweise (lesen „Grunwald" als „Grünwald") und verdecken damit genau die
Fehler, die wir messen. Der Transkriptions-Prompt (`_OCR_SYSTEM` in `ocr_text.py`)
erzwingt daher buchstabengetreue Ausgabe; vor dem ersten echten Lauf gegen die
bekannt-fehlerhaften FLUX-Bilder (img-007 „BAKEERRIE", img-010 „Grunwald")
**kalibrieren** (Muster von `judge_bench.py`: erst die Metrik prüfen, dann trauen).

## Wiederverwendung aus der Familie

Framework-Rückgrat übertragbar aus `southbyte-vllm/testplan` / `southbyte-tts/eval`:
Config-Loader, `EvalResult`/`Verdict`-Dataclasses, Testdaten-Loader/Validator,
Reporter/Dashboard, das raw-first + `results/YYYY-MM-DD_<modell>/`-Muster. Neu:
Serving-Adapter (diffusers) und die konkreten Bildmetriken/Testset.

## Phasen

1. **Serving** — `server_image.py` + `Dockerfile.image` + `run_image.sh` + Profile
   für die 3 Modelle; End-to-End-Erzeugung je Modell validieren. ⟵ *aktuell*
2. **Testset** — deutsche Prompts inkl. Textrendering-Fälle + Schema.
3. **Eval** — Prompt-Treue (CLIP + VLM-Judge) + OCR-Textrendering + Performance.
4. **Feldvergleich** — 3 Modelle, `docs/`-Vergleichsseite publizieren.
5. **Aufgeschoben** — VLM-Judge-Kalibrierung (Analog `judge_bench.py`), Bild-Moderation.

---

Built by [southbyte](https://southbyte.de).
