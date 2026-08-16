# southbyte-image

Text-to-image on an **NVIDIA DGX Spark**: one serving adapter for any diffusers
model, and an evaluation harness that scores them against a German testset and
publishes the comparison.

**→ [The comparison page](https://mvdb.github.io/southbyte-image/)** — six models,
22 cases each, generated and scored on the machine.

> **Proof of concept, not a product.** No guaranteed availability, fitness or
> output quality, no support, no roadmap.

## What it does

- **Serves** any text-to-image model from the local model store behind one
  OpenAI-Images-compatible API. Four loader strategies cover standard diffusers,
  custom pipeline classes, external libraries and NVFP4 single-file checkpoints.
- **Evaluates** them against 22 German prompts across five categories —
  adherence (6), text rendering (6), composition (3), style (3) and traps (4) —
  with two metrics: prompt adherence scored by a vision judge, and a containment
  CER for text the image was asked to render.
- **Publishes** a comparison page with images, per-case scores and timings.
- **Adding a model is configuration, not code**: one entry in
  `config/image_models.yaml` plus a profile in
  [southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles).

Six models measured so far, 22 cases each:

| Model | Generation | Text-CER | Exact text | Adherence |
|---|---|---|---|---|
| ERNIE-Image-Turbo | 121.9 s | **0.000** | **100 %** | **0.973** |
| FLUX.2-dev-bnb4 | 96.4 s | 0.017 | 83 % | **0.973** |
| Qwen-Image-Flash | 22.6 s | 0.026 | 67 % | 0.927 |
| Qwen-Image-2512 | 139.0 s | 0.049 | 50 % | 0.891 |
| Mage-Flow | 15.0 s | 0.021 | 83 % | 0.882 |
| FLUX.1-schnell | **6.8 s** | 0.147 | 33 % | 0.700 |

Lower CER is better, higher adherence is better. These are single runs on one
testset, not a benchmark — see *What to watch out for*.

## Getting it running

```bash
# 1. Build the serving image
docker build -t spark-southbyte-image:v1 -f serving/Dockerfile.image serving/

# 2. Start the adapter on port 8010 (model from ~/hf_models, read-only)
cd serving && ./run_image.sh
MODEL_DIR=Qwen--Qwen-Image-Flash ./run_image.sh    # or pick another

# 3. Generate, score and rebuild the comparison page
python eval/orchestrate_images.py --models Qwen-Image-Flash
python eval/orchestrate_images.py                   # all active models
```

You need Docker with GPU access, the models in `~/hf_models/` (populated by
[southbyte-sync](https://github.com/MvdB/southbyte-sync)), and — for scoring
only — a vision judge reachable over the network.

**Scoring is not local.** Generation runs on your GPU; the adherence and OCR
metrics call a vision model through an OpenAI-compatible endpoint. Set it before
scoring, or run `--no-score` to generate only:

| Variable | |
|---|---|
| `JUDGE_ENDPOINT` | base URL of the judge (OpenAI-compatible `/v1/chat/completions`) |
| `JUDGE_MODEL` | the model to ask |
| `VISION_API_KEY` | its key |

Mage-Flow and the FLUX.2 NVFP4 loaders need the larger image instead:
`docker build -t spark-southbyte-image:v2 -f serving/Dockerfile.image.v2 .`
(adds flash-attn, `mage_flow`, nunchaku).

Layers, loader strategies and how to add a model: [`HARNESS.md`](HARNESS.md).

## What to watch out for

**One big model at a time.** Generation and scoring run as two phases because
only one large model fits in memory. The orchestrator starts a model, generates,
stops it, then scores — do not expect to keep two adapters up.

**The numbers above are one run per model, not a benchmark.** One testset, one
seed set, no repeats. The vision judge itself has never been calibrated against
known-correct answers the way the TTS judge was, so absolute scores carry more
uncertainty than the table suggests. Compare deltas, not decimals.

**Three FLUX.2 NVFP4 variants are configured but inactive** (`FLUX.2-klein-9b`,
`-4b`, `FLUX.2-dev`). They wait on the v2 image being verified for flash-attn and
the NVFP4 backend on sm_120. `FLUX.2-dev-bnb4` is the 4-bit bitsandbytes build
and does run.

**Generated images and raw runs stay local.** `results/` and `*.png`/`*.jpg` are
gitignored; only the curated gallery under `docs/img/` is committed.

**Port 8010** is this adapter's. 8000–8011 are taken by the other stacks in the
family.

## Licence

Code in this repository is **MIT** — see [LICENSE](LICENSE).

**Model licences travel with the models, not with this repository.** Each model
under evaluation carries its own terms; check them before using output for
anything beyond experimentation. The comparison page names each model and links
it upstream.

**Vendored third-party code, and it is signposted.** `_vendor/mage_flow/` is
taken from [microsoft/Mage](https://github.com/microsoft/Mage), which is MIT.
Its licence and copyright notice travel with it as
[`_vendor/mage_flow/LICENSE`](_vendor/mage_flow/LICENSE); origin and what was
copied are in [`_vendor/mage_flow/HERKUNFT.md`](_vendor/mage_flow/HERKUNFT.md).

## Where this is going

The field comparison is done and published. What remains is named rather than
planned, and both items are in [`docs/ROADMAP.md`](docs/ROADMAP.md) as deferred:

- **Judge calibration.** [southbyte-tts](https://github.com/MvdB/southbyte-tts)
  proved its ASR judge against audio whose content was known before trusting it.
  Nothing equivalent has been done for the vision judge here, which is the
  biggest open question about the numbers above.
- **Image moderation.** No safety classification of generated images.

Issues and pull requests are welcome; nobody is on call for them.

## Part of the southbyte family

- [southbyte-core](https://github.com/MvdB/southbyte-core) — shared index
- [southbyte-sync](https://github.com/MvdB/southbyte-sync) — HuggingFace mirror → local model store
- [southbyte-vllm](https://github.com/MvdB/southbyte-vllm) — vLLM runner + LLM testplan
- [southbyte-tts](https://github.com/MvdB/southbyte-tts) — TTS/STT serving + German evaluation
- [southbyte-music](https://github.com/MvdB/southbyte-music) — text-to-music serving + web interface
- [southbyte-results](https://github.com/MvdB/southbyte-results) — cross-modality results site
- [southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles) — GB10 profiles, kernels, benchmarks
- **southbyte-image** — text-to-image serving + evaluation *(this repository)*

---

Built by [southbyte](https://southbyte.de).
