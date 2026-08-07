# southbyte-image

> **Status: in progress (Phase 1 — serving).** Implementation plan in
> [`docs/ROADMAP.md`](docs/ROADMAP.md). The diffusers serving adapter
> (`serving/`) is in place; testset + evaluation harness are next. Not yet
> validated end-to-end on hardware.

Text-to-image serving + evaluation on the **NVIDIA DGX Spark** (GB10 SoC, sm_120,
128 GB unified memory, aarch64). It will follow the same family conventions as the
other stacks:

- reads models from `~/hf_models/` (populated by [southbyte-sync](https://github.com/MvdB/southbyte-sync))
- GB10-tuned serving profiles in [southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles) under `image/`
- an evaluation harness mirroring [southbyte-vllm](https://github.com/MvdB/southbyte-vllm)'s `testplan/`,
  extended with image-specific test types (prompt adherence, quality, safety)

## Part of the southbyte family

- [southbyte-core](https://github.com/MvdB/southbyte-core) — shared index
- [southbyte-sync](https://github.com/MvdB/southbyte-sync) — HuggingFace collection mirror → local model store
- [southbyte-vllm](https://github.com/MvdB/southbyte-vllm) — vLLM serving runner + LLM evaluation testplan
- [southbyte-tts](https://github.com/MvdB/southbyte-tts) — TTS/STT serving + German-language evaluation
- [southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles) — DGX Spark (GB10) validated profiles, kernels, benchmarks
- **southbyte-image** — text-to-image serving + evaluation *(this repo — planned)*

## License

MIT — see [LICENSE](LICENSE)

---

Built by [southbyte](https://southbyte.de).
