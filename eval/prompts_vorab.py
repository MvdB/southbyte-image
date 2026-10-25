#!/usr/bin/env python3
"""Prompts vorab umschreiben lassen — fuer Bildmodelle, deren Refiner nicht neben
dem Bildmodell in den Speicher passt (bisher HiDream-O1 mit Prompt-Refine,
einem Gemma-4-31B-Finetune von ~59 GB).

Laeuft im Image `spark-southbyte-image:hidream`, weil dort der Prompt-Agent der
Referenz liegt (/opt/hidream-o1/prompt_agent_v2.py). Aufgerufen wird genau
dessen refine_prompt(): derselbe Systemprompt, dieselbe Wiederholung bei
entarteter Ausgabe, derselbe Rueckfall auf den Originalprompt. Der Refiner
selbst laeuft vorher als vLLM-Dienst; dieses Skript braucht keine GPU.

Ergebnis je Fall eine Zeile JSON: id, revised_prompt, upsample_seconds und
fallback (true, wenn der Agent auf den Originalprompt zurueckfiel).
eval/image_eval.py liest die Datei mit --vorab und rechnet die Zeit in
gen_seconds ein, wie beim FLUX.3-Upsampler.

  docker run --rm --network host -v "$PWD/testset:/testset:ro" -v "$OUT:/out" \\
    spark-southbyte-image:hidream python prompts_vorab.py \\
    --testset /testset/image_de_v2.jsonl --base-url http://localhost:8014/v1 \\
    --model HiDream-ai--Prompt-Refine --out /out/vorab.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Prompts vorab ueber den Refiner der Referenz umschreiben")
    ap.add_argument("--testset", required=True)
    ap.add_argument("--base-url", required=True, help="OpenAI-kompatibler Refiner, etwa http://localhost:8014/v1")
    ap.add_argument("--model", required=True, help="served-model-name des Refiners")
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent-dir", default="/opt/hidream-o1", help="Verzeichnis mit prompt_agent_v2.py")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, args.agent_dir)
    from prompt_agent_v2 import refine_prompt  # type: ignore

    faelle = [json.loads(z) for z in Path(args.testset).read_text(encoding="utf-8").splitlines()
              if z.strip() and not z.lstrip().startswith("#")]
    if args.limit:
        faelle = faelle[: args.limit]
    ziel = Path(args.out)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    rueckfall = 0
    with ziel.open("w", encoding="utf-8") as f:
        for fall in faelle:
            t0 = time.perf_counter()
            text = refine_prompt(fall["prompt"], model_id=args.model, base_url=args.base_url)
            dauer = round(time.perf_counter() - t0, 2)
            fallback = text.strip() == fall["prompt"].strip()
            rueckfall += fallback
            f.write(json.dumps({"id": fall["id"], "revised_prompt": text, "upsample_seconds": dauer,
                                "fallback": fallback}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {'↩' if fallback else '✓'} {fall['id']}  {dauer}s  {len(text.split())} Woerter", flush=True)
    print(f"{len(faelle)} Prompts, {rueckfall} davon unveraendert (Rueckfall) → {ziel}")
    # Faellt der Agent ueberall zurueck, lief der Refiner nicht — dann ist der
    # Lauf keiner mit Refiner und soll nicht so aussehen.
    return 1 if faelle and rueckfall == len(faelle) else 0


if __name__ == "__main__":
    raise SystemExit(main())
