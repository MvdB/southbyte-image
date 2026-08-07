#!/usr/bin/env python3
"""southbyte-image — Evaluations-Runner (Text-to-Image).

Zwei Phasen (nur ein großes Modell passt in den Speicher):
  1. GENERIEREN: Testset gegen den laufenden Serving-Adapter (:8010) rendern,
     Bilder + Rohdaten sofort auf Platte schreiben (raw-first).
  2. BEWERTEN (optional): gespeicherte Bilder mit einem servierten OCR- bzw.
     Judge-VLM bewerten — separat, weil erst das Bildmodell entladen sein muss.

Beispiel:
  # Phase 1 (Adapter läuft mit FLUX auf :8010):
  python image_eval.py --model-name FLUX.1-schnell
  # Phase 2 später, wenn OCR-/Judge-VLM auf :8000 bzw. :8001 läuft:
  python image_eval.py --model-name FLUX.1-schnell --score-only \
      --ocr-endpoint http://localhost:8000 --ocr-model baidu--Unlimited-OCR \
      --judge-endpoint http://localhost:8001 --judge-model gemma-4
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # vision_client / metrics importierbar machen

from metrics.adherence import judge_adherence  # noqa: E402
from metrics.ocr_text import score_text_rendering  # noqa: E402


def load_testset(path: Path) -> list[dict]:
    cases = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        cases.append(json.loads(ln))
    return cases


def generate_one(endpoint: str, case: dict, args) -> dict:
    """Ein Bild rendern; gibt PNG-Bytes + Timing zurück (oder Fehler)."""
    payload = {
        "prompt": case["prompt"],
        "negative_prompt": case.get("negative_prompt"),
        "n": 1,
        "response_format": "b64_json",
        "size": args.size,
        "steps": args.steps,
        "guidance_scale": args.guidance,
        "seed": args.seed,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/images/generations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        body = json.loads(resp.read())
    dt = time.perf_counter() - t0
    png = base64.b64decode(body["data"][0]["b64_json"])
    return {"png": png, "gen_seconds": body.get("gen_seconds", round(dt, 2))}


def phase_generate(cases: list[dict], run_dir: Path, args) -> list[dict]:
    img_dir = run_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "results_raw.jsonl"
    records: list[dict] = []
    with raw_path.open("w", encoding="utf-8") as raw:
        for case in cases:
            for r in range(args.repeats):
                suffix = f"_r{r}" if args.repeats > 1 else ""
                rec = {"id": case["id"], "category": case["category"],
                       "repeat": r, "prompt": case["prompt"]}
                try:
                    g = generate_one(args.endpoint, case, args)
                    fn = f"{case['id']}{suffix}.png"
                    (img_dir / fn).write_bytes(g["png"])
                    rec.update(image=f"img/{fn}", gen_seconds=g["gen_seconds"])
                    print(f"  ✓ {case['id']}{suffix}  {g['gen_seconds']}s")
                except Exception as e:  # noqa: BLE001 — Ausfall eines Falls kippt den Lauf nicht
                    rec.update(image=None, gen_seconds=None, error=str(e))
                    print(f"  ✗ {case['id']}{suffix}  {e}")
                raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                raw.flush()  # raw-first: nach jedem Fall persistiert
                records.append(rec)
    return records


def phase_score(records: list[dict], cases_by_id: dict, run_dir: Path, args) -> None:
    for rec in records:
        if not rec.get("image"):
            continue
        case = cases_by_id[rec["id"]]
        png = (run_dir / rec["image"]).read_bytes()
        if args.ocr_endpoint and case["category"] == "text_rendering":
            rec["text_rendering"] = score_text_rendering(
                case, png, args.ocr_endpoint, args.ocr_model)
        if args.judge_endpoint and case.get("criteria"):
            rec["adherence"] = judge_adherence(
                case, png, args.judge_endpoint, args.judge_model)
    (run_dir / "results_scored.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8")


def summarize(records: list[dict], run_dir: Path, args) -> dict:
    gens = [r["gen_seconds"] for r in records if r.get("gen_seconds") is not None]
    by_cat_gen: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.get("gen_seconds") is not None:
            by_cat_gen[r["category"]].append(r["gen_seconds"])
    cers = [r["text_rendering"]["cer"] for r in records
            if r.get("text_rendering", {}).get("cer") is not None]
    exacts = [r["text_rendering"]["exact"] for r in records if "text_rendering" in r]
    adh = [r["adherence"]["score"] for r in records
           if r.get("adherence", {}).get("score") is not None]
    summary = {
        "model": args.model_name,
        "cases": len({r["id"] for r in records}),
        "generated": sum(1 for r in records if r.get("image")),
        "failed": sum(1 for r in records if r.get("error")),
        "gen_seconds_mean": round(statistics.mean(gens), 2) if gens else None,
        "gen_seconds_per_category": {c: round(statistics.mean(v), 2)
                                     for c, v in by_cat_gen.items()},
        "text_rendering_cer_mean": round(statistics.mean(cers), 4) if cers else None,
        "text_rendering_exact_rate": round(sum(exacts) / len(exacts), 3) if exacts else None,
        "adherence_score_mean": round(statistics.mean(adh), 3) if adh else None,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="southbyte-image Eval-Runner")
    ap.add_argument("--endpoint", default="http://localhost:8010", help="Serving-Adapter")
    ap.add_argument("--testset", default=str(_HERE.parent / "testset" / "image_de_v1.jsonl"))
    ap.add_argument("--model-name", default="modell", help="Label für den results-Ordner")
    ap.add_argument("--out", default=str(_HERE.parent / "results"))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="nur erste N Fälle (0=alle)")
    ap.add_argument("--size")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--guidance", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--score-only", action="store_true",
                    help="nicht neu generieren, letzten Lauf dieses Modells bewerten")
    ap.add_argument("--ocr-endpoint")
    ap.add_argument("--ocr-model", default="")
    ap.add_argument("--judge-endpoint")
    ap.add_argument("--judge-model", default="")
    args = ap.parse_args()

    cases = load_testset(Path(args.testset))
    if args.limit:
        cases = cases[: args.limit]
    cases_by_id = {c["id"]: c for c in cases}
    out = Path(args.out)

    if args.score_only:
        runs = sorted(out.glob(f"*_{args.model_name}"))
        if not runs:
            print(f"kein Lauf für {args.model_name} unter {out}", file=sys.stderr)
            return 1
        run_dir = runs[-1]
        records = [json.loads(l) for l in
                   (run_dir / "results_raw.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"Bewerte bestehenden Lauf: {run_dir}")
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        run_dir = out / f"{stamp}_{args.model_name}"
        print(f"Generiere ({len(cases)} Fälle × {args.repeats}) → {run_dir}")
        records = phase_generate(cases, run_dir, args)

    if args.ocr_endpoint or args.judge_endpoint:
        print("Bewerte (OCR/Judge) …")
        phase_score(records, cases_by_id, run_dir, args)

    summary = summarize(records, run_dir, args)
    print("\n=== Zusammenfassung ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
