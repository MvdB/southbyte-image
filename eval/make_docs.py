#!/usr/bin/env python3
"""southbyte-image — baut die publizierte Vergleichsseite docs/index.html.

Liest results/<datum>_<modell>/ (summary.json + results_scored/raw.jsonl),
kopiert die Bilder self-contained nach docs/img/<modell>/ und rendert eine
Vergleichstabelle (Modelle × Metriken) plus eine Galerie je Testfall.
Kein GPU nötig; nur stdlib.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def load_runs(results: Path) -> list[dict]:
    """Neuester Lauf je Modell (letzter alphabetisch = neuestes Datum)."""
    runs: dict[str, dict] = {}
    for d in sorted(results.glob("*_*")):
        sj = d / "summary.json"
        if not sj.exists():
            continue
        summary = json.loads(sj.read_text(encoding="utf-8"))
        scored = d / "results_scored.jsonl"
        raw = d / "results_raw.jsonl"
        src = scored if scored.exists() else raw
        recs = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()] if src.exists() else []
        runs[summary.get("model", d.name)] = {"dir": d, "summary": summary, "records": recs}
    return list(runs.values())


def _fmt(v, suffix="") -> str:
    return "—" if v is None else f"{v}{suffix}"


def build_html(runs: list[dict], docs: Path) -> str:
    img_root = docs / "img"
    if img_root.exists():
        shutil.rmtree(img_root)

    # Metrik-Übersicht
    head = "".join(f"<th>{html.escape(r['summary']['model'])}</th>" for r in runs)
    def row(label, key, suffix=""):
        cells = "".join(f"<td>{_fmt(r['summary'].get(key), suffix)}</td>" for r in runs)
        return f"<tr><th>{label}</th>{cells}</tr>"
    metrics = "\n".join([
        row("Bilder erzeugt", "generated"),
        row("Ø Zeit/Bild", "gen_seconds_mean", " s"),
        row("Textrendering CER (Ø)", "text_rendering_cer_mean"),
        row("Textrendering exakt", "text_rendering_exact_rate"),
        row("Prompt-Treue (Ø)", "adherence_score_mean"),
    ])

    # Galerie je Testfall
    case_ids: list[str] = []
    for r in runs:
        for rec in r["records"]:
            if rec["id"] not in case_ids:
                case_ids.append(rec["id"])
    gallery = []
    for cid in case_ids:
        prompt = next((rec.get("prompt", "") for r in runs for rec in r["records"] if rec["id"] == cid), "")
        cells = []
        for r in runs:
            rec = next((x for x in r["records"] if x["id"] == cid and x.get("repeat", 0) == 0), None)
            if rec and rec.get("image"):
                src_img = r["dir"] / rec["image"]
                dst_rel = f"img/{r['summary']['model']}/{Path(rec['image']).name}"
                dst = docs / dst_rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src_img.exists():
                    shutil.copy2(src_img, dst)
                extra = []
                if rec.get("adherence", {}).get("score") is not None:
                    extra.append(f"Treue {rec['adherence']['score']}")
                if rec.get("text_rendering", {}).get("cer") is not None:
                    extra.append(f"CER {rec['text_rendering']['cer']}")
                cap = " · ".join(extra)
                cells.append(f'<td><img loading="lazy" src="{html.escape(dst_rel)}" alt="{cid}"><div class="cap">{html.escape(cap)}</div></td>')
            else:
                cells.append('<td class="miss">—</td>')
        gallery.append(f'<tr><th class="cid">{cid}<div class="pr">{html.escape(prompt)}</div></th>{"".join(cells)}</tr>')

    empty = "" if runs else '<p class="empty">Noch keine Ergebnisse — erst einen Feldlauf mit <code>image_eval.py</code> fahren.</p>'
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>southbyte-image — Modellvergleich</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0}}
 th,td{{border:1px solid #ddd;padding:.5rem;text-align:center;vertical-align:top}}
 img{{max-width:220px;height:auto;border-radius:4px}}
 .cid{{text-align:left;font-family:monospace}} .pr{{font:400 .8rem/1.3 sans-serif;color:#666;max-width:220px}}
 .cap{{font-size:.75rem;color:#555;margin-top:.25rem}} .miss{{color:#bbb}} .empty{{color:#a00}}
 a{{color:#06c}}
</style></head><body>
<h1>southbyte-image — Modellvergleich (DGX Spark / GB10)</h1>
{empty}
<h2>Metriken</h2>
<table><tr><th>Metrik</th>{head}</tr>{metrics}</table>
<h2>Galerie</h2>
<table><tr><th>Fall</th>{head}</tr>{"".join(gallery)}</table>
<hr><p>Built by <a href="https://southbyte.de">southbyte</a>.</p>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="southbyte-image Vergleichsseite bauen")
    ap.add_argument("--results", default=str(_ROOT / "results"))
    ap.add_argument("--docs", default=str(_ROOT / "docs"))
    args = ap.parse_args()
    docs = Path(args.docs)
    docs.mkdir(parents=True, exist_ok=True)
    runs = load_runs(Path(args.results))
    (docs / "index.html").write_text(build_html(runs, docs), encoding="utf-8")
    print(f"✓ docs/index.html geschrieben ({len(runs)} Modell(e))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
