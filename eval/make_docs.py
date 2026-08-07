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
_THUMB_PX = 360


def _make_thumb(src: Path, dst: Path, max_px: int = _THUMB_PX) -> bool:
    """Kleines JPEG-Thumbnail nur fürs Overview-Raster; Original bleibt erhalten.
    Rückgabe False, falls Pillow fehlt (dann nutzt der Aufrufer das Original)."""
    try:
        from PIL import Image
    except ImportError:
        return False
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_px, max_px))
        im.save(dst, "JPEG", quality=82)
    return True


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
                model = r["summary"]["model"]
                name = Path(rec["image"]).name
                full_rel = f"img/{model}/{name}"                        # Original (behalten)
                thumb_rel = f"img/{model}/thumb/{Path(name).stem}.jpg"  # nur Overview
                (docs / full_rel).parent.mkdir(parents=True, exist_ok=True)
                (docs / thumb_rel).parent.mkdir(parents=True, exist_ok=True)
                thumbed = src_img.exists() and _make_thumb(src_img, docs / thumb_rel)
                if src_img.exists():
                    shutil.copy2(src_img, docs / full_rel)
                if not thumbed:
                    thumb_rel = full_rel  # Fallback ohne Pillow: Original im Raster
                extra = []
                if rec.get("adherence", {}).get("score") is not None:
                    extra.append(f"Treue {rec['adherence']['score']}")
                if rec.get("text_rendering", {}).get("cer") is not None:
                    extra.append(f"CER {rec['text_rendering']['cer']}")
                cap = " · ".join(extra)
                cells.append(f'<td><a href="{html.escape(full_rel)}" title="Original"><img loading="lazy" src="{html.escape(thumb_rel)}" alt="{cid}"></a><div class="cap">{html.escape(cap)}</div></td>')
            else:
                cells.append('<td class="miss">—</td>')
        gallery.append(f'<tr><th class="cid">{cid}<div class="pr">{html.escape(prompt)}</div></th>{"".join(cells)}</tr>')

    empty = "" if runs else '<p class="empty">Noch keine Ergebnisse — erst einen Feldlauf mit <code>image_eval.py</code> fahren.</p>'
    # SouthByte Web-CI (southbyte-brand skill): Dark-Theme, Matrix-Grid, Wortmarke.
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SOUTH.BYTE — Text-to-Image Modellvergleich</title>
<style>
 :root{{--bg:#060C0A;--bg-raised:#0A1410;--bg-card:#0E1A14;--border:#162A1E;--border-hi:#1A5C38;
   --green:#00E676;--green-dim:#00994A;--text:#D4EDE0;--text-muted:#5E8A72;
   --mono:'Courier New',Consolas,'SF Mono',Menlo,monospace;--sans:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);line-height:1.7}}
 .grid-bg{{position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.5;
   background-image:linear-gradient(rgba(0,230,118,.15) 1px,transparent 1px),
     linear-gradient(90deg,rgba(0,230,118,.15) 1px,transparent 1px);background-size:80px 80px}}
 .wrap{{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:2.5rem 1.25rem}}
 .wordmark{{font-family:var(--mono);font-weight:700;font-size:1.4rem;letter-spacing:1.4px;color:var(--text)}}
 .wordmark .dot{{color:var(--green)}}
 .tagline{{font-family:var(--mono);font-size:.68rem;letter-spacing:.25em;text-transform:uppercase;
   color:var(--text-muted);margin-top:.3rem}}
 h1{{font-family:var(--mono);color:var(--text);font-size:1.7rem;margin:1.4rem 0 .4rem}}
 h2{{font-family:var(--mono);text-transform:uppercase;letter-spacing:.15em;color:var(--green);
   border-top:1px solid var(--border-hi);padding-top:.6rem;margin-top:2.4rem}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0}}
 th,td{{border:1px solid var(--border);padding:.5rem;text-align:center;vertical-align:top}}
 th{{font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
   color:var(--text-muted);background:var(--bg-raised)}}
 img{{max-width:220px;height:auto;border-radius:4px;border:1px solid var(--border)}}
 .cid{{text-align:left;font-family:var(--mono);color:var(--green)}}
 .pr{{font:400 .8rem/1.3 var(--sans);color:var(--text-muted);max-width:220px}}
 .cap{{font-size:.75rem;color:var(--text-muted);margin-top:.25rem}} .miss{{color:var(--text-dim,#2E5040)}}
 .empty{{color:var(--green)}}
 a{{color:var(--green)}} a:hover{{color:var(--green-dim)}} code{{font-family:var(--mono);color:var(--green)}}
 footer{{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--border);color:var(--text-muted);font-size:.85rem}}
 footer .wm{{font-family:var(--mono);font-weight:700;letter-spacing:1px;color:var(--text)}}
 footer .wm .dot{{color:var(--green)}}
</style></head><body><div class="grid-bg"></div><div class="wrap">
<header><div class="wordmark">SOUTH<span class="dot">.</span>BYTE</div>
<div class="tagline">AI Governance &amp; IT-Beratung</div></header>
<h1>Text-to-Image — Modellvergleich (DGX Spark / GB10)</h1>
{empty}
<h2>Metriken</h2>
<table><tr><th>Metrik</th>{head}</tr>{metrics}</table>
<h2>Galerie</h2>
<table><tr><th>Fall</th>{head}</tr>{"".join(gallery)}</table>
<footer><span class="wm">SOUTH<span class="dot">.</span>BYTE</span> — Michael van den Berg ·
<a href="https://southbyte.de">southbyte.de</a></footer>
</div></body></html>
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
