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
import re
import shutil
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_THUMB_PX = 360
_CONFIG = _ROOT / "config" / "image_models.yaml"
_MODELS_YAML = Path.home() / "southbyte/southbyte-vllm/testplan/config/models.yaml"


def _load_models() -> dict:
    """Release-Datum je Modell aus der zentralen models.yaml (name → release_date)."""
    out: dict = {}
    try:
        lines = _MODELS_YAML.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    cur = None
    for ln in lines:
        m = re.match(r'\s*-\s*name:\s*"?([^"#\n]+?)"?\s*$', ln)
        if m:
            cur = {}
            out[m.group(1).strip()] = cur
            continue
        f = re.match(r'\s+(release_date|license):\s*"?([^"#\n]+?)"?\s*$', ln)
        if f and cur is not None:
            cur[f.group(1)] = f.group(2).strip()
    return out


_MODELS = _load_models()


def _rel_cell(name: str) -> str:
    d = str(_MODELS.get(name or "", {}).get("release_date", "") or "")
    m = re.match(r"(\d{4})-(\d{2})", d)
    return f'<td data-sort="{m.group(1)}{m.group(2)}">{html.escape(d)}</td>' if m else '<td>—</td>'

# Klick-Sortierung (wie southbyte-vllm/results): Header klicken → tbody-Zeilen sortieren.
SORT_CSS = (
    " table th{cursor:pointer;user-select:none}"
    " table th::after{content:' ';opacity:.35;font-size:.75em}"
    " table th[aria-sort=ascending]::after{content:' \\25B2';opacity:.9}"
    " table th[aria-sort=descending]::after{content:' \\25BC';opacity:.9}"
    " table td.best{font-weight:700;color:var(--green);background:var(--bg-raised)}"
)
SORT_SCRIPT = """
<script>
(function(){
  function val(td){var s=td.getAttribute('data-sort');if(s===null){var el=td.querySelector('[data-sort]');if(el)s=el.getAttribute('data-sort');}return (s!==null?s:(td.textContent||'')).trim();}
  function num(t){var m=t.replace(/\\u00a0/g,'').replace(/\\s+/g,'').replace(',','.').match(/-?\\d+(?:\\.\\d+)?/);return m?parseFloat(m[0]):null;}
  function isEmpty(t){return t===''||t==='\\u2014'||t==='-';}
  function sortTable(table,idx,asc){
    var tb=table.tBodies[0]; if(!tb) return;
    var rows=Array.prototype.slice.call(tb.rows);
    var allNum=rows.every(function(r){var c=r.cells[idx];if(!c)return true;var v=val(c);return isEmpty(v)||num(v)!==null;});
    rows.sort(function(a,b){
      var av=a.cells[idx]?val(a.cells[idx]):'',bv=b.cells[idx]?val(b.cells[idx]):'';
      var e1=isEmpty(av),e2=isEmpty(bv);
      if(e1&&e2)return 0; if(e1)return 1; if(e2)return -1;
      var r=allNum?((num(av)||0)-(num(bv)||0)):av.localeCompare(bv,'de',{numeric:true});
      return asc?r:-r;
    });
    rows.forEach(function(r){tb.appendChild(r);});
  }
  document.querySelectorAll('table.sortable').forEach(function(table){
    var head=table.tHead; if(!head||!head.rows.length) return;
    Array.prototype.forEach.call(head.rows[0].cells,function(th,idx){
      th.setAttribute('title','Klick: sortieren');
      th.addEventListener('click',function(){
        var asc=th.getAttribute('aria-sort')!=='ascending';
        Array.prototype.forEach.call(head.rows[0].cells,function(o){o.removeAttribute('aria-sort');});
        th.setAttribute('aria-sort',asc?'ascending':'descending');
        sortTable(table,idx,asc);
      });
    });
  });
  // Bestwert je Spalte grün markieren (data-best=min|max am th); überlebt Sortierung.
  document.querySelectorAll('table.sortable').forEach(function(table){
    var head=table.tHead, tb=table.tBodies[0]; if(!head||!head.rows.length||!tb) return;
    Array.prototype.forEach.call(head.rows[0].cells,function(th,idx){
      var dir=th.getAttribute('data-best'); if(dir!=='min'&&dir!=='max') return;
      var best=null;
      Array.prototype.forEach.call(tb.rows,function(r){var c=r.cells[idx];if(!c)return;var v=num(val(c));if(v===null)return;if(best===null||(dir==='min'?v<best:v>best))best=v;});
      if(best===null)return;
      Array.prototype.forEach.call(tb.rows,function(r){var c=r.cells[idx];if(!c)return;var v=num(val(c));if(v!==null&&v===best)c.classList.add('best');});
    });
  });
})();
</script>
"""


def _load_dirs() -> dict:
    """name→hf-dir aus image_models.yaml (stdlib-Regex, kein yaml-Dependency)."""
    out: dict = {}
    try:
        lines = _CONFIG.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    name = None
    for ln in lines:
        g = re.match(r'\s*-\s*name:\s*"?([^"#\n]+?)"?\s*$', ln)
        if g:
            name = g.group(1).strip()
            continue
        d = re.match(r'\s*dir:\s*"?([^"#\s]+)"?', ln)
        if d and name:
            out[name] = d.group(1)
            name = None
    return out


def _card_url(hf_dir: str) -> str:
    """owner--model → https://huggingface.co/owner/model."""
    return "https://huggingface.co/" + hf_dir.replace("--", "/", 1)


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

    dirs = _load_dirs()

    def _mlabel(mdl: str) -> str:
        d = dirs.get(mdl)
        return (f'<a href="{_card_url(d)}" target="_blank" rel="noopener">{html.escape(mdl)}</a>'
                if d else html.escape(mdl))

    # Galerie-Header: Modelle als Spalten, Name → Model-Card verlinkt
    head = "".join(f"<th>{_mlabel(r['summary']['model'])}</th>" for r in runs)

    # Metrik-Übersicht: Modelle als ZEILEN (sortierbare Spalten), Name → Card
    def _mcell(v, suffix="") -> str:
        return f'<td data-sort="{"" if v is None else v}">{_fmt(v, suffix)}</td>'

    def _mrow(r: dict) -> str:
        s = r["summary"]; mdl = s["model"]
        return ("<tr>"
                f'<td data-sort="{html.escape(mdl)}" class="mname">{_mlabel(mdl)}</td>'
                + _rel_cell(mdl)
                + _mcell(s.get("adherence_score_mean"))
                + _mcell(s.get("text_rendering_cer_mean"))
                + _mcell(s.get("text_rendering_exact_rate"))
                + _mcell(s.get("gen_seconds_mean"), " s")
                + _mcell(s.get("generated"))
                + "</tr>")
    metrics_rows = "\n".join(_mrow(r) for r in runs)

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
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMiAzMiIgcm9sZT0iaW1nIiBhcmlhLWxhYmVsPSJTb3V0aEJ5dGUiPgogIDx0aXRsZT5Tb3V0aEJ5dGU8L3RpdGxlPgogIDxyZWN0IHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgZmlsbD0iIzA2MEMwQSIvPgogIDx0ZXh0IHg9IjIiIHk9IjIzIgogICAgICAgIGZvbnQtZmFtaWx5PSInQ291cmllciBOZXcnLCBDb25zb2xhcywgJ1NGIE1vbm8nLCBtb25vc3BhY2UiCiAgICAgICAgZm9udC1zaXplPSIxNiIKICAgICAgICBmb250LXdlaWdodD0iNzAwIgogICAgICAgIGxldHRlci1zcGFjaW5nPSIwLjUiPgogICAgPHRzcGFuIGZpbGw9IiNENEVERTAiPlM8L3RzcGFuPjx0c3BhbiBmaWxsPSIjMDBFNjc2Ij4uPC90c3Bhbj48dHNwYW4gZmlsbD0iI0Q0RURFMCI+QjwvdHNwYW4+CiAgPC90ZXh0PgogIDxyZWN0IHg9IjIiIHk9IjI2IiB3aWR0aD0iMjgiIGhlaWdodD0iMS41IiBmaWxsPSIjMDBFNjc2IiBvcGFjaXR5PSIwLjQiLz4KPC9zdmc+Cg==">
<style>
 :root{{--bg:#060C0A;--bg-raised:#0A1410;--bg-card:#0E1A14;--border:#162A1E;--border-hi:#1A5C38;
   --green:#00E676;--green-dim:#00994A;--text:#D4EDE0;--text-muted:#5E8A72;
   --mono:'Courier New',Consolas,'SF Mono',Menlo,monospace;--sans:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);line-height:1.7}}
 .grid-bg{{position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.5;
   background-image:linear-gradient(rgba(0,230,118,.15) 1px,transparent 1px),
     linear-gradient(90deg,rgba(0,230,118,.15) 1px,transparent 1px);background-size:80px 80px}}
 .wrap{{position:relative;z-index:1;max-width:min(1720px,97vw);margin:0 auto;padding:2.5rem 1.25rem}}
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
 table.metrics{{width:auto;max-width:100%;margin:1rem auto}}
 table.metrics th.mname,table.metrics td.mname{{text-align:left;white-space:nowrap}}
 .scroll{{overflow-x:auto;margin:1rem 0}}
 .hint{{color:var(--text-muted);font-size:.78rem;margin:.1rem 0 0}}
 {SORT_CSS}
 @keyframes scanline{{0%{{transform:translateY(-100vh)}}100%{{transform:translateY(100vh)}}}}
 .scanline{{position:fixed;left:0;top:0;width:100%;height:80px;background:linear-gradient(to bottom,transparent,rgba(0,230,118,.03) 40%,rgba(0,230,118,.07) 50%,rgba(0,230,118,.03) 60%,transparent);pointer-events:none;z-index:0;animation:scanline 8s linear infinite;will-change:transform}}
 @media(prefers-reduced-motion:reduce){{.scanline{{display:none}}}}
</style></head><body><div class="grid-bg"></div><div class="scanline"></div><div class="wrap">
<header><div class="wordmark">SOUTH<span class="dot">.</span>BYTE</div>
<div class="tagline">AI Governance &amp; IT-Beratung</div></header>
<h1>Text-to-Image — Modellvergleich (DGX Spark / GB10)</h1>
{empty}
<h2>Metriken</h2>
<div class="scroll"><table class="metrics sortable"><thead><tr>
<th class="mname">Modell</th><th>Release</th><th data-best="max">Prompt-Treue</th><th data-best="min">Text-CER</th><th data-best="max">Text exakt</th><th data-best="min">Ø Zeit/Bild</th><th>Bilder</th>
</tr></thead><tbody>{metrics_rows}</tbody></table></div>
<p class="hint">Spaltenüberschrift klicken zum Sortieren · Modellname → Model-Card</p>
<h2>Galerie</h2>
<div class="scroll"><table><tr><th>Fall</th>{head}</tr>{"".join(gallery)}</table></div>
<footer><span class="wm">SOUTH<span class="dot">.</span>BYTE</span> — Michael van den Berg ·
<a href="https://southbyte.de">southbyte.de</a></footer>
</div>{SORT_SCRIPT}</body></html>
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
