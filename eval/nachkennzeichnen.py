#!/usr/bin/env python3
"""Fehlende EU-Kennzeichnung in vorhandenen Laufordnern nachziehen.

Seit 2026-09-30 kennzeichnet eval/image_eval.py direkt bei der Erzeugung und
legt die Lieferfassung unter img_gekennzeichnet/ ab. Laeufe VON VORHER haben
diesen Ordner nicht — der Seitenbau wuerde sie deshalb nicht veroeffentlichen.
Dieses Skript erzeugt die fehlenden Fassungen aus den vorhandenen Messbildern.

Die Messbilder unter img/ bleiben unberuehrt. Sie sind die Beweisebene; wer
sie kennzeichnet, macht die gemessenen Zahlen nachtraeglich ungueltig.

    python3 eval/nachkennzeichnen.py                 # alle Laeufe unter results/
    python3 eval/nachkennzeichnen.py results/<lauf>  # einzelner Lauf
    python3 eval/nachkennzeichnen.py --probe         # nur zeigen
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kennzeichnung import kennzeichnen  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("laeufe", nargs="*", type=Path,
                    help="Laufordner; ohne Angabe alle unter results/")
    ap.add_argument("--probe", action="store_true", help="nur zeigen, nichts schreiben")
    a = ap.parse_args()

    ordner = a.laeufe or sorted(p for p in (REPO / "results").glob("*_*") if (p / "img").is_dir())
    gesamt = fehlend = erzeugt = gescheitert = 0
    for lauf in ordner:
        quelle, ziel = lauf / "img", lauf / "img_gekennzeichnet"
        if not quelle.is_dir():
            continue
        offen = [p for p in sorted(quelle.glob("*.png")) if not (ziel / p.name).exists()]
        gesamt += len(list(quelle.glob("*.png")))
        fehlend += len(offen)
        if not offen:
            continue
        print(f"{lauf.name}: {len(offen)} fehlen")
        if a.probe:
            continue
        ziel.mkdir(parents=True, exist_ok=True)
        for p in offen:
            if kennzeichnen(p, ziel / p.name):
                erzeugt += 1
            else:
                gescheitert += 1
                print(f"  ✗ {p.name}")
    wort = "waeren zu erzeugen" if a.probe else "erzeugt"
    print(f"\n{gesamt} Messbilder geprueft, {fehlend} ohne Lieferfassung, {erzeugt} {wort}.")
    if gescheitert:
        print(f"✗ {gescheitert} gescheitert — Pillow oder Asset pruefen.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
