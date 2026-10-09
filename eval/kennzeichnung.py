"""EU-Kennzeichnung fuer KI-erzeugte Bilder — eine Stelle, zwei Aufrufer.

Icon der Europaeischen Kommission (Variante "AI GENERATED") aus
assets/eu-ki-kennzeichnung/. Quelle:
digital-strategy.ec.europa.eu/en/policies/eu-icons-labelling-ai-generated-content

WANN gekennzeichnet wird: direkt bei der Erzeugung (eval/image_eval.py), nicht
erst beim Veroeffentlichen. Ein Bild ist damit von Anfang an gekennzeichnet und
kann gar nicht ungekennzeichnet irgendwo landen.

WAS NICHT gekennzeichnet wird: die Messbilder unter img/. Unsere Bewertung
laeuft ueber ein Bildmodell als Judge und misst gerenderten Text per OCR — ein
eingebranntes "AI GENERATED" liefe in die Text-CER und in die Prompt-Treue.
Deshalb zwei Dateien je Bild: img/ ist die Messgrundlage, img_gekennzeichnet/
der Liefergegenstand. Der Seitenbau kopiert nur noch.

Unten links, weil unsere Szenen das Motiv meist mittig bis oben tragen und
Provenienzangaben konventionell unten gesucht werden. Schwarz oder weiss
entscheidet die Helligkeit der Zielflaeche — beide Fassungen liefert die
Kommission mit.
"""
from __future__ import annotations

from pathlib import Path

_KENNZEICHEN_DIR = Path(__file__).resolve().parents[1] / "assets" / "eu-ki-kennzeichnung"
_KENNZEICHEN_BREITE = 0.34   # Anteil der Bildbreite
_KENNZEICHEN_RAND = 0.025    # Abstand zum Bildrand, Anteil der Bildbreite


def _kennzeichen_fuer(bild, x: int, y: int, breite: int, hoehe: int) -> Path:
    """Schwarze oder weisse Fassung — je nachdem, worauf das Label zu liegen kommt.

    Die Kommission liefert beide Fassungen genau dafuer mit. Eine selbstgebaute
    Hinterlegung waere ein Eingriff in fremdes Gestaltungsmaterial und sah in
    der ersten Fassung auch so aus (grauer Kasten mit harten Kanten). Statt
    dessen: die Helligkeit der Zielflaeche messen und die passende Fassung
    nehmen — heller Hintergrund bekommt die schwarze Pille, dunkler die weisse.
    """
    ausschnitt = bild.convert("L").crop((x, y, x + breite, y + hoehe))
    mittel = sum(ausschnitt.getdata()) / max(len(ausschnitt.getdata()), 1)
    fassung = "black" if mittel > 127 else "white"
    return _KENNZEICHEN_DIR / f"ai-generated_{fassung}.png"


def kennzeichnen(src: Path, dst: Path) -> bool:
    """Bild mit der EU-Kennzeichnung nach dst schreiben.

    Rueckgabe False, wenn es nicht ging (Pillow fehlt, Asset fehlt). Der
    Aufrufer muss das melden — eine ungekennzeichnete Auslieferung soll
    auffallen, nicht stillschweigend passieren.
    """
    try:
        from PIL import Image
    except ImportError:
        return False
    with Image.open(src) as roh:
        bild = roh.convert("RGBA")
    breite = max(int(bild.width * _KENNZEICHEN_BREITE), 96)
    rand = int(bild.width * _KENNZEICHEN_RAND)
    # Hoehe aus dem Seitenverhaeltnis des Assets (3,16:1), vor dem Laden gebraucht
    with Image.open(_KENNZEICHEN_DIR / "ai-generated_black.png") as muster:
        hoehe = max(round(muster.height * breite / muster.width), 1)
    x, y = rand, bild.height - hoehe - rand
    pfad = _kennzeichen_fuer(bild, x, y, breite, hoehe)
    if not pfad.exists():
        return False
    with Image.open(pfad) as ikon_roh:
        ikon = ikon_roh.convert("RGBA").resize((breite, hoehe), Image.LANCZOS)
    bild.alpha_composite(ikon, (x, y))
    bild.convert("RGB").save(dst, "PNG", optimize=True)
    return True


