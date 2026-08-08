"""Objektive Textrendering-Metrik: OCR des Bildes gegen den erwarteten Text.

Für category=text_rendering. Ein serviertes OCR-/VLM-Modell (z.B. lokal
baidu--Unlimited-OCR oder ibm-granite--granite-docling-258M) transkribiert das
Bild; wir vergleichen per Character-Error-Rate gegen expected_text.
"""
from __future__ import annotations

import re

from vision_client import chat_with_image

_OCR_SYSTEM = (
    "Du bist eine strikt buchstabengetreue OCR-Engine. Gib GENAU die Zeichen "
    "zurück, die im Bild zu sehen sind — Buchstabe für Buchstabe, inklusive "
    "Tippfehler, verdrehter oder doppelter Buchstaben und fehlender Umlaute. "
    "KORRIGIERE NICHTS: rate nicht das gemeinte Wort, vervollständige nichts, "
    "verändere keine Schreibweise. Steht 'Grunwald' im Bild, antworte 'Grunwald', "
    "nicht 'Grünwald'. Nur der sichtbare Text, ohne Anführungszeichen, ohne "
    "Erklärung. Kein Text sichtbar → leere Zeile."
)


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _norm(s: str) -> str:
    """Robuster Vergleich: Groß/Klein + Mehrfach-Whitespace vereinheitlichen."""
    return re.sub(r"\s+", " ", s.strip().lower())


def cer(reference: str, hypothesis: str) -> float:
    """Character-Error-Rate (0.0 = perfekt) auf normalisierten Strings — Voll-String."""
    ref, hyp = _norm(reference), _norm(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def _min_substring_edit(needle: str, haystack: str) -> int:
    """Min. Edit-Distanz, um `needle` gegen IRGENDEINEN Teilstring von `haystack`
    zu matchen (fuzzy substring search). Row 0 = 0 → Start beliebig gratis;
    Antwort = min über die letzte Zeile → Ende beliebig."""
    if not needle:
        return 0
    if not haystack:
        return len(needle)
    prev = [0] * (len(haystack) + 1)          # leeres needle-Präfix matcht überall gratis
    for i, cn in enumerate(needle, 1):
        cur = [i]                              # needle-Präfix vs. leer = i Löschungen
        for j, ch in enumerate(haystack, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cn != ch)))
        prev = cur
    return min(prev)


def cer_contains(reference: str, hypothesis: str) -> float:
    """Containment-CER: ist der Soll-Text (annähernd) in der Transkription enthalten?
    Bestraft fehlende/falsche Soll-Zeichen, NICHT zusätzlich gerenderten/beschriebenen
    Text. Fair für Modelle, die mehr (korrekten) Text malen als der Minimal-Sollwert."""
    ref, hyp = _norm(reference), _norm(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _min_substring_edit(ref, hyp) / len(ref)


def score_text_rendering(case: dict, png_bytes: bytes, endpoint: str, model: str) -> dict:
    """OCR das Bild, vergleiche gegen case['expected_text']."""
    expected = case.get("expected_text", "")
    try:
        ocr_text = chat_with_image(endpoint, model, "Transkribiere den Text im Bild.",
                                   png_bytes, system=_OCR_SYSTEM, max_tokens=128)
    except Exception as e:  # noqa: BLE001 — Netz-/Serverfehler als Metrik-Fehler melden
        return {"expected": expected, "ocr": None, "cer": None, "exact": False, "error": str(e)}
    # Containment-CER als faire Hauptmetrik; Voll-String-CER als Referenz behalten.
    score = cer_contains(expected, ocr_text)
    return {"expected": expected, "ocr": ocr_text.strip(),
            "cer": round(score, 4), "cer_full": round(cer(expected, ocr_text), 4),
            "exact": cer_contains(expected, ocr_text) == 0.0}
