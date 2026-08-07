"""Objektive Textrendering-Metrik: OCR des Bildes gegen den erwarteten Text.

Für category=text_rendering. Ein serviertes OCR-/VLM-Modell (z.B. lokal
baidu--Unlimited-OCR oder ibm-granite--granite-docling-258M) transkribiert das
Bild; wir vergleichen per Character-Error-Rate gegen expected_text.
"""
from __future__ import annotations

import re

from vision_client import chat_with_image

_OCR_SYSTEM = (
    "Du bist eine OCR-Engine. Gib ausschließlich den im Bild sichtbaren Text "
    "wörtlich zurück, ohne Anführungszeichen, ohne Erklärung. Wenn kein Text "
    "zu sehen ist, antworte mit einer leeren Zeile."
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
    """Character-Error-Rate (0.0 = perfekt) auf normalisierten Strings."""
    ref, hyp = _norm(reference), _norm(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def score_text_rendering(case: dict, png_bytes: bytes, endpoint: str, model: str) -> dict:
    """OCR das Bild, vergleiche gegen case['expected_text']."""
    expected = case.get("expected_text", "")
    try:
        ocr_text = chat_with_image(endpoint, model, "Transkribiere den Text im Bild.",
                                   png_bytes, system=_OCR_SYSTEM, max_tokens=128)
    except Exception as e:  # noqa: BLE001 — Netz-/Serverfehler als Metrik-Fehler melden
        return {"expected": expected, "ocr": None, "cer": None, "exact": False, "error": str(e)}
    score = cer(expected, ocr_text)
    return {"expected": expected, "ocr": ocr_text.strip(), "cer": round(score, 4),
            "exact": _norm(expected) == _norm(ocr_text)}
