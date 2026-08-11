"""Prompt-Treue per VLM-as-Judge.

Ein serviertes multimodales Modell (z.B. lokal ein Gemma-4-Multimodal) bekommt
Bild + Original-Prompt + Prüfpunkte und bewertet, wie viele Kriterien erfüllt
sind. Rückgabe ist ein 0..1-Score plus Roh-Begründung.
"""
from __future__ import annotations

import json
import re

from vision_client import chat_with_image

_JUDGE_SYSTEM = (
    "Du bewertest, wie gut ein generiertes Bild zu einem Bild-Prompt passt. "
    "Sei streng und objektiv. Antworte AUSSCHLIESSLICH als JSON: "
    '{\"erfuellte_kriterien\": [..], \"fehlende_kriterien\": [..], '
    '\"score\": <0..5>, \"begruendung\": \"...\"}. '
    "score 5 = alle Kriterien klar erfüllt, 0 = Bild passt gar nicht."
)


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge_adherence(case: dict, png_bytes: bytes, endpoint: str, model: str) -> dict:
    """Bewerte Prompt-Treue; Score 0..1 (score/5)."""
    crit = case.get("criteria") or []
    user = (
        f"Prompt: {case['prompt']}\n"
        f"Prüfpunkte: {', '.join(crit) if crit else '(keine)'}\n"
        "Bewerte das Bild gegen den Prompt und die Prüfpunkte."
    )
    try:
        # 2026-08-11: 400→700 Tokens — bei langer begruendung wurde das JSON abgeschnitten
        # (kein schließendes } → _extract_json scheitert → score:null, z.B. img-018).
        raw = chat_with_image(endpoint, model, user, png_bytes,
                              system=_JUDGE_SYSTEM, max_tokens=700)
    except Exception as e:  # noqa: BLE001
        return {"score": None, "raw": None, "error": str(e)}
    parsed = _extract_json(raw)
    if not parsed or "score" not in parsed:
        # Fallback für trotzdem abgeschnittene Antworten: "score" steht vor der begruendung,
        # überlebt die Truncation → per Regex direkt aus dem raw ziehen.
        m = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw or "")
        if m:
            parsed = {"score": m.group(1)}
        else:
            return {"score": None, "raw": raw, "error": "unparsebare Judge-Antwort"}
    try:
        s5 = max(0.0, min(5.0, float(parsed["score"])))
    except (TypeError, ValueError):
        return {"score": None, "raw": raw, "error": "score nicht numerisch"}
    return {"score": round(s5 / 5.0, 3), "score_5": s5,
            "fehlend": parsed.get("fehlende_kriterien", []),
            "begruendung": parsed.get("begruendung", "")}
