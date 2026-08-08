"""Minimaler OpenAI-kompatibler Vision-Chat-Client (nur stdlib).

Wird von den Metriken genutzt, um ein gespeichertes PNG an ein serviertes
multimodales Modell (OCR- oder Judge-VLM) zu schicken. Kein SDK-Zwang: das
Bild geht als data:-URI im image_url-Feld mit.
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.request

# Reasoning-Modelle (Qwen3.x) emittieren einen <think>…</think>-Vorspann; alles
# bis zum schließenden Tag wird verworfen, damit nur die eigentliche Antwort bleibt.
_THINK_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)


def chat_with_image(
    endpoint: str,
    model: str,
    user_text: str,
    png_bytes: bytes,
    *,
    system: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: int = 120,
) -> str:
    """Ein Vision-Chat-Turn; gibt den Text der Antwort zurück."""
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
    content = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    max_tokens = int(os.environ.get("VISION_MAX_TOKENS", max_tokens))
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("VISION_API_KEY", "")
    if api_key:
        # Cloud-Judge (z.B. claude-sonnet-5 via Proxy): Auth + schlankes Payload.
        # Keine vLLM-spezifischen Felder (chat_template_kwargs) — Cloud lehnt sie ab.
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        # Lokale vLLM-Judges (Qwen): Thinking aus + temperature, sonst verseucht
        # der Reasoning-Vorspann OCR-CER und JSON-Parsing.
        payload["temperature"] = temperature
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    content = body["choices"][0]["message"]["content"] or ""
    return _THINK_RE.sub("", content, count=1).strip()  # Reasoning-Vorspann abschneiden
