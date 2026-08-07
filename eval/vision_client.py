"""Minimaler OpenAI-kompatibler Vision-Chat-Client (nur stdlib).

Wird von den Metriken genutzt, um ein gespeichertes PNG an ein serviertes
multimodales Modell (OCR- oder Judge-VLM) zu schicken. Kein SDK-Zwang: das
Bild geht als data:-URI im image_url-Feld mit.
"""
from __future__ import annotations

import base64
import json
import urllib.request


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
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]
