#!/usr/bin/env python3
"""southbyte-image Orchestrator — config-getriebener Feldlauf über mehrere Modelle.

Generalisiert eval/field_run.sh: liest config/image_models.yaml, fährt jedes aktive
Modell nacheinander (Container via serving/run_image.sh mit dem passenden Loader),
generiert + bewertet das Testset (Treue via VLM-Judge, Textrendering via OCR) und
baut am Ende die Vergleichsseite. Läuft auf der Serving-Box (lokales Docker).

  python eval/orchestrate_images.py                      # alle aktiven Modelle
  python eval/orchestrate_images.py --models ERNIE-Image-Turbo,FLUX.1-schnell
  python eval/orchestrate_images.py --no-score           # nur generieren (Phase 1)

Judge/OCR über Env (wie score_all.sh):
  VISION_API_KEY, JUDGE_ENDPOINT (Default http://10.0.0.6:4000), JUDGE_MODEL
  (Default qwen/qwen3.7-plus), VISION_MAX_TOKENS.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PROFILES = Path(os.environ.get("SPARK_PROFILES_DIR",
                               Path.home() / "southbyte/southbyte-spark-profiles")) / "image"
HOST_PORT = os.environ.get("IMG_HOST_PORT", "8010")
CONTAINER = os.environ.get("IMG_CONTAINER", "southbyte-image")
JUDGE_ENDPOINT = os.environ.get("JUDGE_ENDPOINT", "http://10.0.0.6:4000")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen/qwen3.7-plus")


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def write_profile(m: dict, defaults: dict) -> None:
    """Schreibt image_profile.conf aus der Config (Config = Source of Truth)."""
    d = PROFILES / m["dir"]
    d.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# auto-generiert aus config/image_models.yaml — {m['name']}",
        f"PROFILE_LOADER={m.get('loader', 'auto')}",
        f"PROFILE_STEPS={m.get('steps', 20)}",
        f"PROFILE_GUIDANCE={m.get('guidance', 3.5)}",
        f"PROFILE_SIZE={m.get('size', defaults.get('size', '1024x1024'))}",
        f"PROFILE_DTYPE={m.get('dtype', defaults.get('dtype', 'bfloat16'))}",
    ]
    if m.get("components_dir"):
        lines.append(f"PROFILE_COMPONENTS_DIR={m['components_dir']}")
    if m.get("notes"):
        lines.append(f"PROFILE_NOTES={m['notes']!r}")
    (d / "image_profile.conf").write_text("\n".join(lines) + "\n", encoding="utf-8")


def wait_health(timeout: int = 900) -> bool:
    url = f"http://localhost:{HOST_PORT}/health"
    for _ in range(timeout // 5):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                body = r.read()  # 2026-08-11: EINMAL lesen — vorher 2x r.read() → 2. Read leer,
                if b'"status": "ok"' in body or b'"status":"ok"' in body:  # gesunder Server nie erkannt
                    return True
        except Exception:
            pass
        # Container tot?
        alive = subprocess.run(["docker", "ps", "--filter", f"name={CONTAINER}",
                                "--format", "{{.Names}}"], capture_output=True, text=True)
        if CONTAINER not in alive.stdout:
            return False
        time.sleep(5)
    return False


def run_model(m: dict, defaults: dict, score: bool) -> None:
    log(f"── {m['name']} ({m['dir']}, loader={m.get('loader','auto')}) ──")
    write_profile(m, defaults)
    env = {**os.environ, "MODEL_DIR": m["dir"], "HOST_PORT": HOST_PORT,
           "CONTAINER_NAME": CONTAINER, "IMAGE": m.get("image", defaults.get("image"))}
    r = subprocess.run(["bash", str(REPO / "serving/run_image.sh")], env=env)
    if r.returncode != 0:
        log(f"FEHLER Start {m['name']} (rc={r.returncode}) — übersprungen"); return
    if not wait_health():
        log(f"FEHLER {m['name']} nicht healthy — übersprungen")
        subprocess.run(["docker", "logs", "--tail", "8", CONTAINER])
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True); return

    cmd = [sys.executable, str(REPO / "eval/image_eval.py"),
           "--endpoint", f"http://localhost:{HOST_PORT}", "--model-name", m["name"]]
    if score:
        cmd += ["--judge-endpoint", JUDGE_ENDPOINT, "--judge-model", JUDGE_MODEL,
                "--ocr-endpoint", JUDGE_ENDPOINT, "--ocr-model", JUDGE_MODEL]
    subprocess.run(cmd)
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    log(f"{m['name']} fertig, Container gestoppt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config/image_models.yaml"))
    ap.add_argument("--models", help="Kommagetrennte Namen (sonst alle aktiven)")
    ap.add_argument("--no-score", action="store_true", help="nur generieren (Phase 1)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    defaults, models = cfg.get("defaults", {}), cfg.get("models", [])
    if args.models:
        want = {n.strip() for n in args.models.split(",")}
        models = [m for m in models if m["name"] in want]
    else:
        models = [m for m in models if m.get("active")]
    log(f"═══ Feldlauf: {len(models)} Modelle ═══")
    for m in models:
        run_model(m, defaults, score=not args.no_score)
    log("── Vergleichsseite bauen ──")
    subprocess.run([sys.executable, str(REPO / "eval/make_docs.py")])
    log("═══ Feldlauf Ende ═══")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
