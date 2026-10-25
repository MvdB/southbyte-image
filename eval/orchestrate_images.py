#!/usr/bin/env python3
"""southbyte-image Orchestrator — config-getriebener Feldlauf über mehrere Modelle.

Generalisiert eval/field_run.sh: liest config/image_models.yaml, fährt jedes aktive
Modell nacheinander (Container via serving/run_image.sh mit dem passenden Loader),
generiert + bewertet das Testset (Treue via VLM-Judge, Textrendering via OCR) und
baut am Ende die Vergleichsseite. Läuft auf der Serving-Box (lokales Docker).

  python eval/orchestrate_images.py                      # alle aktiven Modelle
  python eval/orchestrate_images.py --models ERNIE-Image-Turbo,FLUX.1-schnell
  python eval/orchestrate_images.py --no-score           # nur generieren (Phase 1)

Judge und OCR laufen NICHT lokal — sie brauchen einen OpenAI-kompatiblen
Endpunkt. Konfiguriert wird er über die Umgebung oder eine `.env` neben dieser
Datei im Repo-Wurzelverzeichnis (`.env.example` als Vorlage, `.env` ist
gitignoriert):

  JUDGE_ENDPOINT     Basis-URL des Judge. Kein Standardwert — bewusst.
  JUDGE_MODEL        Modell dort (Standard qwen/qwen3.7-plus)
  VISION_API_KEY     Schlüssel dafür
  VISION_MAX_TOKENS  optional

Ohne JUDGE_ENDPOINT bricht nur die Bewertung ab; `--no-score` läuft weiter.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def env_datei_laden(pfad: Path) -> None:
    """Liest eine `.env` nach os.environ — ohne bereits Gesetztes zu überschreiben.

    Warum überhaupt: Die Adresse des Judge ist Betriebswissen und stand vorher als
    Standardwert im Code — in einem oeffentlichen Repository also eine private
    Netzadresse für jeden lesbar, und für jeden, der klont, ein Vorgabewert, der
    ins Leere zeigt. Die Datei ist gitignoriert; `.env.example` zeigt die Form.

    Bewusst kein python-dotenv: Das Repo kommt mit der Standardbibliothek aus, und
    eine Abhaengigkeit für zwoelf Zeilen lohnt nicht. Entsprechend eng ist der
    Parser — KEY=WERT je Zeile, `#` als Kommentar, umschliessende Anfuehrungszeichen
    werden entfernt. Kein Zeilenumbruch im Wert, keine Ersetzung von Variablen.
    """
    if not pfad.is_file():
        return
    for zeile in pfad.read_text(encoding="utf-8").splitlines():
        zeile = zeile.strip()
        if not zeile or zeile.startswith("#") or "=" not in zeile:
            continue
        schluessel, _, wert = zeile.partition("=")
        schluessel = schluessel.strip().removeprefix("export ").strip()
        os.environ.setdefault(schluessel, wert.strip().strip("'\""))


env_datei_laden(REPO / ".env")

PROFILES = Path(os.environ.get("SPARK_PROFILES_DIR",
                               Path.home() / "southbyte/southbyte-spark-profiles")) / "image"
HOST_PORT = os.environ.get("IMG_HOST_PORT", "8010")
CONTAINER = os.environ.get("IMG_CONTAINER", "southbyte-image")
# Kein Standardwert: Der Judge steht auf einer anderen Maschine, und welche das
# ist, gehoert nicht in ein oeffentliches Repository. Fehlt der Wert, scheitert
# nur die Bewertung — und zwar mit einer Ansage, nicht mit einer Zeitueberschreitung.
JUDGE_ENDPOINT = os.environ.get("JUDGE_ENDPOINT", "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen/qwen3.7-plus")


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def log_sichern(container: str, name: str) -> None:
    """Vollen Log eines fehlgestarteten Containers sichern.

    Der Container wird danach entfernt, und die letzten Zeilen allein zeigen die
    Ursache selten — bei Ming-Image am 25.10.2026 stand dort nur die Folgemeldung
    des Orchestrators. Ablage unter results/_fehlstarts/ (nicht im Repo).
    """
    ziel = REPO / "results" / "_fehlstarts" / f"{time.strftime('%Y-%m-%d_%H%M')}_{name}_{container}.log"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    with ziel.open("wb") as f:
        subprocess.run(["docker", "logs", container], stdout=f, stderr=subprocess.STDOUT)
    log(f"Voller Log: {ziel.relative_to(REPO)}")


def write_profile(m: dict, defaults: dict) -> None:
    """Schreibt image_profile.conf aus der Config (Config = Source of Truth)."""
    d = PROFILES / m["dir"]
    d.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# auto-generiert aus config/image_models.yaml — {m['name']}",
        f"PROFILE_LOADER={m.get('loader', 'auto')}",
        f"PROFILE_STEPS={m.get('steps', 20)}",
        # None/"none": die Pipeline behaelt ihre eigene Guidance (ideogram-4: Plan je Schritt).
        f"PROFILE_GUIDANCE={'none' if m.get('guidance', 3.5) in (None, 'none') else m.get('guidance', 3.5)}",
        f"PROFILE_SIZE={m.get('size', defaults.get('size', '1024x1024'))}",
        f"PROFILE_DTYPE={m.get('dtype', defaults.get('dtype', 'bfloat16'))}",
    ]
    if m.get("components_dir"):
        lines.append(f"PROFILE_COMPONENTS_DIR={m['components_dir']}")
    # Nur FLUX.3 und ideogram-4 kennen den Schalter (Loader flux3_modular, ideogram4);
    # ohne Eintrag in der Registry entscheidet der Loader, und der laesst ihn an.
    if "prompt_upsampler" in m:
        lines.append(f"PROFILE_PROMPT_UPSAMPLER={1 if m['prompt_upsampler'] else 0}")
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


def lauf_leer(m: dict) -> bool:
    """True, wenn der juengste Lauf dieses Modells kein einziges Bild hat.

    Am 27.09.2026 scheiterten alle 22 FLUX.2-Faelle mit HTTP 500 (das v2-Image
    trug eine alte server_image.py), und der Orchestrator endete trotzdem mit
    rc=0 — der Fehllauf sah von aussen aus wie ein Erfolg. Ein Lauf ohne ein
    einziges Bild ist keiner.
    """
    kandidaten = sorted((REPO / "results").glob(f"*_{m['name']}"))
    if not kandidaten:
        return True
    sj = kandidaten[-1] / "summary.json"
    if not sj.exists():
        return True
    try:
        return int(json.loads(sj.read_text(encoding="utf-8")).get("generated", 0)) == 0
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return True


def bereit(url: str, container: str, timeout: int = 1200) -> bool:
    """Antwortet die URL mit 200? Bricht ab, sobald der Container nicht mehr läuft."""
    for _ in range(timeout // 5):
        try:
            with urllib.request.urlopen(url, timeout=5):
                return True
        except Exception:
            pass
        alive = subprocess.run(["docker", "ps", "--filter", f"name={container}", "--format", "{{.Names}}"],
                               capture_output=True, text=True)
        if container not in alive.stdout.split():
            return False
        time.sleep(5)
    return False


def run_model_omni(m: dict, score: bool) -> None:
    """Bildmodell über vLLM-Omni nach seinem Rezept, Upsampler als eigener Dienst."""
    hoch = m.get("upsampler") or {}
    log(f"── {m['name']} ({m['dir']}, vLLM-Omni{', Upsampler ' + hoch['dir'] if hoch else ''}) ──")
    env = {**os.environ, "MODEL_DIR": m["dir"], "HOST_PORT": HOST_PORT, "CONTAINER_NAME": CONTAINER,
           "IMAGE": m["image"], "SERVED_NAME": m["name"], "PUP_DIR": hoch.get("dir", ""),
           # Zusatzargumente fuer vllm serve, etwa die Deploy-Config von Ming-Image.
           "OMNI_ARGS": json.dumps(m.get("omni_args") or []),
           # Eintraege des Modellordners, die vLLM-Omni nicht sehen soll (run_omni.sh).
           "MODEL_AUSBLENDEN": " ".join(m.get("ausblenden") or [])}
    if hoch.get("image"):
        env["PUP_IMAGE"] = hoch["image"]
    try:
        if subprocess.run(["bash", str(REPO / "serving/run_omni.sh")], env=env).returncode != 0:
            log(f"FEHLER Start {m['name']} — übersprungen"); return
        if not bereit(f"http://localhost:{HOST_PORT}/v1/models", CONTAINER):
            log(f"FEHLER {m['name']} nicht bereit — übersprungen")
            log_sichern(CONTAINER, m["name"])
            subprocess.run(["docker", "logs", "--tail", "15", CONTAINER]); return
        testset = m.get("testset", "testset/image_de_v2.jsonl")
        cmd = [sys.executable, str(REPO / "eval/image_eval.py"),
               "--endpoint", f"http://localhost:{HOST_PORT}", "--model-name", m["name"],
               "--testset", str(REPO / testset)]
        if m.get("size"):
            cmd += ["--size", m["size"]]
        if m.get("negative_prompt") is False:
            cmd += ["--ohne-negativ"]
        if m.get("api") == "chat":  # Ming-Image: Bilder ueber chat/completions (Rezept)
            cmd += ["--api", "chat"]
            for feld, flag in (("steps", "--steps"), ("guidance", "--guidance")):
                if m.get(feld) is not None:
                    cmd += [flag, str(m[feld])]
        if hoch:
            modelle = Path(os.environ.get("HF_MODELS_DIR", Path.home() / "hf_models"))
            cmd += ["--upsampler", f"http://localhost:{os.environ.get('PUP_PORT', '8014')}",
                    "--upsampler-model", hoch["dir"],
                    "--upsampler-system", str(modelle / hoch["dir"] / "system_prompts.json")]
        if score:
            cmd += ["--judge-endpoint", JUDGE_ENDPOINT, "--judge-model", JUDGE_MODEL,
                    "--ocr-endpoint", JUDGE_ENDPOINT, "--ocr-model", JUDGE_MODEL]
        subprocess.run(cmd)
    finally:
        for c in (CONTAINER, f"{CONTAINER}-pup"):
            subprocess.run(["docker", "rm", "-f", c], capture_output=True)
    log(f"{m['name']} fertig, Container gestoppt")


REFINE_PORT = os.environ.get("REFINE_PORT", "8014")


def refiner_vorab(m: dict, testset: str) -> Path | None:
    """Prompt-Refiner VOR dem Bildmodell: starten, alle Faelle umschreiben, stoppen.

    Fuer Modelle, deren Refiner nicht neben dem Bildmodell in den Speicher passt
    (HiDream-O1 mit Prompt-Refine, ~59 GB). Umgeschrieben wird im Image des
    Bildmodells, weil dort der Prompt-Agent der Referenz liegt
    (eval/prompts_vorab.py). Gibt die JSONL-Datei zurueck oder None bei Fehler.
    """
    r = m["refiner"]
    name = f"{CONTAINER}-refine"
    modelle = Path(os.environ.get("HF_MODELS_DIR", Path.home() / "hf_models"))
    ziel = REPO / "results" / "_vorab" / f"{time.strftime('%Y-%m-%d_%H%M')}_{m['name']}.jsonl"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    log(f"Refiner {r['dir']} startet ({r['image']}, :{REFINE_PORT})")
    start = ["docker", "run", "-d", "--name", name, "--gpus", "all", "--ipc=host",
             "-p", f"{REFINE_PORT}:8000", "-v", f"{modelle}:/hf_models:ro", "-e", "HF_HUB_OFFLINE=1",
             r["image"], "--model", f"/hf_models/{r['dir']}", "--served-model-name", r["dir"],
             "--gpu-memory-utilization", str(r.get("gpu_anteil", 0.8)), *[str(a) for a in r.get("args", [])]]
    try:
        if subprocess.run(start, capture_output=True).returncode != 0:
            log("FEHLER Refiner-Start"); return None
        if not bereit(f"http://localhost:{REFINE_PORT}/v1/models", name):
            log("FEHLER Refiner nicht bereit")
            log_sichern(name, "refiner")
            subprocess.run(["docker", "logs", "--tail", "15", name]); return None
        lauf = subprocess.run(
            ["docker", "run", "--rm", "--network", "host", "--user", f"{os.getuid()}:{os.getgid()}",
             "-v", f"{REPO / 'testset'}:/testset:ro", "-v", f"{ziel.parent}:/out",
             m["image"], "python", "prompts_vorab.py",
             "--testset", f"/testset/{Path(testset).name}", "--base-url", f"http://localhost:{REFINE_PORT}/v1",
             "--model", r["dir"], "--out", f"/out/{ziel.name}"])
        if lauf.returncode != 0 or not ziel.exists():
            log("FEHLER Vorab-Umschreiben (alle Faelle Rueckfall oder Abbruch)"); return None
        return ziel
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        log("Refiner gestoppt")


def run_model(m: dict, defaults: dict, score: bool) -> None:
    if m.get("backend") == "omni":
        return run_model_omni(m, score)
    log(f"── {m['name']} ({m['dir']}, loader={m.get('loader','auto')}) ──")
    testset = m.get("testset", defaults.get("testset", "testset/image_de_v1.jsonl"))
    vorab = None
    if m.get("refiner"):
        vorab = refiner_vorab(m, testset)
        if vorab is None:
            log(f"FEHLER {m['name']}: ohne Refiner kein Lauf — übersprungen"); return
    write_profile(m, defaults)
    env = {**os.environ, "MODEL_DIR": m["dir"], "HOST_PORT": HOST_PORT,
           "CONTAINER_NAME": CONTAINER, "IMAGE": m.get("image", defaults.get("image"))}
    r = subprocess.run(["bash", str(REPO / "serving/run_image.sh")], env=env)
    if r.returncode != 0:
        log(f"FEHLER Start {m['name']} (rc={r.returncode}) — übersprungen"); return
    if not wait_health():
        log(f"FEHLER {m['name']} nicht healthy — übersprungen")
        log_sichern(CONTAINER, m["name"])
        subprocess.run(["docker", "logs", "--tail", "8", CONTAINER])
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True); return

    cmd = [sys.executable, str(REPO / "eval/image_eval.py"),
           "--endpoint", f"http://localhost:{HOST_PORT}", "--model-name", m["name"],
           "--testset", str(REPO / testset)]
    if vorab:
        cmd += ["--vorab", str(vorab)]
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

    # Vor dem ersten Modell pruefen, nicht nach dem letzten: Ein Feldlauf ueber
    # sechs Modelle laeuft Stunden, und die Bewertung kommt je Modell ganz am
    # Ende. Ohne diesen Riegel faellt eine fehlende Konfiguration erst auf,
    # nachdem die erste GPU-Stunde verbrannt ist.
    if not args.no_score and not JUDGE_ENDPOINT:
        print(
            "JUDGE_ENDPOINT ist nicht gesetzt — die Bewertung braucht einen\n"
            "OpenAI-kompatiblen Endpunkt und laeuft nicht lokal.\n\n"
            f"  cp {REPO}/.env.example {REPO}/.env   und dort eintragen\n"
            "  oder: JUDGE_ENDPOINT=... python eval/orchestrate_images.py\n\n"
            "Nur erzeugen, ohne zu bewerten: --no-score",
            file=sys.stderr,
        )
        return 2

    log(f"═══ Feldlauf: {len(models)} Modelle ═══")
    leer = []
    for m in models:
        run_model(m, defaults, score=not args.no_score)
        if lauf_leer(m):
            log(f"FEHLER {m['name']}: kein einziges Bild erzeugt")
            leer.append(m["name"])
    log("── Vergleichsseite bauen ──")
    subprocess.run([sys.executable, str(REPO / "eval/make_docs.py")])
    log("═══ Feldlauf Ende ═══")
    if leer:
        print(f"Ohne Ergebnis geblieben: {', '.join(leer)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
