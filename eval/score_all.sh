#!/bin/bash
# Alle 4 Bildmodelle einheitlich mit qwen/qwen3.7-plus (via LiteLLM-Proxy) neu
# scoren: Adherence (Prompt-Treue) + OCR (Textrendering-CER). Dann Seite + Hub.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
IMG=/home/mvdb/southbyte/southbyte-image
RESULTS=/home/mvdb/southbyte/southbyte-results
TPENV=/home/mvdb/southbyte/southbyte-vllm/testplan/.env
CO='Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019YpS7jJGxYt9cD5nRMS7fH'
SL=$IMG/results/score_all.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$SL"; }
push(){ local repo="$1" msg="$2"; shift 2; cd "$repo" || return
  git add "$@" 2>/dev/null
  git commit -q -m "$msg

$CO" 2>/dev/null && log "commit: $msg" || log "nichts zu committen ($repo)"
  git pull --rebase origin main >/dev/null 2>&1 || true
  git push origin main 2>&1 | tail -1 | tee -a "$SL"; }

set -a; . "$TPENV" 2>/dev/null; set +a
export VISION_API_KEY="$JUDGE_API_KEY"
export VISION_MAX_TOKENS=2048
JE=http://10.0.0.6:4000
JM=qwen/qwen3.7-plus
cd "$IMG"
log "=== SCORE-ALL START (Judge+OCR=$JM) ==="
for m in FLUX.1-schnell Qwen-Image-2512 Qwen-Image-Flash ERNIE-Image-Turbo; do
  log "score $m ..."
  python3 eval/image_eval.py --model-name "$m" --score-only \
    --judge-endpoint "$JE" --judge-model "$JM" \
    --ocr-endpoint "$JE" --ocr-model "$JM" >> "$SL" 2>&1
  s=$(python3 -c "import json,glob;d=json.load(open(sorted(glob.glob('$IMG/results/*_$m/summary.json'))[-1]));print('Treue=%s CER=%s exakt=%s'%(d.get('adherence_score_mean'),d.get('text_rendering_cer_mean'),d.get('text_rendering_exact_rate')))" 2>/dev/null)
  log "  $m → $s"
done
log "Vergleichsseite bauen"
python3 eval/make_docs.py >> "$SL" 2>&1
push "$IMG" "Bild-Scores: Adherence+OCR einheitlich via qwen3.7-plus (alle 4)" docs eval/vision_client.py eval/score_all.sh
log "Hub aktualisieren"
python3 "$RESULTS/build_site.py" >> "$SL" 2>&1
push "$RESULTS" "Hub: Bild-Scores (qwen3.7-plus) 4 Modelle" build_site.py docs
log "SCORE_ALL_DONE"
log "=== SCORE-ALL ENDE ==="
