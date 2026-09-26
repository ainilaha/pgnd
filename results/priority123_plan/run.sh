#!/usr/bin/env bash
# Remote only. Fixed 12-run seed-0 screen, staged for inspection; no search.
# Run from /root/pgnd: bash results/priority123_plan/run.sh p1 results/<new-run>
set -euo pipefail
if [[ $# != 2 || ! -f src/train.py ]]; then
  echo "From the repository root: bash $0 {p1|p2|p3} results/<new-run>" >&2
  exit 1
fi
STAGE=$1
RUN=$2
case "$STAGE" in p1|p2|p3) ;; *) echo "Choose p1, p2 or p3" >&2; exit 1 ;; esac
if [[ "$STAGE" == p1 ]]; then
  if [[ -e "$RUN" ]]; then echo "Choose a new RUN directory: $RUN" >&2; exit 1; fi
  mkdir -p "$RUN/source"
  cp src/*.py "$RUN/source/"
  cp "$0" "$RUN/run.sh"
  cp results/priority123_plan/protocol.md "$RUN/protocol.md"
else
  # Do not mix source revisions within this controlled screen.
  for source in src/*.py; do
    cmp -s "$source" "$RUN/source/${source##*/}" || {
      echo "Source mismatch or missing p1 snapshot: $source" >&2; exit 1;
    }
  done
fi

COMMON=(
  --train-pattern 'V(300|350|380)_Case[1-4]_CutFre20\.xls$'
  --validation-pattern 'V(300|350|380)_Case[5-6]_CutFre20\.xls$'
  --test-pattern 'V(300|350|380)_Case[7-8]_CutFre20\.xls$'
  --target-start 64 --train-stride 16 --cut-percent 0.1
  --epochs 100 --batch-size 128 --learning-rate 0.001 --seed 0
  --device cuda --threads 1 --preload-data --diagnostics
)
train() {
  local label=$1 model=$2 length=$3
  shift 3
  mkdir "$RUN/$label"
  python -m src.train --model "$model" --sequence-length "$length" "${COMMON[@]}" \
    "$@" --output "$RUN/$label/$label.pt" 2>&1 | tee "$RUN/$label/train.log"
}
require_run() {
  [[ -f "$RUN/$1/$1.pt" ]] || { echo "Complete the earlier stage: $1" >&2; exit 1; }
}
compare() {
  local name=$1
  shift
  local checkpoints=() label
  for label in "$@"; do checkpoints+=("$RUN/$label/$label.best.pt"); done
  python -m src.evaluate "${checkpoints[@]}" --split validation --diagnostics \
    --device cuda --threads 1 --preload-data --output-dir "$RUN/$name"
  python -m src.plot "$RUN/$name" --output-dir "$RUN/$name/figures"
}
contrasts() {
  # Reuse saved recording errors; no model execution or new scoring pipeline.
  python - "$RUN/$1" "${@:2}" <<'PY'
from pathlib import Path
import sys
import pandas as pd
from src.evaluate import paired_case_comparison
directory = Path(sys.argv[1])
records = pd.read_csv(directory / "per_recording.csv")
rows = []
for candidate, control in zip(sys.argv[2::2], sys.argv[3::2]):
    candidate, control = candidate + ".best", control + ".best"
    pair = records[records.run.isin([candidate, control])]
    rows.append(paired_case_comparison(pair.to_dict("records"), candidate))
pd.concat(rows, ignore_index=True).to_csv(directory / "primary_contrasts.csv", index=False)
PY
}

case "$STAGE" in
  p1)
    for length in 16 64; do
      for model in pgnd cnn_gru gru; do
        train "${model}_l${length}" "$model" "$length"
      done
    done
    train pgnd_delayed_l16 pgnd 16 --initialization delayed --warmup-steps 8
    train pgnd_history_l16 pgnd 16 --initialization history --warmup-steps 8
    compare context_validation pgnd_l16 pgnd_l64 cnn_gru_l16 cnn_gru_l64 gru_l16 gru_l64
    contrasts context_validation pgnd_l64 pgnd_l16 cnn_gru_l64 cnn_gru_l16 gru_l64 gru_l16
    compare initialization_validation pgnd_l16 pgnd_delayed_l16 pgnd_history_l16
    contrasts initialization_validation pgnd_history_l16 pgnd_delayed_l16 pgnd_delayed_l16 pgnd_l16
    ;;
  p2)
    require_run pgnd_l16
    train pgnd_no_residual_l16 pgnd 16 --no-residual
    train pgnd_unpenalized_l16 pgnd 16 --residual-weight 0
    compare residual_validation pgnd_l16 pgnd_no_residual_l16 pgnd_unpenalized_l16
    contrasts residual_validation pgnd_unpenalized_l16 pgnd_no_residual_l16 pgnd_l16 pgnd_unpenalized_l16
    ;;
  p3)
    require_run pgnd_no_residual_l16
    require_run pgnd_unpenalized_l16
    train second_order_l16 pgnd 16 --dynamics second_order
    train first_order_l16 pgnd 16 --dynamics first_order
    compare structure_validation pgnd_unpenalized_l16 pgnd_no_residual_l16 second_order_l16 first_order_l16
    contrasts structure_validation pgnd_unpenalized_l16 second_order_l16 second_order_l16 first_order_l16
    ;;
esac
echo "Completed $STAGE. Inspect validation metrics, probes and solver checks before the next stage."
