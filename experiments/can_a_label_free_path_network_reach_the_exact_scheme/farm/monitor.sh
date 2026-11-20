#!/bin/bash
# The state of every run of the experiment on one screen.
#
#   monitor.sh <store> <logs> [code]
#
# For each configuration in runs.txt: the key, the state, rounds done, minutes
# per round (from rounds.csv), the last and the best validation error, and
# the farm job status. Then the last five lines of the keeper's log. It reads
# and changes nothing.
set -u
STORE="$1"; LOGS="$2"; CODE="${3:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
EXPERIMENT=$(basename "$HERE")
PYTHON=/data/bfys/gscriven/conda/envs/TE/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH="$CODE/RK_Pinn_module/src"

QUEUE=$(condor_q "$USER" -constraint "RkpinnExperiment == \"$EXPERIMENT\"" \
        -af:j JobStatus RkpinnConfiguration 2>/dev/null)

printf "%-32s %-13s %-12s %6s %8s %10s %10s  %s\n" \
    configuration key state rounds min/rnd last_um best_um farm
while read -r configuration; do
    [ -z "$configuration" ] && continue
    key=$($PYTHON -m rkpinn.run_record.configuration "$HERE/configurations/$configuration" 2>/dev/null)
    run="$STORE/runs/$key"
    if [ ! -f "$run/state.json" ]; then
        printf "%-32s %-13s %-12s\n" "$configuration" "$key" "not made"; continue
    fi
    read -r state rounds minutes last best <<< "$($PYTHON - "$run" <<'EOF'
import json, sys, csv, os
run = sys.argv[1]
s = json.load(open(os.path.join(run, "state.json")))
rows = list(csv.DictReader(open(os.path.join(run, "rounds.csv")))) if os.path.exists(os.path.join(run, "rounds.csv")) else []
q = "validation_endpoint_error_larger_of_x_and_y_median_micrometres"
vals = [float(r[q]) for r in rows if r.get(q)]
mins = "-"
if rows and "seconds_of_training" in rows[-1]:
    # the columns the round trainer writes (round_trainer.py, row_of_round)
    last = rows[-5:]
    mins = "%.1f" % (sum(float(r["seconds_of_training"]) + float(r["seconds_of_validation"])
                         for r in last) / len(last) / 60)
print(s["state"].replace(" ", "_"), s["rounds_done"], mins,
      "%.3f" % vals[-1] if vals else "-", "%.3f" % min(vals) if vals else "-")
EOF
)"
    job=$(echo "$QUEUE" | awk -v c="$configuration" '$3 == c {print ($2 == 1 ? "idle" : $2 == 2 ? "running" : $2 == 5 ? "HELD" : $2)}' | head -1)
    printf "%-32s %-13s %-12s %6s %8s %10s %10s  %s\n" \
        "$configuration" "$key" "$state" "$rounds" "$minutes" "$last" "$best" "${job:-not queued}"
done < "$HERE/farm/runs.txt"

echo; echo "keeper.log, last five lines:"; tail -n 5 "$LOGS/keeper.log" 2>/dev/null
held=$(echo "$QUEUE" | awk '$2 == 5' | wc -l)
[ "$held" -gt 0 ] && echo "ATTENTION: $held job(s) held"
exit 0
