#!/bin/bash
# Part C smoke run: C1a and C2 on the frozen per-track straight-line network,
# a small draw (4,000 states, 5 restarts of 100 iterations, 3 rounds), then the
# standard report of each on the test split.
#
#   bash part_c_smoke.sh <store>
#
# The store is a scratch store that holds links to the track set 12a8d35c3165
# and to the run 25dfa17ef168; the runs and the reports of the smoke run are
# written there, not to the project store. The log goes to logs/.
set -eu
STORE="$1"
HERE=$(cd "$(dirname "$0")" && pwd)
MODULE="$HERE/../../../RK_Pinn_module"
PYTHON=/data/bfys/gscriven/conda/envs/TE/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH="$MODULE/src" OMP_NUM_THREADS=4
cd "$MODULE"
for name in c1a_learned_gain c2_error_conditioned; do
    configuration="$HERE/part_c_smoke_configurations/${name}__straight_line_per_track__smoke.yaml"
    key=$($PYTHON -m rkpinn.run_record.configuration "$configuration")
    echo "== $name, run $key, $(date '+%F %T')"
    $PYTHON -m rkpinn.training.round_trainer --store "$STORE" \
        --configuration "$configuration" --if-the-run-exists refuse
    $PYTHON -m rkpinn.run_record.report_of_a_run --store "$STORE" --run "$key" \
        --round last --split test --tracks-for-the-stage-errors 200 \
        --step-length-of-the-carrying-mm 1.0 --fraction-for-the-derivative 1e-3 \
        --states-for-the-loss-shares 4000 --bins-of-the-histograms 60
done
echo "== done $(date '+%F %T')"
