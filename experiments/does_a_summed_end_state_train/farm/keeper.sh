#!/bin/bash
# The keeper of the experiment's runs on the farm.
#
#   keeper.sh <code> <store> <logs> [once]
#
# It submits the runs of runs.txt and keeps them going until each has ended,
# plateaued or at its cap. Every 30 minutes it
#
#   * releases the jobs the farm holds after its time limit. The runs resume
#     from their last snapshot. A job held for going over its memory is
#     released with twice the memory. A job held for another reason is left
#     and written to the log;
#   * submits a run that is not in the queue and has not ended.
#
# Rules that keep two jobs from one run. They come from what went wrong on the
# frozen runs on 2026-09-21, when a second job was submitted for a run that
# was training:
#
#   1. a queue that cannot be read is unknown, not empty. Nothing is submitted;
#   2. a run whose lock was touched in the last 30 minutes has a job that is
#      alive, whatever the queue shows. It is not submitted.
#
# The package has its own guard under these: a run is locked, and a second
# job is refused.
#
# It stops when every run has ended. A run that ends at its cap without a
# plateau is not extended here: that is for George to rule on.
#
# Run it so that it outlives the session:
#   setsid nohup farm/keeper.sh <code> <store> <logs> > /dev/null 2>&1 &
set -u
CODE="$1"; STORE="$2"; LOGS="$3"; ONCE="${4:-}"
HERE="$CODE/experiments/does_a_summed_end_state_train"
EXPERIMENT=does_a_summed_end_state_train
PYTHON=/data/bfys/gscriven/conda/envs/TE/bin/python
MINUTES_A_LOCK_SAYS_ALIVE=30
SECONDS_BETWEEN_PASSES=1800
LOG="$LOGS/keeper.log"
mkdir -p "$LOGS"
export PYTHONNOUSERSITE=1 PYTHONPATH="$CODE/RK_Pinn_module/src"

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

state_of() {     # the state of a run, or "not made"
    local file="$STORE/runs/$1/state.json"
    if [ -f "$file" ]; then
        $PYTHON -c "import json,sys; s=json.load(open(sys.argv[1])); print(s['state'].replace(' ','_'), s['rounds_done'])" "$file"
    else
        echo "not_made 0"
    fi
}

lock_is_young() {   # true when the lock of a run was touched lately
    local lock="$STORE/runs/$1/LOCK"
    [ -f "$lock" ] && [ -n "$(find "$lock" -mmin -$MINUTES_A_LOCK_SAYS_ALIVE 2>/dev/null)" ]
}

say "keeper started: code $CODE, store $STORE"
while true; do
    # -- the queue, or nothing at all if it cannot be read
    if QUEUE=$(condor_q "$USER" -constraint "RkpinnExperiment == \"$EXPERIMENT\" && RkpinnStore == \"$STORE\"" \
               -af:j JobStatus RkpinnConfiguration 2>"$LOGS/keeper.condor_q.err") \
       && [ ! -s "$LOGS/keeper.condor_q.err" ]; then
        QUEUE_IS_KNOWN=yes
    else
        QUEUE_IS_KNOWN=no
        say "the queue cannot be read; nothing is submitted in this pass"
    fi

    # -- the jobs the farm holds (JobStatus 5)
    if [ "$QUEUE_IS_KNOWN" = yes ]; then
        echo "$QUEUE" | awk '$2 == 5 {print $1}' | while read -r id; do
            [ -z "$id" ] && continue
            reason=$(condor_q "$id" -af HoldReason 2>/dev/null)
            if echo "$reason" | grep -qi "MaxWallTime\|exceeded the time\|wall"; then
                condor_release "$id" > /dev/null && say "released $id after the time limit"
            elif echo "$reason" | grep -qi "memory"; then
                memory=$(condor_q "$id" -af RequestMemory 2>/dev/null)
                condor_qedit "$id" RequestMemory $(( memory * 2 )) > /dev/null \
                    && condor_release "$id" > /dev/null \
                    && say "released $id with $(( memory * 2 )) MB"
            else
                say "$id is held for another reason and is left: $reason"
            fi
        done
    fi

    # -- the runs
    LEFT=0
    while read -r configuration; do
        [ -z "$configuration" ] && continue
        key=$($PYTHON -m rkpinn.run_record.configuration "$HERE/configurations/$configuration")
        read -r state rounds <<< "$(state_of "$key")"
        case "$state" in
            plateaued|at_its_cap)
                say "$configuration ($key): ended, $state after $rounds rounds"; continue ;;
        esac
        LEFT=$(( LEFT + 1 ))
        if [ "$QUEUE_IS_KNOWN" = no ]; then continue; fi
        if echo "$QUEUE" | awk '{print $3}' | grep -qx "$configuration"; then
            say "$configuration ($key): in the queue, $state after $rounds rounds"
        elif lock_is_young "$key"; then
            say "$configuration ($key): not in the queue but its lock was touched; left"
        else
            condor_submit "$HERE/farm/jobs.sub" code="$CODE" store="$STORE" logs="$LOGS" \
                experiment="$EXPERIMENT" configurations="$HERE/configurations" \
                configuration="$configuration" -queue 1 >> "$LOG" 2>&1 \
                && say "$configuration ($key): submitted, $state after $rounds rounds"
        fi
    done < "$HERE/farm/runs.txt"

    if [ "$LEFT" -eq 0 ]; then
        say "every run has ended; keeper stopped"
        break
    fi
    [ -n "$ONCE" ] && { say "one pass was asked for; keeper stopped"; break; }
    sleep $SECONDS_BETWEEN_PASSES
done
