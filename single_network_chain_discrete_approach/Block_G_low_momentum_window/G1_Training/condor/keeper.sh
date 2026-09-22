#!/bin/bash
# Block G farm keeper. Every 30 minutes it
#   * releases jobs the farm held: after the 24 h wall-time limit (the runs are
#     resumable, so they just go back), or for going over their memory request
#     (with the request doubled);
#   * takes out of condor/jobs_active.txt any run whose validation error has
#     stopped falling (condor/prune_active.py, which imports the rule);
#   * resubmits any network that has left the queue without finishing
#     (condor/resubmit.py, which reads condor/jobs_active.txt when it exists);
# and it stops once the queue is empty and nothing is left to resubmit.
#
# resubmit.py submits NOTHING when condor_q cannot be read, so a keeper that
# cannot see the queue waits rather than doubling a running job (Block F,
# 2026-09-21).
#
# Run it in tmux so it survives the editor closing:
#   tmux new-window -t claude-phone -n blockG-keeper <this script>
cd "$(dirname "$0")/.."
LOG=condor/keeper.log
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
echo "$(date '+%F %T') keeper started" >> $LOG
while true; do
  for id in $(condor_q gscriven -hold -af:j JobStatus 2>/dev/null | awk "{print \$1}"); do
    reason=$(condor_q "$id" -af HoldReason 2>/dev/null)
    if echo "$reason" | grep -qi "MaxWallTime\|exceeded the time"; then
      condor_release "$id" >/dev/null \
        && echo "$(date '+%F %T') released $id after the wall-time limit" >> $LOG
    elif echo "$reason" | grep -qi "memory"; then
      mem=$(condor_q "$id" -af RequestMemory 2>/dev/null); new=$(( mem * 2 ))
      condor_qedit "$id" RequestMemory "$new" >/dev/null && condor_release "$id" >/dev/null \
        && echo "$(date '+%F %T') released $id with $new MB" >> $LOG
    else
      echo "$(date '+%F %T') $id held for another reason: $reason" >> $LOG
    fi
  done
  # take out anything whose error has stopped falling before resubmitting
  PYTHONNOUSERSITE=1 $PY condor/prune_active.py --write >> $LOG 2>&1
  todo=$(PYTHONNOUSERSITE=1 $PY condor/resubmit.py --submit 2>&1 | grep "runs unfinished" | head -1)
  echo "$todo" >> $LOG
  queued=$(condor_q gscriven -af:j JobStatus 2>/dev/null | wc -l)
  left=$(grep -c . condor/jobs_active.txt 2>/dev/null || echo 0)
  echo "$(date '+%F %T') $left runs still to train, $queued in the queue" >> $LOG
  if [ "$queued" -eq 0 ] && echo "$todo" | grep -q "^0 runs"; then
    break
  fi
  sleep 1800
done
echo "$(date '+%F %T') queue empty and nothing to resubmit; keeper stopped" >> $LOG
