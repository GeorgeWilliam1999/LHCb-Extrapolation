#!/bin/bash
# Block E farm keeper. Every 30 minutes it
#   * releases jobs the farm held: after the 24 h wall-time limit (the runs are
#     resumable, so they just go back), or for going over their memory request
#     (with the request doubled);
#   * resubmits any network that has left the queue without finishing
#     (`resubmit.py`, which reads condor/jobs_active.txt when it exists);
# and it stops once the queue is empty and nothing is left to resubmit.
#
# Run it in tmux so it survives the editor closing:
#   tmux new-window -t claude-phone -n farm-keeper <this script>
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
  # take out anything that is finished on BOTH rules before resubmitting
  (cd /data/bfys/gscriven/LHCb_Extrapolation_Project/single_network_chain_discrete_approach/Block_F_reweighted_loss/F2_Analysis && PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python prune_active.py --block E --write >> "$OLDPWD/$LOG" 2>&1)
  todo=$(PYTHONNOUSERSITE=1 $PY resubmit.py --submit 2>&1 | head -1)
  echo "$todo" >> $LOG
  queued=$(condor_q gscriven -af:j JobStatus 2>/dev/null | wc -l)
  done_runs=$(grep -l '"phase": "done"' results/N*_q*/progress.json 2>/dev/null | wc -l)
  echo "$(date '+%F %T') $done_runs of 16 at their cap, $queued in the queue, exact $(ls ../E2_Comparators/results/exact_N*_q*.json 2>/dev/null | wc -l)/8" >> $LOG
  if [ "$queued" -eq 0 ] && echo "$todo" | grep -q "^0 networks"; then
    break
  fi
  sleep 1800
done
echo "$(date '+%F %T') queue empty and nothing to resubmit; keeper stopped" >> $LOG
