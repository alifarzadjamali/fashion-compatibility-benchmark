#!/usr/bin/env bash
# Phase 2b watchdog (deviation D2). Every 5 min, per python process of this user: if its CPU time
# has not increased for 20 min (and it is not SIGSTOPped for profiling, state T) -> stall:
# record in $F/STALLED; if it is a probe process, kill it (run_phase2b_cpu.sh retries the unit).
# Writes $F/HEARTBEAT each pass. Monitoring only; no science change.
F=${F:-/data/$USER/repbench/data/v2/full}
declare -A last idle
while true; do
  declare -A seen=(); n=0; stalled=0
  while read -r pid stat cpu cmd; do
    n=$((n+1)); seen[$pid]=1
    if [[ $stat == T* ]]; then idle[$pid]=0
    elif [ "${last[$pid]:--1}" = "$cpu" ]; then idle[$pid]=$(( ${idle[$pid]:-0} + 5 ))
    else idle[$pid]=0; fi
    last[$pid]=$cpu
    if [ ${idle[$pid]} -ge 20 ]; then
      stalled=$((stalled+1))
      echo "STALL $(date '+%F %T') pid=$pid idle=${idle[$pid]}min ${cmd:0:160}" | tee -a $F/STALLED
      [[ $cmd == *"pipeline.py --stage probes"* ]] && kill $pid && echo "killed probe pid $pid (unit will retry)" | tee -a $F/STALLED
      idle[$pid]=0
    fi
  done < <(ps -u $USER -o pid=,stat=,cputimes=,args= | grep "\.venv/bin/python" | grep -v grep)
  for p in "${!last[@]}"; do [ -n "${seen[$p]:-}" ] || { unset "last[$p]"; unset "idle[$p]"; }; done
  echo "$(date '+%F %T') python_procs=$n stalled_now=$stalled" > $F/HEARTBEAT
  unset seen; sleep 300
done
