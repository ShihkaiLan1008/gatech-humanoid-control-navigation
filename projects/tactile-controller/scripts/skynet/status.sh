#!/usr/bin/env bash
#
# What is free on Skynet, and what am I holding?
#
#     scripts/skynet/status.sh              # everything
#     scripts/skynet/status.sh rtx_6000     # only that GPU type
#     scripts/skynet/status.sh -m           # only my jobs (fast)
#
# The whole payload runs in ONE ssh call via a quoted heredoc into `bash -l -s`,
# so nothing is expanded locally and Slurm is on the PATH remotely (its binaries
# live in /opt/slurm/... which only a LOGIN shell picks up).
#
# The numbers only count nodes in idle/mixed/allocated state.  Without that
# filter the totals look great and none of it is schedulable -- a large share of
# Skynet's GPUs sit on drained/down/inval nodes at any given time.

set -euo pipefail

REMOTE_HOST=${TACNAV_HOST:-skynet}
FILTER=""
MINE_ONLY=0

for a in "$@"; do
  case "$a" in
    -m|--mine) MINE_ONLY=1 ;;
    -h|--help)
      sed -n '3,9p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    -*) echo "status.sh: unknown option '$a'" >&2; exit 2 ;;
    *)  FILTER=$a ;;
  esac
done

# stderr is kept: swallowing it turned an ssh auth failure into a silent empty
# report that reads as "no jobs running" -- exactly the wrong answer under the
# 2-concurrent-job quota.
if ! ssh -o BatchMode=yes -o ConnectTimeout=25 "$REMOTE_HOST" true 2>/dev/null; then
  echo "status.sh: cannot reach $REMOTE_HOST -- NOT a report of 'no jobs'." >&2
  ssh -o BatchMode=yes -o ConnectTimeout=25 "$REMOTE_HOST" true
  exit 1
fi

ssh -o BatchMode=yes "$REMOTE_HOST" "bash -l -s '$FILTER' '$MINE_ONLY'" <<'REMOTE'
FILTER=$1
MINE_ONLY=$2
ME=$(whoami)

# ---------------------------------------------------------------- my jobs
echo "=============================== MY JOBS ==============================="
if squeue -h -u "$ME" -o %i | grep -q .; then
  squeue -u "$ME" -O JobID:9,Name:18,Partition:10,StateCompact:4,TimeUsed:9,TimeLimit:11,NodeList:12,tres-alloc:40
  echo
  # How many GPUs am I actually holding?  overcap caps a user at 48.
  squeue -h -u "$ME" -t R -O tres-alloc:200,NodeList:20 | awk '
    { n=0; if (match($0, /gres\/gpu=[0-9]+/)) n=substr($0, RSTART+9, RLENGTH-9)
      total += n; nodes[$NF] += n }
    END { printf "  holding %d GPU(s) across %d node(s)", total, length(nodes)
          for (k in nodes) printf "   [%s x%d]", k, nodes[k]
          print "   (overcap per-user cap: 48)" }'
else
  echo "  (no jobs queued or running)"
fi

[ "$MINE_ONLY" = "1" ] && exit 0

# ---------------------------------------------------------------- free GPUs
# sort -u -k1,1 dedupes: sinfo -N repeats a node once per partition it is in.
SNAP=$(sinfo -h -N -O NodeList:20,Gres:32,GresUsed:32,StateLong:16 | sort -u -k1,1)

echo
echo "============================= FREE GPUs ==============================="
printf '%s\n' "$SNAP" | awk -v f="$FILTER" '
  $2 ~ /^gpu:/ {
    split($2,g,":"); split(g[3],a,"("); split($3,u,":"); split(u[3],b,"(")
    t=g[2]; if (f != "" && t != f) next
    if ($4 ~ /^(idle|mixed|alloc)/) { tot[t]+=a[1]; used[t]+=b[1]; nn[t]++ }
    else                             { down[t]+=a[1] }
  }
  END {
    printf "%-11s %6s %7s %7s %7s %8s %11s\n","TYPE","NODES","TOTAL","USED","FREE","UTIL","UNUSABLE"
    for (t in tot)
      printf "%-11s %6d %7d %7d %7d %7.0f%% %11d\n", t, nn[t], tot[t], used[t], tot[t]-used[t],
             (tot[t] ? 100*used[t]/tot[t] : 0), down[t]
  }' | { read -r h; echo "$h"; sort -k5 -nr; }

echo
echo "======================= NODES WITH A FREE GPU ========================="
printf '%s\n' "$SNAP" | awk -v f="$FILTER" '
  $2 ~ /^gpu:/ && $4 ~ /^(idle|mixed|alloc)/ {
    split($2,g,":"); split(g[3],a,"("); split($3,u,":"); split(u[3],b,"(")
    t=g[2]; if (f != "" && t != f) next
    free=a[1]-b[1]
    if (free > 0) printf "  %-9s %-15s free %d/%-3d  %s\n", t, $1, free, a[1], $4
  }' | sort -k1,1 -k4,4r | head -40
REMOTE
