#!/usr/bin/env bash
#
# Submit a tactile_controller training run to the Skynet Slurm cluster.
#
#     scripts/skynet/submit.sh [options] <training command>
#
# Everything from the first non-option argument onward is the training command,
# forwarded to the compute node verbatim -- so the command you type here is the
# same one you would type locally:
#
#     scripts/skynet/submit.sh \
#       uv run python scripts/rl/train.py Mjlab-Decoupled-Joint-Vel-VForce-MPC-G1 \
#         --env.scene.num_envs 4096
#
# The command is base64'd through ssh -> sbatch, so its own --flags, quotes and
# spaces survive intact and never collide with this script's options.
#
# WHAT IT DOES
#   1. rsyncs src/ scripts/ pyproject.toml uv.lock to $REMOTE_REPO  (--no-rsync to skip)
#      plus, on first use, data/ to $TACNAV_DATA (shared, symlinked in -- see 2)
#   2. ALSO rsyncs a private copy of src/ to a fresh job-src/ dir -- this job's
#      code is then immune to a later `submit.sh` resyncing REMOTE_REPO while
#      this one is still queued (see the "isolation" comment before CMD_B64)
#   3. writes an sbatch script that sets up uv / warp caches on flash
#   4. submits it and prints the job id, log path and follow command
#
# Slurm is not on the PATH of a non-login shell on Skynet, hence the `bash -lc`
# wrapper on every remote call.  W&B needs no setup here: mjlab already defaults
# to `logger="wandb"`, and ~/.netrc is on NFS home so every node sees the key.

set -euo pipefail

# ------------------------------------------------------------------ location
REMOTE_HOST=${TACNAV_HOST:-skynet}
REMOTE_ROOT=${TACNAV_ROOT:-/coc/flash12/jjang318/tacnav}
REMOTE_REPO=${TACNAV_REPO:-$REMOTE_ROOT/tactile_controller}
# Shared dataset root: lives OUTSIDE the repo so the --delete sync cannot
# touch it, and is symlinked into REMOTE_REPO and each job-src (see below).
DATA_DIR=${TACNAV_DATA:-$REMOTE_ROOT/data}
SLURM_LOG_DIR=$REMOTE_ROOT/slurm-logs
LOCAL_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# ------------------------------------------------------------------ defaults
# overcap/overcap is the preemptible scavenger tier: priority 0, GPUs capped at
# 48 per user, 2-day wall.  It costs the lab nothing and disturbs nobody.
#
# THESE DEFAULTS ARE THE ONLY SANCTIONED CONFIGURATION (2026-08-26).  The lab
# allows exactly one submit shape -- these defaults plus `-c 2 -t 12:00:00` --
# and at most 2 concurrent jobs, because CPU/GPU quotas are shared lab-wide.
# -p/-A/--qos/-g still exist as flags but are NOT ours to use; wu-lab in
# particular is off limits.  See docs/skynet_cluster.md.
PARTITION=overcap
ACCOUNT=overcap
QOS=""
GPU_TYPE=rtx_6000       # sm_75, 24 GB, and the least-contended type on Skynet
GPUS=1
CPUS=""                 # unset -> DEFAULT_CPUS_PER_GPU below (2026-08-17 measurement)
MEM=32G

# Measured (rtx_6000, 4096 envs, settled steady-state over 20 iterations):
#   -c 7 (node CPU/GPU ratio): 6.700 s/iter
#   -c 2                     : 7.111 s/iter  (+6.1%)
# A real but small cost -- judged an acceptable default given it leaves far
# more headroom on wu-lab's shared GrpTRES pool (cpu=96 shared with gres/gpu:
# a40=16 -- CPUs there are rationed the same as GPUs, not just a courtesy).
DEFAULT_CPUS_PER_GPU=2

# The node-level "golden spot": node_CPUs / node_GPUs for each type, measured
# off sinfo.  No longer the default (see above) -- kept as the ceiling for the
# over-request WARNING below.  Asking for MORE than this strands GPUs on the
# node for other users (not enough CPUs left to pair with them), regardless of
# what a good default for your OWN performance is.
cpus_per_gpu() {
  case "$1" in
    l40s)              echo 32 ;;
    a40)               echo 16 ;;
    a5000|2080_ti)     echo  8 ;;
    rtx_6000|titan_x)  echo  7 ;;
    *)                 echo  8 ;;
  esac
}
TIME=8:00:00
JOB_NAME=""
RUN_NAME=""
STAMP_FMT=${TACNAV_STAMP_FMT:-%m-%d-%H-%M}   # Month-Date-Hour-Minute
CKPT_ENDPOINT=dropbox   # wandb | dropbox | local -- see -e/--ckpt-endpoint below
# W&B entity every run is pinned to.  Without it runs follow the API key's
# account-level default entity, which is NOT the team the project lives in and
# can be changed from the web UI at any time -- silently scattering runs into a
# fresh project.  Exported as BOTH WANDB_ENTITY and WANDB_USERNAME: rsl_rl reads
# WANDB_USERNAME and passes it as wandb.init(entity=...) -- WANDB_ENTITY alone
# does not reach it.  Empty string disables the pin.
WANDB_ENTITY_PIN=${TACNAV_WANDB_ENTITY-tacnav-2026}
# Who submitted, for the `<who>-cluster` W&B tag.  Cluster runs are AUTHORED by
# whoever owns the node's ~/.netrc W&B key -- the shared cluster account, not
# the person who typed the command -- so W&B's own "Created by" cannot separate
# us.  train.py emits the matching `<who>-local` for local runs.  Empty string
# disables the tag.
WHO_TAG=${TACNAV_WHO-$(id -un)}
DROPBOX_PATH="TacNav/tactile_controller/runs"
KEEP_LOCAL=0
SYNC_DATA=0
DO_RSYNC=1
MODE=sbatch
WANDB_OFFLINE=0
FOLLOW=0
DRY_RUN=0

usage() {
  cat <<'EOF'
usage: scripts/skynet/submit.sh [options] <training command>

Resource options
  -g, --gpu TYPE       GPU type          (default: rtx_6000)
                       a40 l40s rtx_6000 2080_ti a5000 titan_x
  -G, --gpus N         GPUs per job      (default: 1)
  -c, --cpus N         CPUs per task     (default: 2 x --gpus.  Measured on
                       rtx_6000 @ 4096 envs: -c 2 is 7.111s/iter settled vs
                       6.700s at the node's CPU/GPU ratio (+6.1%) -- judged a
                       fine tradeoff, and it leaves more of wu-lab's shared
                       CPU quota for labmates.  Going ABOVE the node ratio
                       (l40s 32, a40 16, a5000/2080_ti 8, rtx_6000/titan_x 7,
                       per --gpus) strands GPUs for other users; warns if you
                       explicitly ask for more than that.)
  -m, --mem SIZE       host RAM          (default: 32G)
  -t, --time HH:MM:SS  wall limit        (default: 8:00:00)
  -p, --partition P    partition         (default: overcap)
  -A, --account A      account           (default: overcap)
      --qos Q          QOS (short=2d, long=7d; needed on lab partitions)
  -J, --name NAME      job name          (default: derived from the task id)

Run naming
  -r, --run-name NAME  passed as --agent.run-name NAME_MM-DD-HH-MM.  The stamp
                       is generated at submit time, so the W&B display name and
                       the log directory both carry it.  Override the format
                       with TACNAV_STAMP_FMT (date(1) syntax).
                       Ignored if you pass --agent.run-name yourself.

Where checkpoints end up
  -e, --ckpt-endpoint E  wandb | dropbox | local   (default: dropbox)
                       dropbox  checkpoints stay OFF W&B entirely
                                (--agent.upload-model False) and the whole run
                                directory is rclone'd to dropbox:PATH after
                                training.  Once `rclone check` confirms an
                                exact match, local model_*.pt files are
                                deleted; everything else stays local.
                                Needs rclone on the cluster; see NOTES below.
                       wandb    checkpoints upload to W&B during training
                                instead (mjlab's own default behaviour).
                                After the run, this script verifies each
                                checkpoint landed there (size-checked via the
                                W&B API) and, once verified, DELETES the
                                local model_*.pt files to free flash.
                                tfevents/config/git-diff.txt are untouched
                                (wandb never stores those).
                       local    checkpoints are NOT uploaded anywhere and
                                NEVER auto-deleted (there would be no other
                                copy) -- everything just sits under
                                logs/rsl_rl/... on flash.
                       In all three cases, deletion only ever touches
                       model_*.pt / model_*.onnx, and only after a verified
                       transfer -- a failed or skipped verification always
                       leaves the local files in place.
      --dropbox-path PATH  destination for --ckpt-endpoint dropbox
                       (default: TacNav/tactile_controller/runs)
      --who NAME       submitter name for the 'NAME-cluster' W&B tag
                       (default: your local username, or $TACNAV_WHO).
                       Local runs get 'NAME-local' from train.py, so one tag
                       family covers both.  Cluster runs are authored by the
                       shared cluster account, making this the only
                       per-person dimension that works in both places.
      --wandb-entity E  W&B team/user every run is filed under
                       (default: tacnav-2026, or $TACNAV_WANDB_ENTITY).
                       Pass '' to fall back to whatever the API key's account
                       currently defaults to -- which is how runs end up
                       scattered into a project nobody is watching.
      --keep-local     never delete local checkpoints, whatever the endpoint
                       (still uploads/verifies -- just skips the rm step)

Run naming
  -r, --run-name NAME  passed as --agent.run-name NAME_MM-DD-HH-MM.  The stamp
                       is generated at submit time, so the W&B display name and
                       the log directory both carry it.  Override the format
                       with TACNAV_STAMP_FMT (date(1) syntax).
                       Ignored if you pass --agent.run-name yourself.

Behaviour options
      --sync-data      re-upload the WHOLE local data/ tree to $TACNAV_DATA even
                       if it is already there.  data/ is tens of GB; it uploads
                       automatically only while $TACNAV_DATA is still empty, and
                       is then symlinked into each job -- src/ syncs per submit,
                       the datasets do not.  To add ONE dataset, rsync that
                       subdirectory to $TACNAV_DATA by hand instead.
      --no-rsync       do not sync the local tree first
      --offline        WANDB_MODE=offline; sync later with `wandb sync`.
                       Combined with --ckpt-endpoint wandb, auto-prune is
                       skipped (nothing has actually uploaded yet) -- run
                       `wandb sync` yourself, then delete the .pt files by hand.
  -i, --interactive    run with srun in the foreground instead of sbatch
  -f, --follow         tail the job log until the job leaves the queue
  -n, --dry-run        print the generated sbatch script and exit
  -h, --help           this message

Examples
  # quick test: 1x rtx_6000 on the preemptible tier, checkpoints to Dropbox
  # (the default) + auto-prune once the transfer is verified
  scripts/skynet/submit.sh -r smoke -f \
    uv run python scripts/rl/train.py Mjlab-Decoupled-Joint-Vel-VForce-MPC-G1 \
      --env.scene.num_envs 1024 --agent.max-iterations 50

  # THE sanctioned form: 12h on the default rtx_6000, Dropbox checkpoints.
  # Vary only -r, the task id and --env.scene.num_envs.
  scripts/skynet/submit.sh -c 2 -t 12:00:00 -r vforce_v1 \
    uv run python scripts/rl/train.py Mjlab-Decoupled-Joint-Vel-VForce-MPC-G1 \
      --env.scene.num_envs 4096

  # same, but to a different Dropbox folder than the TacNav default
  scripts/skynet/submit.sh -g a40 \
    --dropbox-path TacNav/tactile_controller/sweeps/lr_1e-4 -r sweep_lr1e4 \
    uv run python scripts/rl/train.py Mjlab-Decoupled-Joint-Vel-VForce-MPC-G1 \
      --env.scene.num_envs 4096

  # use W&B's own checkpoint storage instead of Dropbox
  scripts/skynet/submit.sh -e wandb -r vforce_wandb -f \
    uv run python scripts/rl/train.py Mjlab-Decoupled-Joint-Vel-VForce-MPC-G1 \
      --env.scene.num_envs 4096

  # local debugging: interactive srun, nothing uploaded or deleted anywhere
  scripts/skynet/submit.sh --ckpt-endpoint local -i \
    uv run python scripts/rl/train.py Mjlab-Decoupled-Joint-Vel-VForce-MPC-G1 \
      --env.scene.num_envs 256 --agent.max-iterations 5

If a run is preempted, resume it straight from its W&B checkpoints (only
works with --ckpt-endpoint wandb, and only before this script's auto-prune
has deleted the local copies -- the W&B copy itself is untouched by pruning):
    --agent.resume True --wandb-run-path <entity>/<project>/<run_id>

NOTES -- one-time Dropbox setup (rclone is NOT installed on Skynet):
    ssh skynet
    curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip
    unzip -j /tmp/rc.zip '*/rclone' -d /coc/flash12/jjang318/tacnav/bin
    chmod +x /coc/flash12/jjang318/tacnav/bin/rclone
    /coc/flash12/jjang318/tacnav/bin/rclone config
                                        # n) new remote -> name it "dropbox" -> dropbox
                                        # answer "n" to auto-config, then paste the
                                        # token from `rclone authorize "dropbox"`
                                        # run on your workstation (login node has no browser)
    (/coc/flash12/jjang318/tacnav is this script's default TACNAV_ROOT. If you
    invoke submit.sh with a custom TACNAV_ROOT env var, use that same path here
    instead -- the two must match or the job won't find the rclone binary.)
The binary and its rclone.conf live on flash/NFS home, so every compute node
sees them -- no per-job setup.

NOTES -- git provenance: every submit captures the LOCAL repo's branch, commit,
and full uncommitted diff at submit time (before rsync).  It is printed on the
node (slurm log, first 80 lines), written to <run_dir>/git-diff.txt (so it
travels with --ckpt-endpoint dropbox), and attached to the W&B run as tags
(branch-*, commit-*, dirty|clean) and as WANDB_NOTES.  A run's provenance is
recoverable even after you've since committed or edited locally.

NOTES -- source isolation: each submit (unless --no-rsync) also gets its own
~80 MB copy of src/ under $REMOTE_ROOT/job-src/, so editing code and
submitting again while an earlier job is still QUEUED can never make that
earlier job silently train the newer code -- it keeps importing from its own
private copy no matter what REMOTE_REPO looks like by the time it starts.
The shared venv is still reused (no per-job rebuild): uv is pointed at the
isolated pyproject.toml/uv.lock via UV_PROJECT but told never to sync
(UV_NO_SYNC=1) and to use the existing venv (VIRTUAL_ENV + --active, spliced
into the command automatically when it starts with `uv run`); PYTHONPATH
makes python resolve imports from the isolated copy ahead of the shared one.
Only src/ is isolated -- scripts/rl/train.py itself still runs from
REMOTE_REPO (it changes far less often; a deliberate simplification, not an
oversight).  Logs/wandb/checkpoints are unaffected -- CWD stays REMOTE_REPO,
so nothing about where run output lands has changed.
These job-src/ copies are cheap and left in place (flash has terabytes free)
-- they double as an audit trail of exactly what ran.  To reclaim space later:
    ssh skynet 'find /coc/flash12/jjang318/tacnav/job-src -maxdepth 1 -mtime +30 -exec rm -rf {} +'
EOF
}

# ------------------------------------------------------------------ parse
# Options must precede the training command.  The loop stops at `--` or at the
# first bare word, and everything left is the command.
while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpu)        GPU_TYPE=$2; shift 2 ;;
    -G|--gpus)       GPUS=$2; shift 2 ;;
    -c|--cpus)       CPUS=$2; shift 2 ;;
    -m|--mem)        MEM=$2; shift 2 ;;
    -t|--time)       TIME=$2; shift 2 ;;
    -p|--partition)  PARTITION=$2; shift 2 ;;
    -A|--account)    ACCOUNT=$2; shift 2 ;;
    --qos)           QOS=$2; shift 2 ;;
    -J|--name)       JOB_NAME=$2; shift 2 ;;
    -r|--run-name)   RUN_NAME=$2; shift 2 ;;
    -e|--ckpt-endpoint) CKPT_ENDPOINT=$2; shift 2 ;;
    --dropbox-path)  DROPBOX_PATH=$2; shift 2 ;;
    --wandb-entity)  WANDB_ENTITY_PIN=$2; shift 2 ;;
    --who)           WHO_TAG=$2; shift 2 ;;
    --keep-local)    KEEP_LOCAL=1; shift ;;
    --sync-data)     SYNC_DATA=1; shift ;;
    --no-rsync)      DO_RSYNC=0; shift ;;
    --offline)       WANDB_OFFLINE=1; shift ;;
    -i|--interactive) MODE=srun; shift ;;
    -f|--follow)     FOLLOW=1; shift ;;
    -n|--dry-run)    DRY_RUN=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    --)              shift; break ;;
    -*)              echo "submit.sh: unknown option '$1'" >&2
                     echo "(script options must come BEFORE the training command)" >&2
                     exit 2 ;;
    *)               break ;;
  esac
done

case "$CKPT_ENDPOINT" in
  wandb|dropbox|local) ;;
  *) echo "submit.sh: --ckpt-endpoint must be wandb, dropbox, or local (got '$CKPT_ENDPOINT')" >&2; exit 2 ;;
esac

if [[ $# -eq 0 ]]; then
  echo "submit.sh: no training command given" >&2; echo >&2; usage >&2; exit 2
fi
CMD=( "$@" )

# Default to the flat, measured-sufficient count unless -c was given
# explicitly; warn (rather than silently clamp) if an explicit -c would strand
# GPUs on the node for other users -- that ceiling is still the node ratio.
GOLDEN=$(cpus_per_gpu "$GPU_TYPE")
if [[ -z $CPUS ]]; then
  CPUS=$(( DEFAULT_CPUS_PER_GPU * GPUS ))
elif (( CPUS > GOLDEN * GPUS )); then
  echo "submit.sh: warning -- ${CPUS} CPUs for ${GPUS}x${GPU_TYPE} exceeds the" >&2
  echo "           $((GOLDEN * GPUS)) that node's CPU/GPU ratio allows; this strands GPUs" >&2
  echo "           on the node for other users. Proceeding as asked." >&2
fi

# --agent.upload-model is a real mjlab field (mjlab/rl/config.py), so this is a
# config change rather than an env-var hack -- metrics still go to W&B either
# way.  Only the "wandb" endpoint wants mjlab's own upload path; dropbox/local
# keep checkpoints off W&B and this script (or nothing, for local) handles them.
# Passed explicitly in BOTH directions: train.py defaults it False, so `-e wandb`
# has to opt back in, and a reader here should not have to know that default.
if [[ $CKPT_ENDPOINT == wandb ]]; then
  CMD+=( --agent.upload-model True )
else
  CMD+=( --agent.upload-model False )
fi

# Run name + timestamp.  Stamped here rather than in the job so that the name is
# known at submit time and identical across a requeue.  An explicit
# --agent.run-name in the training command always wins.
if [[ -n $RUN_NAME ]]; then
  if printf '%s\n' "${CMD[@]}" | grep -qx -- '--agent.run-name'; then
    echo "submit.sh: --run-name ignored, the training command sets --agent.run-name" >&2
  else
    CMD+=( --agent.run-name "${RUN_NAME}_$(date +"$STAMP_FMT")" )
  fi
fi

# The registered task id, if the command names one.  Only used for the job name --
# the W&B `task-*` / `exp-*` / `where-*` tags are train.py's (run_identity_tags),
# which sees them on the node too.
TASK_ID=""
for a in "${CMD[@]}"; do
  if [[ $a == Mjlab-* || $a == Unitree-* ]]; then TASK_ID=$a; break; fi
done

# Job name defaults to the task id with the Mjlab- prefix stripped.
if [[ -z $JOB_NAME ]]; then
  JOB_NAME=${TASK_ID#Mjlab-}; JOB_NAME=${JOB_NAME#Unitree-}
  JOB_NAME=${JOB_NAME:-tacnav}
fi
JOB_NAME=${JOB_NAME//[^A-Za-z0-9_.-]/_}

# ------------------------------------------------------------------ isolation
# The problem this solves: REMOTE_REPO is one shared, mutable checkout.  If you
# edit code and resubmit while an earlier job is still QUEUED (not yet
# started -- very possible when a GPU type is busy), that job's rsync
# overwrites the tree the earlier job will read once it finally starts, and it
# silently trains the newer code -- while its git-diff.txt snapshot (captured
# at ITS OWN submit time) describes something that never actually ran.
#
# Fix: give every submission its own tiny (~80 MB) copy of src/scripts, and
# point uv at ITS pyproject.toml/uv.lock via UV_PROJECT while still using the
# shared venv (VIRTUAL_ENV + UV_NO_SYNC=1 -- verified empirically: uv ignores
# a bare VIRTUAL_ENV without --active, and skips syncing only with --no-sync,
# both confirmed against the shared 12 GB venv).  PYTHONPATH points python at
# the isolated src/ ahead of the venv's own editable-installed path, so it
# wins for anything actually imported (verified: an isolated copy of a
# package that already exists via the editable install correctly shadows it).
# UV_PROJECT does NOT change the job's CWD (verified), so logs/wandb/checkpoints
# still land under REMOTE_REPO exactly as before -- only the imported *code*
# is isolated, not where its output goes.  --active has no env-var mirror, so
# it's spliced into the command below when it starts with `uv run`.
#
# Only src/ is isolated this way -- scripts/rl/train.py itself still resolves
# from the shared tree (its own path is what the user typed, relative to
# REMOTE_REPO).  scripts/ changes far less often than src/ in practice; this
# is a deliberate, documented gap, not an oversight.
#
# Skipped entirely under --no-rsync: if you're not syncing, there is nothing
# fresh to isolate, and PYTHONPATH would just point at an empty directory.
ISOLATED_DIR="$REMOTE_ROOT/job-src/${JOB_NAME}-$(date +%Y%m%d-%H%M%S)-$$"
if [[ $DO_RSYNC -eq 1 ]]; then
  if [[ ${CMD[0]:-} == uv && ${CMD[1]:-} == run ]]; then
    CMD=( "${CMD[0]}" "${CMD[1]}" --active "${CMD[@]:2}" )
  fi
fi

# printf %q makes the command re-parseable; base64 gets it through ssh and the
# sbatch heredoc without a single layer of quoting hazard.
CMD_B64=$(printf '%q ' "${CMD[@]}" | base64 -w0)

# ------------------------------------------------------------------ git snapshot
# What code is this job actually running?  The remote tree has no .git (it's a
# plain rsync mirror), so without this a job's provenance is unrecoverable the
# moment you commit or edit locally.  This is captured NOW -- the state rsync
# is about to push -- and travels three ways: printed on the node (slurm log),
# written into the run directory (so it rides along to Dropbox), and attached
# to the W&B run as tags + notes.  These git tags are submit.sh's alone: the node
# has no .git, so train.py's own run_identity_tags() cannot produce them there.
GIT_BRANCH=unknown; GIT_SHA=unknown; GIT_DIRTY=unknown
GIT_INFO_FILE=$(mktemp)
if git -C "$LOCAL_REPO" rev-parse --git-dir >/dev/null 2>&1; then
  GIT_BRANCH=$(git -C "$LOCAL_REPO" rev-parse --abbrev-ref HEAD)
  GIT_SHA=$(git -C "$LOCAL_REPO" rev-parse --short HEAD)
  GIT_SHA_FULL=$(git -C "$LOCAL_REPO" rev-parse HEAD)
  GIT_DIRTY=clean
  [[ -n $(git -C "$LOCAL_REPO" status --porcelain) ]] && GIT_DIRTY=dirty
  {
    echo "branch:    $GIT_BRANCH"
    echo "commit:    $GIT_SHA_FULL"
    echo "state:     $GIT_DIRTY"
    echo "submitted: $(date -Iseconds)"
    echo
    echo "--- git status --short ---"
    git -C "$LOCAL_REPO" status --short
    echo
    echo "--- git diff --stat HEAD ---"
    git -C "$LOCAL_REPO" diff --stat HEAD
    echo
    echo "--- git diff HEAD (uncommitted changes, full) ---"
    git -C "$LOCAL_REPO" diff HEAD
  } > "$GIT_INFO_FILE"
else
  echo "(local repo is not a git checkout)" > "$GIT_INFO_FILE"
fi
# 2 MiB cap: a pathological diff should not blow up the sbatch script (it is
# submitted as a single `sbatch` argument with its own size limit).
if [[ $(stat -c%s "$GIT_INFO_FILE") -gt 2097152 ]]; then
  { head -c 2097152 "$GIT_INFO_FILE"; echo; echo "... [truncated, diff too large]"; } \
    > "${GIT_INFO_FILE}.trunc"
  mv "${GIT_INFO_FILE}.trunc" "$GIT_INFO_FILE"
fi
GIT_INFO_B64=$(base64 -w0 < "$GIT_INFO_FILE")
rm -f "$GIT_INFO_FILE"
WANDB_TAGS_VALUE="branch-${GIT_BRANCH//\//_},commit-${GIT_SHA},${GIT_DIRTY}"
[[ -n $WHO_TAG ]] && WANDB_TAGS_VALUE="${WHO_TAG//[^A-Za-z0-9_.-]/_}-cluster,$WANDB_TAGS_VALUE"
WANDB_NOTES_VALUE="branch=${GIT_BRANCH} commit=${GIT_SHA} (${GIT_DIRTY})"

# ------------------------------------------------------------------ preflight
# Fail here rather than 8 hours later when the upload step runs.
if [[ $CKPT_ENDPOINT == dropbox && $DRY_RUN -eq 0 ]]; then
  if ! ssh "$REMOTE_HOST" "test -x $REMOTE_ROOT/bin/rclone"; then
    echo "submit.sh: --ckpt-endpoint dropbox needs rclone at $REMOTE_ROOT/bin/rclone (not found)" >&2
    echo "           run 'scripts/skynet/submit.sh --help' for the one-time setup" >&2
    exit 3
  fi
  if ! ssh "$REMOTE_HOST" "$REMOTE_ROOT/bin/rclone listremotes" 2>/dev/null | grep -q '^dropbox:'; then
    echo "submit.sh: rclone has no 'dropbox:' remote configured" >&2
    echo "           run '$REMOTE_ROOT/bin/rclone config' on $REMOTE_HOST" >&2
    exit 3
  fi
fi

# ------------------------------------------------------------------ sync
# README.md is in this list because pyproject.toml declares `readme =
# "README.md"` and hatchling hard-fails the editable build without it.  It only
# ever worked here because an old manual copy happens to sit in the remote tree;
# a fresh bootstrap (as on PACE-ICE) fails without it.
if [[ $DO_RSYNC -eq 1 ]]; then
  echo "==> syncing $LOCAL_REPO -> $REMOTE_HOST:$REMOTE_REPO"
  rsync -az --delete --exclude '__pycache__/' \
    "$LOCAL_REPO/src" "$LOCAL_REPO/scripts" \
    "$LOCAL_REPO/pyproject.toml" "$LOCAL_REPO/uv.lock" "$LOCAL_REPO/.python-version" \
    "$LOCAL_REPO/README.md" \
    "$REMOTE_HOST:$REMOTE_REPO/"

  echo "==> isolating this submission's source -> $REMOTE_HOST:$ISOLATED_DIR"
  ssh "$REMOTE_HOST" "mkdir -p $ISOLATED_DIR"
  rsync -az --exclude '__pycache__/' \
    "$LOCAL_REPO/src" "$LOCAL_REPO/scripts" \
    "$LOCAL_REPO/pyproject.toml" "$LOCAL_REPO/uv.lock" "$LOCAL_REPO/.python-version" \
    "$LOCAL_REPO/README.md" \
    "$REMOTE_HOST:$ISOLATED_DIR/"

  # -------------------------------------------------------------- dataset
  # data/ is gitignored and tens of GB, so it is deliberately NOT part of the
  # per-job source sync -- copying it per submit would dominate submit time for
  # no benefit, and a blanket --sync-data pushes the WHOLE local data/ tree.
  # But it cannot simply be left in REMOTE_REPO either: command_cfgs.py
  # resolves DATASET_PATH relative to its OWN module file, which under source
  # isolation lives in job-src/, not REMOTE_REPO.  So the dataset lives ONCE
  # at $REMOTE_ROOT/data (outside the repo, immune to the --delete above) and
  # is symlinked into both trees.  Uploaded only when $DATA_DIR is still empty;
  # --sync-data forces a full refresh.  The emptiness test is deliberately NOT
  # keyed on any one dataset name -- retiring that dataset would otherwise make
  # the next submit re-upload the entire local data/ tree.  To add a single new
  # dataset, rsync that subdirectory to $DATA_DIR by hand.
  if [[ -d "$LOCAL_REPO/data" ]]; then
    if [[ $SYNC_DATA -eq 1 ]] \
       || ! ssh "$REMOTE_HOST" "ls -A '$DATA_DIR' 2>/dev/null | grep -q ."; then
      echo "==> syncing dataset -> $REMOTE_HOST:$DATA_DIR  (once; --sync-data to refresh)"
      ssh "$REMOTE_HOST" "mkdir -p '$DATA_DIR'"
      # rl_cache.pt is a regenerable derived cache, not input data.
      rsync -az --exclude '__pycache__/' --exclude 'rl_cache.pt' \
        "$LOCAL_REPO/data/" "$REMOTE_HOST:$DATA_DIR/"
    fi
    ssh "$REMOTE_HOST" \
      "ln -sfn '$DATA_DIR' '$ISOLATED_DIR/data'; ln -sfn '$DATA_DIR' '$REMOTE_REPO/data'"
  fi
fi

# ------------------------------------------------------------------ job body
# Caches all live on flash, shared by every compute node over NFS:
#   UV_CACHE_DIR / UV_PYTHON_INSTALL_DIR  nothing re-downloads per node
#   WARP_CACHE_PATH                       keeps the arch-keyed .ptx cache off the
#                                         slower 50 GB NFS home; entries are
#                                         per-arch, so the first job on a new GPU
#                                         type pays a one-off compile
read -r -d '' ENV_SETUP <<EOF || true
export PATH=$REMOTE_ROOT/bin:\$PATH
export UV_CACHE_DIR=$REMOTE_ROOT/.uv-cache
export UV_PYTHON_INSTALL_DIR=$REMOTE_ROOT/.uv-python
export WARP_CACHE_PATH=$REMOTE_ROOT/.warp-cache
export MUJOCO_GL=egl
export WANDB_TAGS="$WANDB_TAGS_VALUE"
export WANDB_NOTES="$WANDB_NOTES_VALUE"
$([[ -n $WANDB_ENTITY_PIN ]] && printf 'export WANDB_ENTITY=%s\nexport WANDB_USERNAME=%s' "$WANDB_ENTITY_PIN" "$WANDB_ENTITY_PIN")
$([[ $WANDB_OFFLINE -eq 1 ]] && echo 'export WANDB_MODE=offline')
$([[ $DO_RSYNC -eq 1 ]] && cat <<ISO
export VIRTUAL_ENV=$REMOTE_REPO/.venv
export UV_NO_SYNC=1
export UV_PROJECT=$ISOLATED_DIR
export PYTHONPATH=$ISOLATED_DIR/src
ISO
)
cd $REMOTE_REPO
EOF

QOS_LINE=""
[[ -n $QOS ]] && QOS_LINE="#SBATCH --qos=$QOS"

# Post-run handling runs whether training succeeded or not -- a preempted
# run's last checkpoint is exactly what you want preserved.  Every branch
# below follows the same rule: upload/copy, VERIFY, and only on a verified
# match delete the local model_*.pt / model_*.onnx.  A failed or skipped
# verification always leaves the local files in place.
case "$CKPT_ENDPOINT" in

  wandb)
    # mjlab's own wandb.save() already pushed each checkpoint live during
    # training (the default --agent.upload-model True).  This just confirms
    # the sizes match on W&B's side before deleting the local copies.
    POST_RUN_BLOCK=$(cat <<EOF
if [ "$KEEP_LOCAL" = "1" ]; then
  echo "[submit.sh] --keep-local set, leaving checkpoints in \$RUN_DIR"
elif [ "${WANDB_OFFLINE}" = "1" ]; then
  echo "[submit.sh] --offline set: checkpoints are only in the local wandb run dir," >&2
  echo "[submit.sh] not yet on W&B.  Run 'wandb sync' yourself before deleting anything." >&2
elif [ -z "\$RUN_DIR" ] || [ ! -d "\$RUN_DIR" ]; then
  echo "[submit.sh] WARNING no run dir found, skipping verify/prune" >&2
elif [ -z "\$WANDB_URL" ]; then
  echo "[submit.sh] WARNING no wandb run URL captured, skipping verify/prune" >&2
else
  echo "[submit.sh] verifying checkpoints landed on \$WANDB_URL"
  if $REMOTE_REPO/.venv/bin/python $REMOTE_REPO/scripts/skynet/verify_wandb_checkpoints.py --run-url "\$WANDB_URL" --run-dir "\$RUN_DIR"; then
    echo "[submit.sh] verified -- removing local checkpoints"
    rm -fv "\$RUN_DIR"/model_*.pt "\$RUN_DIR"/model_*.onnx 2>/dev/null || true
  else
    echo "[submit.sh] WARNING wandb verify failed, keeping local checkpoints" >&2
  fi
fi
EOF
)
    ;;

  dropbox)
    POST_RUN_BLOCK=$(cat <<EOF
if [ -n "\$RUN_DIR" ] && [ -d "\$RUN_DIR" ]; then
  DEST="dropbox:$DROPBOX_PATH/\$(basename "\$RUN_DIR")"
  echo "[submit.sh] rclone copy -> \$DEST"
  if $REMOTE_ROOT/bin/rclone copy --stats-one-line --stats 30s --transfers 4 "\$RUN_DIR" "\$DEST"; then
    if [ "$KEEP_LOCAL" = "1" ]; then
      echo "[submit.sh] --keep-local set, leaving checkpoints in \$RUN_DIR"
    else
      echo "[submit.sh] verifying transfer (rclone check) before deleting anything local"
      if $REMOTE_ROOT/bin/rclone check "\$RUN_DIR" "\$DEST" --one-way; then
        echo "[submit.sh] verified -- removing local checkpoints"
        rm -fv "\$RUN_DIR"/model_*.pt "\$RUN_DIR"/model_*.onnx 2>/dev/null || true
      else
        echo "[submit.sh] WARNING rclone check found mismatches, keeping local checkpoints" >&2
      fi
    fi
  else
    echo "[submit.sh] WARNING rclone copy failed, keeping local checkpoints" >&2
  fi
else
  echo "[submit.sh] WARNING no run dir found, skipping dropbox upload" >&2
fi
EOF
)
    ;;

  local)
    POST_RUN_BLOCK='echo "[submit.sh] --ckpt-endpoint local: checkpoints stay only in \"$RUN_DIR\" (never auto-deleted)"'
    ;;
esac

SBATCH_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
$QOS_LINE
#SBATCH --gres=gpu:$GPU_TYPE:$GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$SLURM_LOG_DIR/%x-%j.out
#SBATCH --open-mode=append
set -euo pipefail
$ENV_SETUP
echo "[submit.sh] node=\$(hostname)  job=\$SLURM_JOB_ID  gpu=\$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$([[ $DO_RSYNC -eq 1 ]] && echo "echo \"[submit.sh] isolated source (this job's own src/, immune to later resubmits): $ISOLATED_DIR\"")
echo "[submit.sh] git snapshot (full diff saved to the run dir + wandb tags/notes):"
# The trailing '|| true' is load-bearing: head closes the pipe after 80 lines,
# and once the diff exceeds the pipe buffer that lands SIGPIPE on the writer,
# which under 'set -euo pipefail' kills this script before training starts.
# NB backticks are command substitution inside this unquoted heredoc -- never
# use them in these comments.
echo "\$(echo $GIT_INFO_B64 | base64 -d)" | head -80 || true
echo "[submit.sh] command:"
echo "\$(echo $CMD_B64 | base64 -d)"
echo "[submit.sh] ----------------------------------------------------------"

# tee so the run directory can be recovered afterwards; PIPESTATUS keeps the
# training exit code rather than tee's.
WORKLOG=\$(mktemp)
set +e
eval "\$(echo $CMD_B64 | base64 -d)" 2>&1 | tee "\$WORKLOG"
RC=\${PIPESTATUS[0]}
set -e
RUN_DIR=\$(sed -n 's/.*Logging experiment in directory: //p' "\$WORKLOG" | head -1) || true
WANDB_URL=\$(grep -oE 'https://wandb\.ai/[^ ]+/runs/[A-Za-z0-9_-]+' "\$WORKLOG" | tail -1)
rm -f "\$WORKLOG"
echo "[submit.sh] training exit=\$RC  run_dir=\${RUN_DIR:-unknown}"
if [ -n "\$RUN_DIR" ] && [ -d "\$RUN_DIR" ]; then
  echo "\$(echo $GIT_INFO_B64 | base64 -d)" > "\$RUN_DIR/git-diff.txt"
fi
$POST_RUN_BLOCK
exit \$RC
EOF
)

if [[ $DRY_RUN -eq 1 ]]; then
  echo "$SBATCH_SCRIPT"; exit 0
fi

# ------------------------------------------------------------------ launch
if [[ $MODE == srun ]]; then
  echo "==> srun (interactive, dies with this shell)"
  SRUN_QOS=""; [[ -n $QOS ]] && SRUN_QOS="--qos=$QOS"
  exec ssh -t "$REMOTE_HOST" "bash -lc '
    $ENV_SETUP
    srun -p $PARTITION -A $ACCOUNT $SRUN_QOS --gres=gpu:$GPU_TYPE:$GPUS \
         -c $CPUS --mem=$MEM -t $TIME -J $JOB_NAME --pty \
         bash -c \"eval \\\"\\\$(echo $CMD_B64 | base64 -d)\\\"\"
  '"
fi

echo "==> sbatch  $PARTITION/$ACCOUNT${QOS:+/$QOS}  ${GPUS}x$GPU_TYPE  ${CPUS}cpu  $MEM  $TIME"
JOB_ID=$(printf '%s\n' "$SBATCH_SCRIPT" | \
  ssh "$REMOTE_HOST" "bash -lc 'mkdir -p $SLURM_LOG_DIR && sbatch --parsable'")

LOG_PATH="$SLURM_LOG_DIR/$JOB_NAME-$JOB_ID.out"
cat <<EOF

    job id   $JOB_ID
    log      $REMOTE_HOST:$LOG_PATH

    ssh $REMOTE_HOST "bash -lc 'squeue -j $JOB_ID'"
    ssh $REMOTE_HOST "tail -f $LOG_PATH"
    ssh $REMOTE_HOST "bash -lc 'scancel $JOB_ID'"
EOF

# ------------------------------------------------------------------ follow
if [[ $FOLLOW -eq 1 ]]; then
  echo
  echo "==> following (Ctrl-C detaches; the job keeps running)"
  # Wait for the log to exist, then tail it until the job leaves the queue.
  ssh "$REMOTE_HOST" "bash -lc '
    while [ ! -f $LOG_PATH ] && squeue -h -j $JOB_ID -o %T | grep -q .; do sleep 5; done
    tail -f -n +1 $LOG_PATH &
    TAILPID=\$!
    while squeue -h -j $JOB_ID -o %T | grep -q .; do sleep 20; done
    sleep 3; kill \$TAILPID 2>/dev/null || true
  '"
  echo
  ssh "$REMOTE_HOST" "bash -lc 'sacct -j $JOB_ID --format=JobID%12,JobName%16,State%14,Elapsed,ExitCode -P | head -3'"
fi
