#!/usr/bin/env bash
# Queued with afterany:<evaluation-job-id> by submit_eval.py.
# Arguments: snapshot evaluation-job-id Dropbox-base-directory remote-rclone
#SBATCH --job-name=tacnav-upload
#SBATCH --partition=ice-cpu
#SBATCH --qos=coe-ice
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:30:00

set -euo pipefail
SNAPSHOT=${1:?snapshot directory required}
EVAL_JOB_ID=${2:?evaluation job ID required}
DROPBOX_BASE=${3:?Dropbox directory required}
RCLONE=${4:?remote rclone executable required}
RUN_DIR="$SNAPSHOT/tactile-navigation/out/evaluation/pace_$EVAL_JOB_ID"
DEST="dropbox:${DROPBOX_BASE%/}/pace_$EVAL_JOB_ID"

# Preserve diagnostics even if the evaluation failed before creating its output.
mkdir -p "$RUN_DIR"
for name in submission.json SHA256SUMS; do
  [[ ! -f "$SNAPSHOT/$name" ]] || cp "$SNAPSHOT/$name" "$RUN_DIR/"
done
if [[ -f "$SNAPSHOT/slurm-$EVAL_JOB_ID.log" ]]; then
  cp "$SNAPSHOT/slurm-$EVAL_JOB_ID.log" "$RUN_DIR/slurm.log"
fi

echo "Uploading $RUN_DIR -> $DEST"
# Copy never deletes PACE results or unrelated Dropbox files. Verification
# failure makes this upload job fail and leaves all original results on PACE.
"$RCLONE" copy "$RUN_DIR" "$DEST" --stats-one-line --stats 30s --transfers 4
"$RCLONE" check "$RUN_DIR" "$DEST" --one-way
echo "Upload verified: $DEST"
