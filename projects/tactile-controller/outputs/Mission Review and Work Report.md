# Zhengyu assignment review and work report

> Scope correction, 3 October 2026: This document evaluates an older foot-tracking assignment, not the current mission. The user's newer Slack screenshots specify navigation on `HuazeLiu/tactile-navigation`, branch `sky/evaluation`, followed by new-map robustness experiments. Its checkpoint results remain valid only for the older screening. Current mission work is in the separate `tactile-navigation` checkout.

Prepared 3 October 2026, Asia/Taipei.

## Outcome

I recovered the earlier assignment from a locally saved screenshot of Zhenyu Wu's Slack message, inspected the existing experiment, and ran a fresh headless comparison of the supplied reference checkpoint and the existing tuned checkpoint. The tuned checkpoint performed worse on average reference-foot tracking in this screening. Higher accuracy has not been achieved or established.

I could not read the current live Slack conversation: Slack is not installed/connected to this chat, and the available browser inventory had no Slack tab. The saved screenshot may omit newer instructions. Install and connect Slack to verify the latest assignment and the precise publication permission.

## Assignment visible in the saved message

- Use the playground branch of tactile_controller.
- Study the README with AI assistance.
- Train `Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1` with TensorBoard.
- Try tuning the foot-tracking reward and compare footstep accuracy against `checkpoints/decoupled_footplacement/0907/foot_mimic_noforce_model_34000.pt`.
- Simulation experiments and data analysis were offered as possible additional work; the screenshot does not specify another assignment for those.

## Work performed in this session

1. Checked Slack integration availability and existing browser access.
2. Read the saved assignment screenshot and relevant controller documentation/source.
3. Inspected the existing reward diff, saved training configuration, checkpoint files, and historical experiment evidence.
4. Verified CUDA availability through the existing WSL Python environment. The checkout-local `.venv` failed to import torch; `/home/sky/.venvs/tactile_controller/bin/python` worked. I used that working environment directly.
5. Created `outputs/mission_review/evaluate_foot_tracking.py` and ran both checkpoints successfully in fresh processes.
6. Saved JSON results and execution logs for each checkpoint.
7. Wrote `outputs/GitHub Portfolio Brainstorm.md`, including repository structure, attribution, contribution statements, evidence presentation, and experiment ideas.

The reward modification and September training run existed before this session. I did not create them today or perform new training. I did not publish to GitHub or send Slack messages.

## New checkpoint screening

Both runs used the same task, seed 42, 32 parallel environments, 500 control steps (10 seconds per environment), noise-disabled play observations, and restored safety terminations and 10-second episode limits. Actors loaded strictly. Each run used a separate process.

| Metric | Supplied reference, model 34000 | Existing tuned, model 499 |
|---|---:|---:|
| Mean reference-foot error | 0.096223 m | 0.330168 m |
| 90th percentile reference-foot error | 0.183674 m | 0.615021 m |
| Episode ends, including timeouts | 32 | 32 |

The mean is the command's `error_feet_pos` metric: the Euclidean error between each foot's reference and simulated position, averaged across both feet and sampled over the rollout. It measures trajectory/reference tracking, not landing accuracy. Episode-end counts include timeouts and must not be labeled fall counts. The error can be stale around automatic resets, and resets can make command histories diverge. This is a single-seed preliminary screening, not a paired landing trial or a matched-training ablation.

The tuned checkpoint's mean error is about 3.43 times the reference's in this screen. Different training histories prevent attributing that gap to the reward modification alone. The result supports saying that the existing tuned checkpoint does not demonstrate improved tracking; it does not prove the reward idea is inherently ineffective.

Results:

- `outputs/mission_review/reference_evaluation.json`
- `outputs/mission_review/tuned_evaluation.json`
- `outputs/mission_review/reference_evaluation.log`
- `outputs/mission_review/tuned_evaluation.log`

## Existing experiment found on disk

The branch is `codex/foot-tracking-tuning`. Its existing uncommitted reward change adds a lower-body foot-position reward kernel with standard deviation 0.05 m and weight 1.0, retaining the original 0.20 m signal.

The saved September run used 1,024 environments, seed 42, 24 rollout steps, 500 training iterations, and TensorBoard. Its checkpoint is `model_499.pt`. Historical evidence records a final training reference-foot error of 0.259466 m and a final-20-iteration average of 0.243013 m. Those historical values are distinct from today's inference screening.

The existing report records dataset generation/validation and baseline/tuned smoke tests. I inspected that report material but did not rerun dataset validation or those training tests today.

## Next experiment needed for a stronger conclusion

Compare baseline and tuned training with the same initialization, training sample budget, command distribution, and evaluation suite. Keep the 0.05 m reward addition as the only changed condition. Evaluate several seeds, measure landing error at step completion separately from reference tracking, and record stability and hand tracking. Save each run's code/configuration and compare videos as well as numerical metrics.

For the existing from-scratch run, approximately 4,000 iterations at 1,024 environments match the sample count of 500 iterations at 8,192 environments. That arithmetic does not establish convergence, equivalence to the mature reference checkpoint, or an improvement caused by the reward.

## Reproduce today's screen

From WSL, in the repository directory:

```bash
cd '/mnt/c/Users/gigabyte/Documents/ChatGPT/GT research'
/home/sky/.venvs/tactile_controller/bin/python outputs/mission_review/evaluate_foot_tracking.py \
  checkpoints/decoupled_footplacement/0907/foot_mimic_noforce_model_34000.pt \
  outputs/mission_review/reference_evaluation.json
/home/sky/.venvs/tactile_controller/bin/python outputs/mission_review/evaluate_foot_tracking.py \
  logs/rsl_rl/g1_decoupled_joint_foot_mimic_arm/2026-09-13_22-38-08_foot_precise_s005_500/model_499.pt \
  outputs/mission_review/tuned_evaluation.json
```

## Portfolio recommendation

Use a profile README and a pinned `g1-footstep-tracking-study` case study. Explain the question, team's foundation, your exact contribution, experiment settings, a simulation demo, results, and what the unsuccessful tuning attempt taught you. Your stated permission from Zhengyu should be reflected in acknowledgements while keeping authorship accurate. The detailed brainstorm contains templates and a proposed folder layout.

The new portfolio and report drafts are local deliverables. Current Slack review and any newer assignment remain pending access; improved accuracy remains an open research outcome.
