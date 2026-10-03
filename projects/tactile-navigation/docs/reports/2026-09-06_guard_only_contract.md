# Guard-only completion fix

Historical stage: superseded by [passive-contact replanning](2026-09-06_passive_contact_replanning.md).

Branch: `punching`. Behavioral baseline: `0d83542`.

This supersedes the combined executor/recovery/calibration implementation and
evaluation at `228064b`. That revision remains available in Git; its room results
are **not measurements of this guard-only version**. Raw recordings and comparison
videos under `out/executor_repair/` have not been deleted.

## Retained behavior

- `TactileIO.move()` returns `guard` immediately after consuming a force abort,
  before checking arrival or watched contact. It issues no further commands within
  that move, including its normal final hold, after a guard.
- A guard during the final hold also overrides `done`. Elapsed odometry is still
  integrated once on return.
- EIG-step and EIG-route callers propagate a guarded move/turn as `move_guard` or
  `turn_guard`, without subsequent punches, submoves, or completed-turn bookkeeping.
  This terminates the policy run; it is not an automatic recovery mechanism.

## Removed / unchanged

The additional completion feedback/dwell contract, geometric recovery/resume,
calibrated execution model, fitting/evaluation tools, and associated strategy and
rendering changes have been removed. Their previous sources and reports can be
recovered from `228064b`.

Original non-guard completion criteria, timing, timeout/watch-contact behavior,
action scoring, controller safety-stop implementation, and punch behavior remain
unchanged. Legacy contact-rule policy handling is not redesigned in this patch.
No room-specific conditions or new thresholds were introduced. No world model is
connected.

## Validation

`tests/execution/test_guard_completion.py`: 15 tests passed, covering interruption,
guard priority, final-hold guards, caller propagation, and unchanged normal timing.
Running the same tests against the untouched `0d83542` worktree reproduces nine
guard-related failures; the six non-guard tests pass on both versions.
The execution test suite (`/usr/bin/python3 -m pytest tests/execution -q`) also
completed successfully with exit code 0 (existing skips remain).
No new physical room benchmark was run, and no navigation-performance improvement
is claimed.
