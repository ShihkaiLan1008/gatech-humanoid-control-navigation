# Research paused at the user's request

Historical note: the user subsequently requested resumption, and the pilot is now complete. See [Research Completion.md](Research%20Completion.md) for the current state.

3 October 2026, Asia/Taipei. No simulation or video-rendering process remained running when checked. Do not resume until the user requests it.

## Saved state

- Original room2: physical goal reached; framework `arrival_uncertain`; no fall; video complete.
- New island choice: physical goal reached; framework `arrival_uncertain`; no fall; video complete.
- New offset gate: restarted after the Windows crash and completed; physical goal reached; framework `arrival_uncertain`; no fall; video not yet rendered.
- New staggered baffles: completed immediately before the pause; `route_complete`, framework success true, physical goal reached, no fall. Final goal error 0.041588 m; localization RMSE 0.068326 m; simulation duration 423.92 s. Video not yet rendered.

All three new maps have completed physical trials. Do not rerun them merely to resume this task.

Completed records:

```text
out/sky_evaluation_20261003/original_room2/
out/sky_evaluation_20261003/new_maps_physics/footstep/sky_island_choice/seed_0/
out/sky_evaluation_20261003/new_maps_physics_restart/
```

The original interrupted offset-gate journal is preserved in `new_maps_physics/footstep/sky_offset_gate/seed_0/recovered.eval.json` with outcome `system_restart`.

## Resume next time

1. Refresh the report from the four completed records using `scripts/sky_report.py`. The current report was last generated before staggered baffles finished and still shows that trial as pending.
2. Render only the two completed restarted trials using `scripts/pace/render_eval.py out/sky_evaluation_20261003/new_maps_physics_restart --stride 10`. Run rendering separately from simulation.
3. Verify video frames and the refreshed trajectory figure, then deliver the completed navigation report and GitHub portfolio plan.
4. If the user wants exact Windows driver diagnosis, obtain an administrator-copied readable dump. `outputs/crash_diagnostics/Crash and Recovery.md` documents the confirmed 0x139 crash and permission limitation. No permanent driver fix has been established.

Repository: `C:\Users\gigabyte\Documents\ChatGPT\tactile-navigation`, branch `sky/evaluation`. WSL Python: `/home/sky/.venvs/tactile_controller/bin/python`. Controller/checkpoint links and unchanged experiment settings are documented in `docs/SKY_MAP_EXPERIMENT.md`.
