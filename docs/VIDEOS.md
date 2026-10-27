![Recorded experiments](../assets/section-videos.svg)

[Portfolio home](../README.md) · [Research overview](ADMISSIONS_RESEARCH_OVERVIEW.md) · [Technical methods](TECHNICAL_DEPTH.md)

# The experiments, in motion

These recordings replay saved, executed simulation states alongside trajectory and localization diagnostics. **They show simulated movement using the team's trained controller, not a real robot or a hand-animated path.** All four trials physically reached the goal region; one met the framework's completion condition.

## Watch or inspect a run

**GitHub's file page cannot preview these MP4 files.** Use the direct **Download MP4** links below, then open the file in a video player. Use **View frames** to inspect start, middle and end images without downloading a video. No GitHub sign-in is needed for the public file links.

| Run | Replay duration | File size | Recording and frame sheet | Framework decision |
| :--- | ---: | ---: | :--- | :--- |
| Original room | 3.1 min | 17.3 MiB | [Download MP4](https://github.com/ShihkaiLan1008/gatech-humanoid-control-navigation/raw/refs/heads/main/projects/tactile-navigation/out/sky_evaluation_20261003/original_room2/footstep/room2/seed_0/room2_footstep_pf_global_s0.mp4) · [View frames](../projects/tactile-navigation/outputs/room2_video_review.jpg) | Arrival uncertain |
| Central island | 4.3 min | 13.3 MiB | [Download MP4](https://github.com/ShihkaiLan1008/gatech-humanoid-control-navigation/raw/refs/heads/main/projects/tactile-navigation/out/sky_evaluation_20261003/new_maps_physics/footstep/sky_island_choice/seed_0/sky_island_choice_footstep_pf_global_s0.mp4) · [View frames](../projects/tactile-navigation/outputs/sky_island_choice_video_review.jpg) | Arrival uncertain |
| Offset opening | 5.2 min | 18.5 MiB | [Download MP4](https://github.com/ShihkaiLan1008/gatech-humanoid-control-navigation/raw/refs/heads/main/projects/tactile-navigation/out/sky_evaluation_20261003/new_maps_physics_restart/footstep/sky_offset_gate/seed_0/sky_offset_gate_footstep_pf_global_s0.mp4) · [View frames](../projects/tactile-navigation/outputs/sky_offset_gate_video_review.jpg) | Arrival uncertain |
| Staggered baffles | 7.1 min | 23.4 MiB | [Download MP4](https://github.com/ShihkaiLan1008/gatech-humanoid-control-navigation/raw/refs/heads/main/projects/tactile-navigation/out/sky_evaluation_20261003/new_maps_physics_restart/footstep/sky_staggered_baffles/seed_0/sky_staggered_baffles_footstep_pf_global_s0.mp4) · [View frames](../projects/tactile-navigation/outputs/sky_staggered_baffles_video_review.jpg) | Complete |

*Duration describes the video file; simulation timing and replay sampling are documented in the [saved video audit](../projects/tactile-navigation/outputs/sky_video_validation.json).*

## A glimpse of the runs

| Original room | Central island |
| :---: | :---: |
| [![Original-room simulation frame](../projects/tactile-navigation/outputs/original_room2_video_frame.png)](../projects/tactile-navigation/outputs/room2_video_review.jpg) | [![Central-island simulation frame](../projects/tactile-navigation/outputs/island_video_frame.png)](../projects/tactile-navigation/outputs/sky_island_choice_video_review.jpg) |

Walls occasionally occlude the robot; the trajectory panel helps track its movement. Replay alone does not establish localization confidence or formal success.

[Read one run alongside its map, measurements and sensing events](RESEARCH_WALKTHROUGH.md#open-one-experiment-the-offset-opening-run) · [Compare the executed trajectories](../projects/tactile-navigation/outputs/sky_navigation_trajectories.png)

## Provenance

The original recordings and frame sheets are preserved. The earlier video audit records full decoding of each file; no new simulation or new video rendering was performed for this presentation revision. Simulation operation and presentation received substantial Codex assistance under Sky's direction. [Team, student and AI roles](AI_AND_SOURCE_DISCLOSURE.md)

For a local repository clone, use `git lfs pull` to obtain the large files.
