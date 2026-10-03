# retarget_motion

Human mocap retargeted to the G1, landed in one canonical clip format and
playable in the browser.

```
raw dataset  ->  [T,36] G1 rows  ->  canonical .npz  ->  viser
 (download)        (sources/)        (convert.py)      (viser_play.py)
```

## Quick start

```bash
uv run python -m retarget_motion.download --source all      # ~5 GB
uv run python -m retarget_motion.convert  --source all
uv run python -m retarget_motion.viser_play                 # then open the printed port
```

`--limit N` on download/convert smoke-tests the pipeline without pulling 17k
clips.

## What the data is

| Source | Clips | Gate | Retargeted by |
|---|---|---|---|
| **AMASS** (`fleaven/Retargeted_AMASS_for_robotics`) | 17,717 over 25 sub-datasets | none, CC-BY-4.0 | that project's own pipeline |
| **LAFAN1** (`lvhaidong/LAFAN1_Retargeting_Dataset`) | 40 | none | Unitree, Interaction-Mesh + IK |
| **Inter-X** | — | **access-gated**, not implemented | see `sources/interx_g1.py` |

The upstream AMASS sub-dataset licences (research / non-commercial) still apply
on top of the CC-BY-4.0 retargeting — cite them.

**Why pre-retargeted rather than running GMR ourselves.** GentleHumanoid
retargets these three with their GMR fork, which needs an AMASS account, a
*separate* SMPL-X body-model account (a hard dependency of their AMASS path),
and hours of IK. Both G1 sources above are already retargeted, ungated, and
carry the same 29-joint order as `assets.TACTILE_XML` — verified joint by joint
— so the whole GMR install is avoidable. The honest cost: a different
retargeter produced them, so motion *style* will not match GentleHumanoid's
clips. The converted *schema* is identical, so nothing downstream cares.

## Output format

One `.npz` per clip under `data/retarget_motion/motions/<SOURCE>/...`, field for
field the GentleHumanoid / GMR-fork schema, so their `generate_dataset.py` can
be pointed straight at `motions/`. Full spec in `schema.py`; the two conventions
that bite:

- **quaternions are XYZW** (scalar last) everywhere — MuJoCo is WXYZ, and
  `Motion.to_qpos()` is the only place the swap happens;
- **`root_pos` is world and ground-adjusted.** The raw sources' root z is
  frame-0-relative: believe it and the robot sits ~0.72–0.75 m *inside* the
  floor (measured). `fk.ground_offset` fixes it from actual forward kinematics
  rather than a constant.

We add `src_dataset` / `src_path` / `src_fps` / `src_ground_dz` for provenance;
reference consumers ignore unknown keys.

## Things that were measured, not assumed

- **AMASS mixes six frame rates** — 120 Hz (12,222 files), 100 (4,546), 60
  (871), 250 (56), 59 (13), 150 (9) — encoded in each filename. Assuming 120
  would time-stretch a quarter of the dataset, so a file without a parseable
  rate is rejected rather than guessed.
- **The ground shift is genuinely per-dataset**: LAFAN needs about −3 cm (its
  CSV root really is world height), AMASS needs about +0.72 m. Computing it
  from FK handles both with no special-casing.
- **Turn the hf-xet backend off for these repos.** Xet takes a per-file read
  token; 17,717 small files at 12 workers drew HTTP 429 after ~1,300 files.
  `download.py` disables it by default and retries with backoff (downloads
  resume, so a retry costs nothing).
- The AMASS dataset card names `g1_29dof_lock_waist_rev_1_0.xml` as its target,
  implying a frozen waist. It does not match the data — waist roll/pitch
  genuinely move (0.45 rad of pitch in AMASS CMU 01_01). The data wins.

## Layout

```
paths.py      where things live (data/retarget_motion/{raw,motions})
schema.py     the canonical clip: Motion, load(), validate(), to_qpos()
sources/      one ingest per raw dataset -> RawClip
fk.py         MuJoCo FK on assets.TACTILE_XML: local_body_*, ground offset
resample.py   fps conversion (SLERP rotations, PCHIP everything else)
download.py   fetch raw datasets from the Hub
convert.py    the driver, raw/ -> motions/
viser_play.py browser playback + library browser
```

`raw/` is never written to by the converter, so re-running conversion can't
damage the slow-to-fetch download.
