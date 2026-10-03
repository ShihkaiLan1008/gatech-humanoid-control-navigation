"""The repo's ONE robot-asset folder.

Every G1 description the code loads lives in `g1_description/` and is resolved
through this module -- `src/mpc_offline`, `src/rl` and the cuRobo tools (now `src/data_gen/ee_target_curobo`) used to carry
their own copy-pasted trees, which drifted (the RL and offline `g1_tactile.xml`
were two different models under one filename).  Import the constants below
instead of building a path from `__file__`, and there is exactly one place to
look when an asset changes.

    from assets import ASSET_DIR, TACTILE_URDF, TACTILE_XML

`src` is on `sys.path` (editable install), so `import assets` works from any
package without a reinstall.

WHAT IS IN HERE
---------------
MJCF -- two different builds, do not confuse them (they used to share a name):

* `g1_tactile.xml`        ROBOT ONLY, no scene/actuators/sensors.  The mjlab
                          TRAINING model (`rl.g1_assets.g1_constants.G1_XML`);
                          mjlab composes terrain/actuators around it.
* `g1_tactile_scene.xml`  STANDALONE MuJoCo model: floor, lights, tracking
                          cameras, actuators, sensors, contact exclusions.
                          The verified twin used for clip PLAYBACK
                          (`mpc_offline.viser_play`, `play_clip`).
* `g1.xml`                sensor-less counterpart of `g1_tactile.xml`
                          (wrist_yaw terminal).  Kept for A/B only.

URDF -- Crocoddyl (pinocchio) and the RL side's pytorch-kinematics IK chains:

* `g1_29dof_tactile.urdf`   TACTILE EE.  Adds one fixed `{left,right}_tactile`
                            link per side on wrist_yaw_link at
                            (0.1325, +/-0.003, 0); the link origin is the
                            sensing dome's APEX, i.e. the point that touches
                            the wall.  The current EE for every live task.
* `g1_29dof_nohand.urdf`    wrist_yaw EE, no hands.  LEGACY: still the default
                            of `mpc_offline.utils.load_robot`, used by
                            wall_wipe / wall_align / locomotion.
* `g1_29dof_rev_1_0.urdf`   stock vendor URDF, rubber hands attached.  Used by
                            hand_pose's (stage-1, wrist-frame) IK chain.
* `g1_29dof_tactile_curobo.urdf` + `g1_tactile*.yml`
                            cuRobo variants (`src/data_gen/ee_target_curobo`).  The .yml files carry
                            ABSOLUTE `asset_root_path` / `urdf_path` entries --
                            regenerate them if this folder ever moves.

Never the `*_with_inspire_hand_*` / `*_with_hand_*` vendor variants; they are
not here and are not used (they remain in git history).

NOT here on purpose: `src/mpc_online/assets/g1_xml/` is a DIFFERENT robot
revision (35.11 kg vs 33.34 kg; pre-rev_1_0 `torso_link` / `waist_*_link`
meshes, its own actuator classes).  Pointing it here would silently change the
SRBD/CD prototypes' physics, so it keeps its own tree.
"""

from pathlib import Path

# g1_description/ holds the descriptions AND their shared meshes/ subfolder.
# Both pinocchio (`mesh_dir`) and MuJoCo (`meshdir="meshes"`) resolve against
# this directory, so it is the one path callers need.
ASSET_DIR: Path = Path(__file__).resolve().parent / "g1_description"
MESH_DIR: Path = ASSET_DIR / "meshes"

# MJCF
TACTILE_XML = "g1_tactile.xml"              # mjlab training model (robot only)
TACTILE_SCENE_XML = "g1_tactile_scene.xml"  # standalone playback model (+scene)
PLAIN_XML = "g1.xml"                        # sensor-less, A/B only
UNITREE_XML = "unitree_g1.xml"                # vendor MJCF (rubber-hand wrists)
# unitree_g1.xml with ONLY the hand swapped (2026-08-08): wrist inertial split
# back to bare (0.08458 — the vendor 0.254576 had the 0.170 rubber hand merged
# in), rubber-hand visual + hand_collision capsule removed, and the VERIFIED
# tactile block added per side (body {left,right}_tactile at (0.1325, ±0.003,
# 0) = dome apex, collision sphere r 0.040 surface-through-apex, apex site).
# Sensor mass 0.050 kg is a PLACEHOLDER (real sensor unweighed, much lighter
# than the rubber hand) — edit the two inertials when measured. Everything
# else byte-identical to unitree_g1.xml; verified: same nq/nv/joint order,
# tactile-vs-wrist FK == g1_tactile.xml to 1e-5, mass ledger exact.
UNITREE_TACTILE_XML = "unitree_g1_tactile.xml"

# URDF
TACTILE_URDF = "g1_29dof_tactile.urdf"
NOHAND_URDF = "g1_29dof_nohand.urdf"
REV_1_0_URDF = "g1_29dof_rev_1_0.urdf"
TACTILE_CUROBO_URDF = "g1_29dof_tactile_curobo.urdf"

# cuRobo robot configs
TACTILE_CUROBO_YML = "g1_tactile.yml"
TACTILE_CUROBO_FIXED_YML = "g1_tactile_fixed.yml"


def asset(name: str) -> str:
    """Absolute path to `name` inside g1_description/, checked to exist."""
    p = ASSET_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"no such asset: {p}")
    return str(p)


__all__ = [
    "ASSET_DIR",
    "MESH_DIR",
    "NOHAND_URDF",
    "PLAIN_XML",
    "REV_1_0_URDF",
    "TACTILE_CUROBO_FIXED_YML",
    "TACTILE_CUROBO_URDF",
    "TACTILE_CUROBO_YML",
    "TACTILE_SCENE_XML",
    "TACTILE_URDF",
    "TACTILE_XML",
    "asset",
]
