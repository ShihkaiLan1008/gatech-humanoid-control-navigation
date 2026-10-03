"""Sample the reachable region of the two tactile end-effectors, on the GPU.

The question this answers: *where, in the PELVIS frame, can the robot put a
tactile tip?*  That set is the domain the hand EE command of CURRENT GOAL lives
in (`{left,right}_tactile` pose in the robot/pelvis frame), so it is the region
a command sampler may draw from.

Pipeline (all batched on the GPU, one cuRobo model shared by both stages):

1. **sample** joint configurations inside the URDF joint limits, from a
   STREAMING Halton sequence (see `build_sampler` -- cuRobo's own
   `RobotSceneCollision.sample` recycles a 2000-point buffer and must not be
   used here),
2. **reject** anything that violates a joint bound or self-collides (cuRobo's
   sphere model over the asset's `self_collision_ignore` matrix),
3. **FK** the surviving configurations to the two tactile frames,
4. **re-express** each tip pose in the pelvis frame -- `T_pelvis^-1 * T_tip` --
   so the result does not depend on where the floating base sits,
5. **cut down to the commandable region**, all in the pelvis frame and all
   reported one at a time so you can see what each cut costs:

   * `x > --x-margin` -- in front of the pelvis,
   * outboard `y` no further than `--y-reach` from THAT ARM'S SHOULDER
     (`*_shoulder_pitch_link`, the arm chain's first joint),
   * inboard `y` no closer than `--y-margin` to the pelvis mid-plane, so the
     left tip stays at `y >= +margin` and the right at `y <= -margin`: the two
     hands can never cross, and a command sampler can draw the two hands
     INDEPENDENTLY without ever producing a self-crossing pair,
   * `z <= --z-max`, default the top of `head_link`,
   * `z >= --z-min` (off by default).

   The y and z landmarks are read off the model at startup (`geometry_bounds`),
   not hard-coded, so they follow the asset.

Steps 4-5 are per HAND: a configuration whose left tip passes the cuts and
whose right tip does not contributes one point, not zero.

Run
---
    uv run python -m data_gen.ee_target_curobo.reach_boundary --num 200000 --visualize

`--visualize` serves a viser scene (forward the printed port when remote) with
one point cloud per hand -- left BLUE, right ORANGE -- the pelvis triad, and the
robot for scale.

Robot model
-----------
Default `g1_tactile_fixed.yml`: base, legs and waist LOCKED at the standing
half-sitting pose, so the 14 arm joints are the only decision variables and the
region is the *arms-alone* reach (the RL stack's torso-rooted IK convention).

`--robot g1_tactile.yml` runs the full 35-DoF model. Step 4 keeps the output
comparable -- the randomized floating base cancels exactly -- and the waist
twist widens the region a lot, but it also samples the LEGS uniformly, so that
region includes postures the robot could not stand in. Upper bound only.

Caveats worth knowing before you trust the numbers
--------------------------------------------------
* **This is the FREE-ORIENTATION reach set.** It answers "can the tip be at
  this point in SOME orientation", which is NOT what a command sampler needs --
  the hand commands pin the tip orientation (hand_force samples rpy +-0.1 rad).
  Conditioning by rejection does not work: only 221 of 600 000 samples land
  within 0.18 rad of identity. For the orientation-conditioned set, sweep it
  with IK instead.
* **The fitted `torso_link` spheres are too fat.** They overlap
  `*_shoulder_{roll,yaw}_link` by up to 11 mm at the robot's NOMINAL standing
  pose, where the real MESHES are +8.6 to +13.9 mm apart -- a fit artifact.
  Left alone it rejects the nominal pose and deletes the whole arms-near-torso
  region. `--torso-shrink` (default -0.012 m) applies a negative
  `self_collision_buffer` in memory only; the asset is untouched. Measured
  against pinocchio MESH distances over 2000 accepted configurations:

      raw asset          accept 3.95 %   nominal REJECTED   mesh gap +10.79 mm
      shrink -12 mm      accept 7.25 %   nominal ok         mesh gap  +0.00 mm
      shrink -14 mm      accept 7.41 %   nominal ok         mesh gap  -1.47 mm
      ignore the pairs   accept 8.12 %   nominal ok         mesh gap  -8.32 mm

  -12 mm is the smallest shrink that admits the nominal pose and it still lets
  through ZERO real mesh penetration. cuRobo's own bundled G1 configs have the
  same defect (worse: their torso spheres protrude 145 mm max vs our 101).
* **Density is not the region.** The cloud is the push-forward of a uniform
  *joint* density, so it piles up mid-range and thins at the boundary. Read the
  voxelized cloud (`--voxel`, on by default), not point density.
* The scene has no floor and no wall: "collision free" here means
  self-collision only.
"""

from __future__ import annotations

import argparse
import copy
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

import assets
from rl.g1_assets.reach_region import command_mask  # noqa: F401  (shared cut logic)
from curobo._src.robot.loader.util import load_robot_yaml
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import ContentPath, JointState, Pose

REF_FRAME = "pelvis"
EE_FRAMES = ("left_tactile", "right_tactile")

EE_COLORS = {
    "left_tactile": (60, 140, 255),    # blue
    "right_tactile": (255, 120, 45),   # orange
}
TORSO_SPHERE_SHRINK = -0.012



def resolve_robot(name_or_path: str) -> str:
    """Absolute path to a cuRobo robot YAML.

    cuRobo resolves a RELATIVE config path against its own
    `content/configs/robot/`, so a repo-relative path silently 404s.  Accept
    either a bare asset filename (`g1_tactile.yml`) or any path, and hand
    cuRobo an absolute one.
    """
    candidate = assets.ASSET_DIR / name_or_path
    if candidate.exists():
        return str(candidate)
    return str(Path(name_or_path).expanduser().resolve())


def load_kinematics_cfg(
    yml_path: str,
    torso_shrink: float = TORSO_SPHERE_SHRINK,
    ee_frames: Sequence[str] = EE_FRAMES,
) -> dict:
    """Read the cuRobo robot YAML into the plain `kinematics` dict.

    Both the collision checker and the FK model are built from this ONE dict so
    they can never disagree about joint limits or the collision model.

    Args:
        yml_path: Path to the cuRobo robot YAML.
        torso_shrink: Self-collision buffer for `torso_link`, in m; negative
            shrinks its over-fitted spheres.  0 leaves the asset alone.
        ee_frames: End-effector frames to solve for.  Defaults to the tactile
            tips; pass e.g. `("left_wrist_yaw_link", "right_wrist_yaw_link")`
            for a wrist-frame consumer.  Must be paired with the same
            `frames` argument to `tip_poses_in_pelvis`.

    The YAML's `asset_root_path` / `urdf_path` are rebased onto the YAML's own
    directory: cuRobo writes them as absolute paths of the machine that
    generated the config, which do not exist on any other checkout or node.
    """
    config = load_robot_yaml(ContentPath(robot_config_file=yml_path))
    kinematics = copy.deepcopy(config["kinematics"])
    root = Path(yml_path).resolve().parent
    kinematics["asset_root_path"] = str(root)
    if kinematics.get("urdf_path"):
        kinematics["urdf_path"] = str(root / Path(kinematics["urdf_path"]).name)
    if torso_shrink:
        kinematics.setdefault("self_collision_buffer", {})["torso_link"] = torso_shrink
    kinematics["tool_frames"] = [REF_FRAME, *ee_frames]
    return kinematics


def build_collision_checker(kinematics: dict) -> RobotCollisionChecker:
    """Build the joint-limit + self-collision rejector."""
    return RobotCollisionChecker(
        RobotCollisionCheckerCfg.load_from_config({"kinematics": copy.deepcopy(kinematics)})
    )


def build_kinematics(kinematics: dict) -> Kinematics:
    """Build the FK model.

    The asset ships `tool_frames: [pelvis, torso_link, right_tactile,
    left_tactile]`; `load_kinematics_cfg` already replaced that with the two
    tips plus the reference frame, so FK returns nothing to filter downstream.
    """
    config = {"robot_cfg": {"kinematics": copy.deepcopy(kinematics)}}
    return Kinematics(KinematicsCfg.from_robot_yaml_file(config))



def build_sampler(col_checker: RobotCollisionChecker, seed: int) -> SampleBuffer:
    limits = col_checker.kinematics.get_joint_limits().position
    return SampleBuffer.create_halton_sample_buffer(
        ndims=col_checker.kinematics.get_dof(),
        up_bounds=limits[1],
        low_bounds=limits[0],
        store_buffer=None,
        seed=seed,
    )


def sample_valid_configs(
    col_checker: RobotCollisionChecker,
    sampler: SampleBuffer,
    num_samples: int,
    chunk: int = 200_000,
    max_rounds: int = 500,
) -> tuple[torch.Tensor, float]:
    valid: list[torch.Tensor] = []
    num_valid = 0
    drawn = 0
    for _ in range(max_rounds):
        if num_valid >= num_samples:
            break
        q = sampler.get_samples(chunk, bounded=True).contiguous()
        mask = col_checker.validate(q.unsqueeze(1)).view(-1)
        drawn += q.shape[0]
        keep = q[mask]
        num_valid += keep.shape[0]
        valid.append(keep)
    if num_valid == 0:
        raise RuntimeError(
            "no collision-free configuration found -- the self-collision model "
            "rejects everything; check the robot YAML"
        )
    if num_valid < num_samples:
        print(
            f"  WARNING: only {num_valid}/{num_samples} valid configs after "
            f"{max_rounds} rounds (acceptance {num_valid / max(drawn, 1) * 100:.2f}%)"
        )
    return torch.cat(valid, dim=0)[:num_samples], num_valid / max(drawn, 1)


def _tool_pose(state, frame: str) -> Pose:
    """Contiguous `Pose` for one tool frame (`get_link_pose` returns a view)."""
    pose = state.tool_poses.get_link_pose(frame)
    return Pose(
        position=pose.position.contiguous(),
        quaternion=pose.quaternion.contiguous(),
    )


def tip_poses_in_pelvis(
    robot: Kinematics,
    q: torch.Tensor,
    chunk: int = 100_000,
    frames: Sequence[str] = EE_FRAMES,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Batched FK, returned as EE poses expressed in the pelvis frame.

    `T_pelvis^-1 * T_ee`, so the result does not depend on where the
    synthesized floating base sits.

    Args:
        robot: FK model from `build_kinematics`.
        q: Joint configurations `[N, dof]`.
        chunk: FK batch size (bounds peak GPU memory).
        frames: EE frames to return; must be the ones the model was built
            with (see `load_kinematics_cfg`).

    Returns:
        `{frame: (position [N,3], quaternion wxyz [N,4])}`.
    """
    out: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {f: [] for f in frames}
    for start in range(0, q.shape[0], chunk):
        q_chunk = q[start : start + chunk].contiguous()
        state = robot.compute_kinematics(
            JointState.from_position(q_chunk, joint_names=robot.joint_names)
        )
        ref = _tool_pose(state, REF_FRAME)
        for frame in frames:
            local = ref.compute_local_pose(_tool_pose(state, frame))
            out[frame].append((local.position, local.quaternion))
    return {
        frame: (
            torch.cat([p for p, _ in parts], dim=0),
            torch.cat([qt for _, qt in parts], dim=0),
        )
        for frame, parts in out.items()
    }


def geometry_bounds(kinematics: dict) -> dict[str, float]:
    """Anatomy landmarks that define the command cuts, read off the model.

    Measured at the nominal configuration and returned in the PELVIS frame, so
    the cuts follow the asset instead of being hard-coded numbers:

    * `shoulder_y_{left,right}` -- the ARM ROOT (`*_shoulder_pitch_link`, the
      first joint of the arm chain), which `--y-reach` measures outward from.
    * `head_top_z` -- top of `head_link`'s collision spheres, the default
      ceiling for `--z-max`.

    Uses a throwaway `Kinematics` with its OWN tool frames: the sampling model's
    frame list stays pinned, so the joint order it was built with is untouched.
    """
    config = {"robot_cfg": {"kinematics": copy.deepcopy(kinematics)}}
    frames = [REF_FRAME, "left_shoulder_pitch_link", "right_shoulder_pitch_link", "head_link"]
    probe = Kinematics(KinematicsCfg.from_robot_yaml_file(config, tool_frames=frames))
    state = probe.compute_kinematics(
        JointState.from_position(
            torch.zeros(1, probe.get_dof(), device="cuda"), joint_names=probe.joint_names
        )
    )
    ref = _tool_pose(state, REF_FRAME)
    out = {}
    for side in ("left", "right"):
        local = ref.compute_local_pose(_tool_pose(state, f"{side}_shoulder_pitch_link"))
        out[f"shoulder_y_{side}"] = float(local.position[0, 1])
    head = ref.compute_local_pose(_tool_pose(state, "head_link"))
    R = head.get_rotation_matrix()[0]
    t = head.position[0]
    spheres = kinematics["collision_spheres"].get("head_link", [])
    if spheres:
        c = torch.tensor([s["center"] for s in spheres], device=R.device, dtype=R.dtype)
        rad = torch.tensor([s["radius"] for s in spheres], device=R.device, dtype=R.dtype)
        out["head_top_z"] = float(((c @ R.T + t)[:, 2] + rad).max())
    else:  # no head spheres in this config -- fall back to the link origin
        out["head_top_z"] = float(t[2])
    return out


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """One point per occupied `voxel`-sized cell.

    Turns the push-forward density (which piles up mid-range) into something
    that actually reads as a *region* -- and keeps the browser responsive.
    """
    if voxel <= 0 or points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def describe(frame: str, pos: np.ndarray) -> None:
    """Print the pelvis-frame extent of one hand's reachable set."""
    if pos.shape[0] == 0:
        print(f"  {frame:15s} EMPTY")
        return
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    radius = np.linalg.norm(pos, axis=1)
    print(
        f"  {frame:15s} n={pos.shape[0]:8d}  "
        f"x [{lo[0]:+.3f},{hi[0]:+.3f}]  "
        f"y [{lo[1]:+.3f},{hi[1]:+.3f}]  "
        f"z [{lo[2]:+.3f},{hi[2]:+.3f}]  "
        f"|p| max {radius.max():.3f}"
    )


def save_npz(path: Path, q: np.ndarray, tips: dict[str, dict[str, np.ndarray]]) -> None:
    """Store the configurations and the per-hand pelvis-frame tip poses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"q": q}
    for frame, data in tips.items():
        side = frame.split("_")[0]
        arrays[f"{side}_pos_pelvis"] = data["pos"]
        arrays[f"{side}_quat_pelvis"] = data["quat"]
        arrays[f"{side}_command_mask"] = data["mask"]
    np.savez_compressed(path, **arrays)
    print(f"\nSaved {path}")


def visualize(
    clouds: dict[str, np.ndarray],
    port: int,
    point_size: float,
    show_robot: bool = True,
) -> None:
    import viser

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/pelvis", axes_length=0.25, axes_radius=0.006)
    server.scene.add_grid("/grid", width=3.0, height=3.0, position=(0.0, 0.0, -0.793))

    if show_robot:
        try:
            import yourdfpy
            from viser.extras import ViserUrdf

            urdf = yourdfpy.URDF.load(
                assets.asset(assets.TACTILE_URDF),
                load_meshes=True,
                build_scene_graph=True,
                filename_handler=lambda fname: yourdfpy.filename_handler_magic(
                    fname, dir=str(assets.ASSET_DIR)
                ),
            )
            viser_urdf = ViserUrdf(
                server, urdf, root_node_name="/robot", mesh_color_override=(0.7, 0.7, 0.72)
            )
            viser_urdf.update_cfg(np.zeros(len(urdf.actuated_joints)))
        except Exception as exc:  # noqa: BLE001 - viz is best effort
            print(f"  (robot mesh not drawn: {exc})")

    handles = {}
    for frame, points in clouds.items():
        handles[frame] = server.scene.add_point_cloud(
            f"/reach/{frame}",
            points=points.astype(np.float32),
            colors=np.array(EE_COLORS[frame], dtype=np.uint8),
            point_size=point_size,
            point_shape="circle",
            precision="float32",
        )

    with server.gui.add_folder("Reach cloud"):
        size = server.gui.add_slider("point size", 0.002, 0.03, 0.001, point_size)

        @size.on_update
        def _(_) -> None:
            for handle in handles.values():
                handle.point_size = size.value

        for frame in clouds:
            box = server.gui.add_checkbox(frame, True)

            @box.on_update
            def _(_, frame=frame, box=box) -> None:
                handles[frame].visible = box.value

    
    print(f"\nviser serving on http://localhost:{server.get_port()}  (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopped")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample the tactile-EE reachable region in the pelvis frame (cuRobo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--robot", type=str, default=assets.TACTILE_CUROBO_FIXED_YML,
        help="cuRobo robot YAML: an asset filename or a path "
             f"(default: {assets.TACTILE_CUROBO_FIXED_YML}, arms only)",
    )
    parser.add_argument(
        "--num", type=int, default=100_000,
        help="Number of collision-free configurations to sample (default: 100000)",
    )
    parser.add_argument(
        "--x-margin", type=float, default=0.1,
        help="Keep tips with pelvis-frame x > this, in m (default: 0.0)",
    )
    parser.add_argument(
        "--y-reach", type=float, default=0.3,
        help="Max OUTBOARD reach in y measured from that arm's shoulder, in m; "
             "inboard is unrestricted. Negative disables (default: 0.4)",
    )
    parser.add_argument(
        "--y-margin", type=float, default=0.05,
        help="Min INBOARD distance from the pelvis mid-plane, in m: left tips keep "
             "y >= +margin, right tips y <= -margin, so the hands can never cross. "
             "0 or negative disables (default: 0.05)",
    )
    parser.add_argument(
        "--z-max", type=float, default=0.3,
        help="Ceiling on pelvis-frame z, in m (default: top of head_link)",
    )
    parser.add_argument(
        "--z-min", type=float, default=0.1,
        help="Floor on pelvis-frame z, in m; may be negative (default: unrestricted)",
    )
    parser.add_argument(
        "--torso-shrink", type=float, default=TORSO_SPHERE_SHRINK,
        help="torso_link self-collision buffer in m, negative shrinks its over-fitted "
             f"spheres; 0 = raw asset (default: {TORSO_SPHERE_SHRINK})",
    )
    parser.add_argument(
        "--voxel", type=float, default=0.01,
        help="Voxel size for the displayed cloud, m; 0 disables (default: 0.01)",
    )
    parser.add_argument("--seed", type=int, default=0, help="torch RNG seed (default: 0)")
    parser.add_argument("--out", type=str, default=None, help="Write the poses to this .npz")
    parser.add_argument("--visualize", action="store_true", help="Serve the clouds in viser")
    parser.add_argument("--no-robot", dest="show_robot", action="store_false",
                        help="Do not draw the robot mesh in viser")
    parser.add_argument("--point-size", type=float, default=0.006, help="viser point size (m)")
    parser.add_argument("--port", type=int, default=8080, help="viser port (default: 8080)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    robot_yml = resolve_robot(args.robot)
    print(f"Robot config : {robot_yml}")
    kinematics = load_kinematics_cfg(robot_yml, torso_shrink=args.torso_shrink)
    col_checker = build_collision_checker(kinematics)
    robot = build_kinematics(kinematics)
    assert robot.joint_names == col_checker.kinematics.joint_names, (
        "FK and collision models disagree on joint order"
    )
    print(f"DoF          : {robot.get_dof()}  {robot.joint_names}")
    print(f"Tool frames  : {robot.tool_frames}   (reference = {REF_FRAME})")
    print(f"torso_link sphere buffer : {args.torso_shrink:+.3f} m")
    bounds = geometry_bounds(kinematics)

    print(f"\nSampling {args.num} collision-free configurations ...")
    t0 = time.perf_counter()
    sampler = build_sampler(col_checker, seed=args.seed)
    q, acceptance = sample_valid_configs(col_checker, sampler, args.num)
    torch.cuda.synchronize()
    print(f"  {q.shape[0]} configs in {time.perf_counter() - t0:.2f} s "
          f"(acceptance {acceptance * 100:.2f}%)")

    print("Running batched FK ...")
    t0 = time.perf_counter()
    poses = tip_poses_in_pelvis(robot, q)
    torch.cuda.synchronize()
    print(f"  {(time.perf_counter() - t0) * 1e3:.1f} ms")

    y_reach = None if args.y_reach < 0 else args.y_reach
    y_margin = None if args.y_margin <= 0 else args.y_margin
    z_max = args.z_max if args.z_max is not None else bounds["head_top_z"]
    z_min = args.z_min
    print("\nCommand cuts (pelvis frame):")
    print(f"  x     > {args.x_margin:+.3f} m")
    if y_reach is None:
        print("  y_out unrestricted")
    else:
        print(f"  y_out outboard <= {y_reach:.3f} m from the shoulder  -> "
              f"left <= {bounds['shoulder_y_left'] + y_reach:+.4f}, "
              f"right >= {bounds['shoulder_y_right'] - y_reach:+.4f}")
    if y_margin is None:
        print("  y_in  unrestricted (hands may cross)")
    else:
        print(f"  y_in  inboard >= {y_margin:.3f} m from the mid-plane  -> "
              f"left >= {y_margin:+.4f}, right <= {-y_margin:+.4f}")
    print(f"  z_max <= {z_max:+.4f} m"
          + ("  (top of head_link)" if args.z_max is None else ""))
    print(f"  z_min >= {z_min:+.4f} m" if z_min is not None else "  z_min unrestricted")

    tips: dict[str, dict[str, np.ndarray]] = {}
    clouds: dict[str, np.ndarray] = {}
    print("\nWhat each cut costs:")
    for frame in EE_FRAMES:
        pos, quat = poses[frame]
        masks = command_mask(
            pos, frame, bounds, args.x_margin, y_reach, z_max,
            y_margin=y_margin, z_min=z_min,
        )
        keep = torch.ones_like(masks["x"])
        parts = []
        for name, m in masks.items():
            before = int(keep.sum())
            keep = keep & m
            after = int(keep.sum())
            parts.append(f"{name} -{before - after}")
        tips[frame] = {
            "pos": pos.cpu().numpy(),
            "quat": quat.cpu().numpy(),
            "mask": keep.cpu().numpy(),
        }
        kept = tips[frame]["pos"][tips[frame]["mask"]]
        print(f"  {frame:15s} {pos.shape[0]} -> {kept.shape[0]} "
              f"({kept.shape[0] / pos.shape[0] * 100:.1f}%)   [{'  '.join(parts)}]")
        clouds[frame] = voxel_downsample(kept, args.voxel)

    print("\nReachable extent after the cuts:")
    for frame in EE_FRAMES:
        describe(frame, tips[frame]["pos"][tips[frame]["mask"]])
    if args.voxel > 0:
        print(f"\nVoxelized to {args.voxel * 1000:.0f} mm: "
              + ", ".join(f"{f} {c.shape[0]}" for f, c in clouds.items()))

    if args.out:
        save_npz(Path(args.out), q.cpu().numpy(), tips)

    if args.visualize:
        visualize(clouds, args.port, args.point_size, show_robot=args.show_robot)


if __name__ == "__main__":
    main()
