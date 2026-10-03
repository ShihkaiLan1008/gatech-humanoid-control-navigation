"""Browser playback of joint trajectories through mjviser on the standalone
playback scene (`assets.TACTILE_SCENE_XML`): `play_solution` replays a Crocoddyl
solution (pinocchio joint order, mapped to MuJoCo qpos by joint name),
`play_qpos` replays a MuJoCo qpos trajectory; both block until Ctrl-C and print
the port to forward. `disable_mesh_lod` / `aim_camera` are the two mjviser
scene fixes every viser player in the repo reuses.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np

import assets

ASSET_DIR = str(assets.ASSET_DIR)
TACTILE_XML = assets.TACTILE_SCENE_XML

# 3/4 front view of a ~1.3 m humanoid at the origin (same as mpc_online's viewer)
_CAM_POS = (2.0, -2.0, 1.4)
_CAM_LOOK = (0.0, 0.0, 0.7)


def _qpos_map(pin_model, mj_model):
    """Per-joint (pin idx_q -> mj qpos_adr) for the 1-DoF joints, matched BY NAME
    (robust to ordering; the two G1 files agree anyway)."""
    import mujoco

    pairs = []
    for j in range(mj_model.njnt):
        if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is None or not pin_model.existJointName(name):
            raise KeyError(f"MuJoCo joint '{name}' not in the pinocchio model")
        pairs.append((pin_model.joints[pin_model.getJointId(name)].idx_q,
                      int(mj_model.jnt_qposadr[j])))
    return pairs


def _to_qpos(q_pin, pairs, nq_mj):
    """Pinocchio free-flyer q (quat xyzw at [3:7]) -> MuJoCo qpos (quat wxyz)."""
    qpos = np.zeros(nq_mj)
    qpos[0:3] = q_pin[0:3]
    qpos[3] = q_pin[6]                  # w
    qpos[4:7] = q_pin[3:6]              # x y z
    for iq, adr in pairs:
        qpos[adr] = q_pin[iq]
    return qpos


def play_solution(robot, ddp_or_qs, dt=0.02, wall=None, xml=None,
                  host="0.0.0.0", port=8080):
    """Serve the solution in a browser and loop it. `ddp_or_qs` is a solved
    crocoddyl solver (uses .xs) or an (F, nq) array of configurations. `wall` =
    dict(axis, side, d[, z]): draws the wall plane the problem pressed against
    (in the clip's world frame). Blocks until interrupted."""
    import mujoco
    import viser
    from mjviser import Viewer

    qs = (np.array([np.asarray(x)[: robot.model.nq] for x in ddp_or_qs.xs])
          if hasattr(ddp_or_qs, "xs") else np.asarray(ddp_or_qs))

    model = mujoco.MjModel.from_xml_path(os.path.join(ASSET_DIR, xml or TACTILE_XML))
    model.opt.timestep = float(dt)      # mjviser paces the loop by this
    data = mujoco.MjData(model)
    pairs = _qpos_map(robot.model, model)

    state = {"k": 0}

    def step(m, d):                     # kinematic playback: one clip frame per step
        d.qpos[:] = _to_qpos(qs[state["k"]], pairs, m.nq)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        state["k"] = (state["k"] + 1) % len(qs)

    def reset(m, d):
        state["k"] = 0
        step(m, d)

    server = viser.ViserServer(host=host, port=port)
    _disable_mesh_lod(server)
    _aim_camera(server)
    reset(model, data)
    viewer = Viewer(model, data, step_fn=step, reset_fn=reset, server=server)
    if wall is not None:
        # AFTER the Viewer: mjviser's "Track camera" mode keeps the robot
        # centered by translating the WHOLE SCENE by -pelvis_pos (it moves the
        # /fixed_bodies frame, not the camera). The wall must live UNDER that
        # frame to ride the same offset as the floor -- added at the root it
        # stays screen-fixed and appears glued to the walking robot.
        _add_wall(server, **wall)
    print(f"[viser_play] {len(qs)} frames @ {1.0/dt:.0f} Hz -- "
          f"serving on http://localhost:{port}  (Ctrl-C to stop)", flush=True)
    viewer.run()


def play_qpos(model, qpos_traj, dt, host="0.0.0.0", port=8080):
    """Serve a MuJoCo-qpos trajectory on an EXISTING model in the browser
    (play_clip --viser). Any wall/floor geoms already in the model render as
    fixed geometry and ride mjviser's camera-tracking scene offset correctly --
    no custom viser props needed. Blocks until Ctrl-C."""
    import mujoco
    import viser
    from mjviser import Viewer

    qs = np.asarray(qpos_traj)
    model.opt.timestep = float(dt)      # mjviser paces the loop by this
    data = mujoco.MjData(model)
    state = {"k": 0}

    def step(m, d):
        d.qpos[:] = qs[state["k"]]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        state["k"] = (state["k"] + 1) % len(qs)

    def reset(m, d):
        state["k"] = 0
        step(m, d)

    server = viser.ViserServer(host=host, port=port)
    _disable_mesh_lod(server)
    _aim_camera(server)
    reset(model, data)
    print(f"[viser_play] {len(qs)} frames @ {1.0/dt:.0f} Hz -- "
          f"serving on http://localhost:{port}  (Ctrl-C to stop)", flush=True)
    Viewer(model, data, step_fn=step, reset_fn=reset, server=server).run()


def _add_wall(server, axis, side, d, z=0.9, length=4.0, height=2.0, thick=0.02):
    """Translucent wall slab whose INNER face lies on the plane {p: p[axis]=side*d},
    centered on the walk tangent at press height. Named UNDER /fixed_bodies so it
    inherits mjviser's camera-tracking scene offset (see play_solution)."""
    dims = [length, length, height]
    dims[axis] = thick
    pos = [0.0, 0.0, min(max(z, height / 2), height / 2 + 0.3)]
    pos[axis] = side * (d + thick / 2)
    server.scene.add_box("/fixed_bodies/wall", dimensions=tuple(dims),
                         position=tuple(pos), color=(140, 160, 180), opacity=0.35)


# -- same two fixes as src/mpc_online/viser_viewer.py (kept local: the packages
#    don't import each other) ---------------------------------------------------
#
# Both were paid for with a debugging session, so anything else in the repo that
# stands up an mjviser scene should REUSE them rather than re-derive them --
# `retarget_motion.viser_play` imports the two public aliases at the bottom.
def _disable_mesh_lod(server) -> None:
    """G1's STLs are dense; mjviser's distance LOD turns them into blobs."""
    orig = server.scene.add_batched_meshes_trimesh

    def no_lod(*args, **kwargs):
        kwargs["lod"] = "off"
        return orig(*args, **kwargs)

    server.scene.add_batched_meshes_trimesh = no_lod


def _aim_camera(server) -> None:
    """Re-aim after mjviser's own on-connect handler (which parks too far out)."""

    @server.on_client_connect
    def _(client) -> None:
        def aim() -> None:
            time.sleep(1.5)
            client.camera.position = _CAM_POS
            client.camera.look_at = _CAM_LOOK

        threading.Thread(target=aim, daemon=True).start()


# Public aliases: the two fixes above are generic mjviser/viser workarounds, not
# Crocoddyl-specific, and other packages need them to stand up a working scene.
disable_mesh_lod = _disable_mesh_lod
aim_camera = _aim_camera
