"""PyRoki model of the G1 arms rooted at the pelvis: forward kinematics of the
tactile tips, capsule self-collision and the joint bookkeeping the per-hand IK
solves need. Built from the `robot:` section of `ee_qpos.yaml`. No costs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# JAX shares the GPU with torch/mjwarp elsewhere: allocate on demand instead of
# grabbing 75 % of the card at first use (no effect if JAX is already up).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jaxls
import numpy as np
import pyroki as pk
import yaml
import yourdfpy
from pyroki.collision import RobotCollision

import assets

SIDES = ("left", "right")
SIDE_SIGN = {"left": 1.0, "right": -1.0}
"""Sign of each arm's outward lateral direction along +y."""
ARM_TOKENS = ("shoulder", "elbow", "wrist", "tactile")
"""Name tokens of the links that belong to an arm chain."""
ELBOW = 3
"""Column of the elbow joint inside a hand's 7-vector."""
CFG_PATH = Path(__file__).with_name("ee_qpos.yaml")
"""The pipeline's default config file."""


def load_cfg(path: str | Path = CFG_PATH) -> dict:
    """The whole `ee_qpos.yaml` as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class RobotCfg:
    """The `robot:` section of `ee_qpos.yaml`.

    Attributes:
        urdf: URDF file name under `assets.ASSET_DIR`.
        root: Root link; the model's FK frame.
        ee: Side -> tip link name.
        arm_joints: Side -> the 7 free joints of that hand's solve, model order.
        rest: Joint -> rad for the locked posture and the arms' rest; unlisted 0.
        ignore_within_arm: Never check pairs among one arm's own links.
        ignore_pairs: Extra link pairs never checked.
        link_buffer: Link -> allowed penetration, m (negative = shrunk).
        nominal_margin: Clearance the rest posture must have on every active pair, m.
    """

    urdf: str
    root: str
    ee: dict[str, str]
    arm_joints: dict[str, tuple[str, ...]]
    rest: dict[str, float]
    ignore_within_arm: bool
    ignore_pairs: tuple[tuple[str, str], ...]
    link_buffer: dict[str, float]
    nominal_margin: float

    @classmethod
    def from_dict(cls, d: dict) -> "RobotCfg":
        """Build from the `robot:` mapping of the yaml."""
        col = d.get("collision", {})
        return cls(
            urdf=d["urdf"],
            root=d["root"],
            ee={s: d["ee"][s] for s in SIDES},
            arm_joints={s: tuple(d["arm_joints"][s]) for s in SIDES},
            rest={k: float(v) for k, v in (d.get("rest") or {}).items()},
            ignore_within_arm=bool(col.get("ignore_within_arm", True)),
            ignore_pairs=tuple((a, b) for a, b in (col.get("ignore_pairs") or [])),
            link_buffer={k: float(v) for k, v in (col.get("link_buffer") or {}).items()},
            nominal_margin=float(col.get("nominal_margin", 0.005)),
        )


def load_urdf(urdf: str, meshes: bool) -> yourdfpy.URDF:
    """The URDF with collision geometry loaded; `meshes=True` also loads the
    visual meshes (viewer use)."""
    return yourdfpy.URDF.load(
        assets.asset(urdf),
        load_meshes=meshes,
        load_collision_meshes=True,
        build_scene_graph=True,
        build_collision_scene_graph=True,
        filename_handler=lambda fname: yourdfpy.filename_handler_magic(
            fname, dir=str(assets.ASSET_DIR)
        ),
    )


@dataclass(frozen=True)
class ArmModel:
    """Fixed-base G1 rooted at the pelvis with legs and waist locked; each hand's
    solve moves its own 7 arm joints while the other arm rests.

    Attributes:
        cfg: The robot config this model was built from.
        robot: PyRoki kinematics; FK is in the pelvis frame.
        coll: Capsule self-collision model with the ignore rules applied.
        joint_names: All actuated joints, model order. Shape (29,).
        rest_cfg: The rest configuration. Shape (29,), rad.
        arm_idx: Side -> model indices of that hand's joints. Shape (7,).
        lower, upper: Side -> position limits of that hand's joints. Shape (7,), rad.
        links: Side -> link indices `tip`, `shoulder`, `elbow`, `wrist`.
        pair_floor: Signed-distance floor per active collision pair, m. Shape (P,).
        pair_names: Link names of the active pairs. Length P.
        var_cls: Side -> 7-dim jaxls variable class whose default is that arm's rest.
    """

    cfg: RobotCfg
    robot: pk.Robot
    coll: RobotCollision
    joint_names: tuple[str, ...]
    rest_cfg: np.ndarray
    arm_idx: dict[str, np.ndarray]
    lower: dict[str, np.ndarray]
    upper: dict[str, np.ndarray]
    links: dict[str, dict[str, int]]
    pair_floor: np.ndarray
    pair_names: tuple[tuple[str, str], ...]
    var_cls: dict[str, type]

    @staticmethod
    def load(cfg: RobotCfg) -> "ArmModel":
        """Build the model.

        Raises:
            ValueError: A joint or link of the config is missing, the arm joints
                are not in model order, or the rest posture does not clear every
                active collision pair by `cfg.nominal_margin`.
        """
        u = load_urdf(cfg.urdf, meshes=False)
        if u.base_link != cfg.root:
            raise ValueError(f"URDF root is {u.base_link}, config says {cfg.root}")
        robot = pk.Robot.from_urdf(u)
        names = tuple(robot.joints.actuated_names)
        link_names = list(robot.links.names)

        rest = np.zeros(len(names), dtype=np.float32)
        for k, v in cfg.rest.items():
            if k not in names:
                raise ValueError(f"rest joint {k} is not an actuated joint")
            rest[names.index(k)] = v

        lo_all = np.asarray(robot.joints.lower_limits, dtype=np.float32)
        hi_all = np.asarray(robot.joints.upper_limits, dtype=np.float32)
        arm_idx, lower, upper, links = {}, {}, {}, {}
        for s in SIDES:
            missing = [j for j in cfg.arm_joints[s] if j not in names]
            if missing or len(cfg.arm_joints[s]) != 7:
                raise ValueError(f"{s}: arm_joints must be 7 actuated joints, missing {missing}")
            idx = np.array([names.index(j) for j in cfg.arm_joints[s]], dtype=np.int32)
            if not np.all(np.diff(idx) > 0):
                raise ValueError(f"{s}: arm_joints are not in model order")
            if "elbow" not in cfg.arm_joints[s][ELBOW]:
                raise ValueError(f"{s}: arm_joints[{ELBOW}] must be the elbow")
            arm_idx[s], lower[s], upper[s] = idx, lo_all[idx], hi_all[idx]
            wanted = {
                "tip": cfg.ee[s],
                "shoulder": f"{s}_shoulder_pitch_link",
                "elbow": f"{s}_elbow_link",
                "wrist": f"{s}_wrist_roll_link",
            }
            for k, ln in wanted.items():
                if ln not in link_names:
                    raise ValueError(f"{s}: link {ln} ({k}) not in the URDF")
            links[s] = {k: link_names.index(ln) for k, ln in wanted.items()}

        is_arm = {n: any(t in n for t in ARM_TOKENS) for n in link_names}
        ignore: set[tuple[str, str]] = set()

        def ignore_pair(a: str, b: str) -> None:
            ignore.add((a, b))
            ignore.add((b, a))

        for a, b in cfg.ignore_pairs:
            if a not in link_names or b not in link_names:
                raise ValueError(f"ignore pair ({a}, {b}) names an unknown link")
            ignore_pair(a, b)
        if cfg.ignore_within_arm:
            for s in SIDES:
                chain = [n for n in link_names if n.startswith(s + "_") and is_arm[n]]
                for a in chain:
                    for b in chain:
                        if a != b:
                            ignore_pair(a, b)
        no_geom = [n for n in link_names if not u.link_map[n].collisions]
        for a in link_names:
            for b in link_names:
                if a != b and (a in no_geom or b in no_geom or not (is_arm[a] or is_arm[b])):
                    ignore_pair(a, b)
        coll = RobotCollision.from_urdf(u, user_ignore_pairs=tuple(sorted(ignore)))
        cl = list(coll.link_names)
        pairs = tuple((cl[i], cl[j]) for i, j in zip(coll.active_idx_i, coll.active_idx_j))
        floor = np.array(
            [cfg.link_buffer.get(a, 0.0) + cfg.link_buffer.get(b, 0.0) for a, b in pairs],
            dtype=np.float32,
        )
        d_rest = np.asarray(
            coll.compute_self_collision_distance(robot, jnp.asarray(rest)[None, :])
        )[0]
        bad = np.nonzero(d_rest - floor < cfg.nominal_margin)[0]
        if bad.size:
            rows = "\n".join(
                f"  {pairs[k][0]} x {pairs[k][1]}: d {d_rest[k] * 1e3:+.1f} mm, floor {floor[k] * 1e3:+.1f} mm"
                for k in bad
            )
            raise ValueError(
                f"rest posture is within {cfg.nominal_margin * 1e3:.0f} mm of its floor on "
                f"{bad.size} pair(s); ignore them or give a link_buffer:\n{rows}"
            )

        var_cls = {}
        for s in SIDES:
            q_rest7 = jnp.asarray(rest[arm_idx[s]])

            class HandVar(jaxls.Var[jax.Array], default_factory=lambda q=q_rest7: q):
                ...

            HandVar.__name__ = f"HandVar_{s}"
            var_cls[s] = HandVar

        return ArmModel(
            cfg=cfg, robot=robot, coll=coll, joint_names=names, rest_cfg=rest,
            arm_idx=arm_idx, lower=lower, upper=upper, links=links,
            pair_floor=floor, pair_names=pairs, var_cls=var_cls,
        )

    # -- configurations -----------------------------------------------------

    def rest7(self, side: str) -> np.ndarray:
        """That hand's rest posture. Shape (7,), rad."""
        return self.rest_cfg[self.arm_idx[side]]

    def full(self, side: str, q7: jax.Array) -> jax.Array:
        """Full configuration with that hand's joints set and everything else
        at rest. Shape (..., 29)."""
        rest = jnp.broadcast_to(jnp.asarray(self.rest_cfg), (*q7.shape[:-1], self.rest_cfg.shape[0]))
        return rest.at[..., self.arm_idx[side]].set(q7)

    def full_both(self, q7_left: jax.Array, q7_right: jax.Array) -> jax.Array:
        """Full configuration with both hands set. Shape (..., 29)."""
        q = self.full("left", q7_left)
        return q.at[..., self.arm_idx["right"]].set(q7_right)

    # -- kinematics (pelvis frame, wxyz quaternions) ------------------------

    def fk(self, q29: jax.Array) -> jax.Array:
        """Link poses. Shape (..., links, 7) as wxyz_xyz."""
        return self.robot.forward_kinematics(q29)

    def tip(self, side: str, q7: jax.Array) -> tuple[jax.Array, jax.Array]:
        """That hand's tip pose: position (..., 3) and quaternion (..., 4) wxyz."""
        T = self.fk(self.full(side, q7))[..., self.links[side]["tip"], :]
        return T[..., 4:7], T[..., 0:4]

    def arm_points(self, side: str, q7: jax.Array) -> dict[str, jax.Array]:
        """Positions of that arm's `shoulder`, `elbow`, `wrist` and `tip`. Shape (..., 3) each."""
        T = self.fk(self.full(side, q7))
        return {k: T[..., i, 4:7] for k, i in self.links[side].items()}

    def clearance(self, side: str, q7: jax.Array) -> jax.Array:
        """Signed distance minus floor for every active pair, m; >= 0 is clear.
        Shape (..., P). The other arm is at rest."""
        return self.coll.compute_self_collision_distance(self.robot, self.full(side, q7)) - jnp.asarray(self.pair_floor)

    def nominal_target(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Tip position (3,) m and orientation (4,) wxyz at the rest posture, pelvis frame."""
        pos, quat = self.tip(side, jnp.asarray(self.rest7(side)))
        return np.asarray(pos), np.asarray(quat)

    def shoulder(self, side: str) -> np.ndarray:
        """That arm's shoulder position at rest, pelvis frame. Shape (3,), m."""
        return np.asarray(self.arm_points(side, jnp.asarray(self.rest7(side)))["shoulder"], dtype=np.float64)

    def capsules(self, q29: np.ndarray):
        """Collision capsules at a configuration as one trimesh (viewer use)."""
        return self.coll.at_config(self.robot, jnp.asarray(q29)).to_trimesh()


def main() -> None:
    cfg = RobotCfg.from_dict(load_cfg()["robot"])
    m = ArmModel.load(cfg)
    print(f"joints: {len(m.joint_names)}  active collision pairs: {len(m.pair_names)}")
    for s in SIDES:
        pos, quat = m.nominal_target(s)
        print(f"{s}: joints {m.arm_idx[s].tolist()}  nominal tip {np.round(pos, 4)}  wxyz {np.round(quat, 3)}  "
              f"shoulder {np.round(m.shoulder(s), 3)}")
        c = np.asarray(m.clearance(s, jnp.asarray(m.rest7(s))))
        k = int(np.argmin(c))
        print(f"  tightest pair at rest: {m.pair_names[k]} {c[k] * 1e3:+.1f} mm")


if __name__ == "__main__":
    main()
