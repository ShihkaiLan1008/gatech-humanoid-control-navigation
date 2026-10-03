"""IK cost design for one hand: the residual blocks, the batched jaxls solver,
per-sample diagnostics and the acceptance gates. Weights and margins travel as
one runtime vector (`CostCfg.theta`), so changing them never recompiles."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from functools import partial

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
from pyroki.collision import colldist_from_sdf

from .model import ELBOW, SIDE_SIGN, ArmModel

THETA_NAMES = (
    "pos", "ori", "limit", "limit_margin", "elbow", "elbow_max", "rest", "coll",
    "coll_activation", "swivel", "swivel_far", "swivel_near", "swivel_r_far",
    "swivel_r_near", "swivel_band",
)
"""Entries of the runtime vector, in order: the block weights plus the margins
`limit_margin` (rad), `elbow_max` (rad), `coll_activation` (m) and the swivel
target ramp (rad, rad, m, m, rad)."""
BLOCKS = ("pos", "ori", "limit", "elbow", "rest", "coll", "swivel")
"""Residual blocks, in the order they are concatenated."""
REASONS = ("pos", "ori", "coll", "limit", "elbow")
"""Rejection reasons; bit i of a reason mask is `REASONS[i]`."""


def _from_dict(cls, d: dict | None, section: str):
    d = dict(d or {})
    unknown = set(d) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {section} keys {sorted(unknown)}")
    return replace(cls(), **d)


@dataclass
class CostCfg:
    """Weights and margins of the residual blocks plus the solver settings.

    Every float except the two solver settings is a runtime value (see
    `theta`); `max_iterations` and `lambda_initial` are compiled in.
    """

    pos: float = 50.0
    """Tip position residual weight, per m."""
    ori: float = 10.0
    """Tip orientation residual weight, per rad of the rotation-error log."""
    limit: float = 100.0
    """Joint-limit hinge weight, per rad of violation of the margin."""
    limit_margin: float = 0.05
    """Distance kept inside every joint limit, rad."""
    elbow: float = 100.0
    """Elbow upper-bound hinge weight, per rad."""
    elbow_max: float = 1.5
    """Elbow angle bound, rad; +1.75 is a straight arm in this URDF."""
    rest: float = 0.1
    """Pull toward the rest posture, per rad."""
    coll: float = 100.0
    """Self-collision weight, per m inside the activation band."""
    coll_activation: float = 0.01
    """Clearance below which the collision residual switches on, m."""
    swivel: float = 5.0
    """Elbow swivel hinge weight, per rad outside the band about the target."""
    swivel_far: float = 0.52
    """Swivel target psi for targets at least `swivel_r_far` from the shoulder
    axis, rad: 0 = elbow straight below the shoulder-wrist line, positive =
    swung outward (away from the mid-plane), pi = above."""
    swivel_near: float = 2.62
    """Swivel target psi for targets within `swivel_r_near`, rad; linear between."""
    swivel_r_far: float = 0.35
    """Horizontal distance of the TARGET from the shoulder axis, m, at and
    beyond which `swivel_far` applies."""
    swivel_r_near: float = 0.15
    """Horizontal target distance, m, at and inside which `swivel_near` applies."""
    swivel_band: float = 0.52
    """Half-width of the zero-cost band about the swivel target, rad."""
    max_iterations: int = 50
    """Levenberg-Marquardt iteration cap per solve."""
    lambda_initial: float = 1.0
    """Initial Levenberg-Marquardt damping."""

    @classmethod
    def from_dict(cls, d: dict | None) -> "CostCfg":
        """Defaults overridden by the yaml `cost:` mapping; unknown keys raise."""
        return _from_dict(cls, d, "cost")

    def theta(self) -> np.ndarray:
        """Runtime vector in `THETA_NAMES` order. Shape (15,), float32."""
        return np.array([getattr(self, n) for n in THETA_NAMES], dtype=np.float32)


@dataclass
class GateCfg:
    """Acceptance thresholds (yaml `gates:`)."""

    pos_tol: float = 0.005
    """Max tip position error, m."""
    ori_tol: float = 3.0
    """Max tip orientation error (rotation angle), deg."""
    elbow_slack: float = 0.05
    """Allowed excess over `CostCfg.elbow_max`, rad."""
    smooth_tol: float = 0.5
    """Max joint distance to the nearest accepted lattice neighbour, rad."""
    max_retries: int = 2
    """Re-solves of a failed cell from other accepted neighbours."""

    @classmethod
    def from_dict(cls, d: dict | None) -> "GateCfg":
        """Defaults overridden by the yaml `gates:` mapping; unknown keys raise."""
        return _from_dict(cls, d, "gate")


def block_slices(model: ArmModel) -> dict[str, slice]:
    """Where each `BLOCKS` entry sits in the concatenated residual."""
    sizes = {
        "pos": 3, "ori": 3, "limit": 14, "elbow": 1, "rest": 7,
        "coll": len(model.pair_names), "swivel": 1,
    }
    out, start = {}, 0
    for b in BLOCKS:
        out[b] = slice(start, start + sizes[b])
        start += sizes[b]
    return out


def swivel_angle(ps: jax.Array, pe: jax.Array, pw: jax.Array, sign: float) -> jax.Array:
    """SEW elbow swivel psi, rad: the angle of the elbow's half-plane about the
    shoulder->wrist axis, measured from the half-plane containing gravity.
    psi = 0: elbow straight below the shoulder-wrist line; psi > 0: swung
    outward (`sign` = that arm's outward y direction); pi: above. Singular for
    a vertical shoulder-wrist axis or a straight arm.

    Args:
        ps, pe, pw: Shoulder, elbow, wrist positions, pelvis frame, m. Shape (3,).
    """
    a = pw - ps
    a = a / (jnp.linalg.norm(a) + 1e-9)
    down = jnp.array([0.0, 0.0, -1.0])
    r0 = down - jnp.dot(down, a) * a
    r0 = r0 / (jnp.linalg.norm(r0) + 1e-9)
    v = pe - ps
    v = v - jnp.dot(v, a) * a
    return jnp.arctan2(sign * jnp.dot(v, jnp.cross(a, r0)), jnp.dot(v, r0))


def swivel_target(th: dict, target_pos: jax.Array, ps: jax.Array) -> jax.Array:
    """Swivel target psi for a tip target, rad: the `th` ramp over the target's
    horizontal distance from the shoulder axis (`ps` = shoulder position, m)."""
    r = jnp.linalg.norm((target_pos - ps)[:2])
    span = jnp.maximum(th["swivel_r_far"] - th["swivel_r_near"], 1e-6)
    t = jnp.clip((r - th["swivel_r_near"]) / span, 0.0, 1.0)
    return th["swivel_near"] + t * (th["swivel_far"] - th["swivel_near"])


def manipulability(model: ArmModel, side: str, q7: jax.Array) -> jax.Array:
    """Yoshikawa measure sqrt(det(J J^T)) of the tip-position Jacobian (3x7)."""
    J = jax.jacfwd(lambda q: model.tip(side, q)[0])(q7)
    return jnp.sqrt(jnp.maximum(jnp.linalg.det(J @ J.T), 0.0))


def residual(
    model: ArmModel, side: str, q7: jax.Array, target_pos: jax.Array,
    target_quat: jax.Array, theta: jax.Array,
) -> jax.Array:
    """Concatenated residual of all `BLOCKS` for one hand. Shape (R,).

    Args:
        q7: Hand joints, rad. target_pos: Tip target, pelvis frame, m.
        target_quat: Tip orientation target, pelvis frame, wxyz.
        theta: `CostCfg.theta()`.
    """
    th = dict(zip(THETA_NAMES, theta))
    L = model.links[side]
    T = model.fk(model.full(side, q7))
    tip, ps = T[L["tip"]], T[L["shoulder"], 4:7]
    r_pos = (tip[4:7] - target_pos) * th["pos"]
    r_ori = (jaxlie.SO3(target_quat).inverse() @ jaxlie.SO3(tip[0:4])).log() * th["ori"]
    lo, hi, m = jnp.asarray(model.lower[side]), jnp.asarray(model.upper[side]), th["limit_margin"]
    r_lim = jnp.concatenate([jnp.maximum(0.0, lo + m - q7), jnp.maximum(0.0, q7 - hi + m)]) * th["limit"]
    r_elb = jnp.maximum(0.0, q7[ELBOW] - th["elbow_max"])[None] * th["elbow"]
    r_rest = (q7 - jnp.asarray(model.rest7(side))) * th["rest"]
    r_coll = -colldist_from_sdf(model.clearance(side, q7), th["coll_activation"]) * th["coll"]
    psi = swivel_angle(ps, T[L["elbow"], 4:7], T[L["wrist"], 4:7], SIDE_SIGN[side])
    dpsi = psi - swivel_target(th, target_pos, ps)
    dpsi = jnp.arctan2(jnp.sin(dpsi), jnp.cos(dpsi))
    r_swivel = (dpsi - jnp.clip(dpsi, -th["swivel_band"], th["swivel_band"]))[None] * th["swivel"]
    return jnp.concatenate([r_pos, r_ori, r_lim, r_elb, r_rest, r_coll, r_swivel])


def _solve_one(
    model: ArmModel, side: str, cfg: CostCfg, target_pos: jax.Array,
    target_quat: jax.Array, q0: jax.Array, theta: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """One hand, one target: solution (7,) and the iteration count."""
    var = model.var_cls[side](0)

    @jaxls.Cost.factory(jac_mode="forward", name=f"ik_{side}")
    def ik_cost(vals, v, tp, tq, th):
        return residual(model, side, vals[v], tp, tq, th)

    problem = jaxls.LeastSquaresProblem(
        costs=[ik_cost(var, target_pos, target_quat, theta)], variables=[var]
    ).analyze()
    sol, summary = problem.solve(
        initial_vals=jaxls.VarValues.make([var.with_value(q0)]),
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(lambda_initial=cfg.lambda_initial),
        termination=jaxls.TerminationConfig(max_iterations=cfg.max_iterations),
        return_summary=True,
    )
    return sol[var], summary.iterations


def _diag_one(
    model: ArmModel, side: str, q7: jax.Array, target_pos: jax.Array,
    target_quat: jax.Array, theta: jax.Array,
) -> dict[str, jax.Array]:
    th = dict(zip(THETA_NAMES, theta))
    pts = model.arm_points(side, q7)
    pos, quat = model.tip(side, q7)
    clear = model.clearance(side, q7)
    r = residual(model, side, q7, target_pos, target_quat, theta)
    sl = block_slices(model)
    return {
        "pos_err": jnp.linalg.norm(pos - target_pos),
        "ori_err": jnp.linalg.norm((jaxlie.SO3(target_quat).inverse() @ jaxlie.SO3(quat)).log()),
        "min_clear": clear.min(),
        "min_pair": jnp.argmin(clear),
        "limit_margin": jnp.minimum(
            q7 - jnp.asarray(model.lower[side]), jnp.asarray(model.upper[side]) - q7
        ).min(),
        "elbow": q7[ELBOW],
        "swivel": swivel_angle(pts["shoulder"], pts["elbow"], pts["wrist"], SIDE_SIGN[side]),
        "swivel_ref": swivel_target(th, target_pos, pts["shoulder"]),
        "manip": manipulability(model, side, q7),
        "res_norm": jnp.stack([jnp.linalg.norm(r[sl[b]]) for b in BLOCKS]),
        "tip_pos": pos,
        "tip_quat": quat,
    }


@dataclass
class Diag:
    """Per-sample diagnostics of a batch of solutions (numpy, leading axis N).

    Attributes:
        pos_err: Tip position error, m. ori_err: Tip orientation error, deg.
        min_clear: Smallest floor-subtracted clearance, m. min_pair: Its pair index.
        limit_margin: Smallest distance to a joint limit, rad (negative = outside).
        elbow: Elbow angle, rad. swivel: Elbow swivel psi, rad (see `swivel_angle`).
        swivel_ref: The swivel target of that sample, rad. manip: Yoshikawa measure.
        res_norm: Residual norm per block, shape (N, len(BLOCKS)).
        tip_pos, tip_quat: Achieved tip pose, pelvis frame, wxyz.
    """

    pos_err: np.ndarray
    ori_err: np.ndarray
    min_clear: np.ndarray
    min_pair: np.ndarray
    limit_margin: np.ndarray
    elbow: np.ndarray
    swivel: np.ndarray
    swivel_ref: np.ndarray
    manip: np.ndarray
    res_norm: np.ndarray
    tip_pos: np.ndarray
    tip_quat: np.ndarray

    def select(self, idx: np.ndarray) -> "Diag":
        """Rows `idx`."""
        return Diag(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})


def accept(d: Diag, gates: GateCfg, cost: CostCfg) -> tuple[np.ndarray, np.ndarray]:
    """Gate a batch: `ok` (N,) bool and `reason` (N,) uint8 bitmask over `REASONS`."""
    fails = [
        d.pos_err > gates.pos_tol,
        d.ori_err > gates.ori_tol,
        d.min_clear < 0.0,
        d.limit_margin < 0.0,
        d.elbow > cost.elbow_max + gates.elbow_slack,
    ]
    reason = np.zeros(d.pos_err.shape[0], dtype=np.uint8)
    for i, f in enumerate(fails):
        reason |= (f.astype(np.uint8) << i)
    return reason == 0, reason


def reason_names(mask: int) -> list[str]:
    """`REASONS` entries set in a reason bitmask."""
    return [r for i, r in enumerate(REASONS) if mask & (1 << i)]


class HandSolver:
    """Batched IK for one hand at a fixed compiled batch size.

    Any number of rows may be passed; they are solved in chunks of `batch`,
    the last one padded with copies of its first row.
    """

    def __init__(self, model: ArmModel, side: str, cfg: CostCfg, batch: int):
        self.model, self.side, self.cfg, self.batch = model, side, cfg, batch
        self._solve = jax.jit(
            jax.vmap(partial(_solve_one, model, side, cfg), in_axes=(0, 0, 0, None))
        )
        self._diag = jax.jit(
            jax.vmap(partial(_diag_one, model, side), in_axes=(0, 0, 0, None))
        )

    def _chunks(self, fn, arrays: list[np.ndarray], theta: np.ndarray) -> list:
        n = arrays[0].shape[0]
        if n == 0:  # one dummy row; callers slice the results back to zero rows
            arrays = [np.zeros((1,) + a.shape[1:], np.float32) for a in arrays]
        outs = []
        for start in range(0, max(n, 1), self.batch):
            parts = []
            for a in arrays:
                a = np.ascontiguousarray(a[start:start + self.batch], dtype=np.float32)
                pad = self.batch - a.shape[0]
                if pad:
                    a = np.concatenate([a, np.repeat(a[:1], pad, axis=0)])
                parts.append(jnp.asarray(a))
            outs.append(fn(*parts, jnp.asarray(theta, dtype=jnp.float32)))
        return outs

    def solve(
        self, target_pos: np.ndarray, target_quat: np.ndarray, q0: np.ndarray,
        theta: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve N targets (positions (N, 3) m, orientations (N, 4) wxyz) from
        warm starts `q0` (N, 7). Returns the solutions (N, 7) and the LM
        iteration counts (N,)."""
        n = target_pos.shape[0]
        theta = self.cfg.theta() if theta is None else theta
        outs = self._chunks(self._solve, [target_pos, target_quat, q0], theta)
        q = np.concatenate([np.asarray(o[0]) for o in outs])[:n]
        it = np.concatenate([np.asarray(o[1]) for o in outs])[:n]
        return q, it

    def diagnostics(
        self, q7: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray,
        theta: np.ndarray | None = None,
    ) -> Diag:
        """Diagnostics of solutions `q7` (N, 7) against their targets."""
        n = q7.shape[0]
        theta = self.cfg.theta() if theta is None else theta
        outs = self._chunks(self._diag, [q7, target_pos, target_quat], theta)
        merged = {k: np.concatenate([np.asarray(o[k]) for o in outs])[:n] for k in outs[0]}
        merged["ori_err"] = np.degrees(merged["ori_err"])
        return Diag(**merged)


def main() -> None:
    import time

    from .generate import Range, quat_from_pyr
    from .model import CFG_PATH, RobotCfg, SIDES, load_cfg

    y = load_cfg(CFG_PATH)
    model = ArmModel.load(RobotCfg.from_dict(y["robot"]))
    cfg, gates = CostCfg.from_dict(y.get("cost")), GateCfg.from_dict(y.get("gates"))
    batch = 2048
    rng = np.random.default_rng(0)
    for side in SIDES:
        solver = HandSolver(model, side, cfg, batch)
        pos, quat = model.nominal_target(side)
        q, _ = solver.solve(pos[None], quat[None], model.rest7(side)[None])
        diag = solver.diagnostics(q, pos[None], quat[None])
        print(f"{side}: nominal solve pos_err {diag.pos_err[0] * 1e3:.2f} mm ori_err {diag.ori_err[0]:.2f} deg")
        pos, pyr = Range.from_cfg(side, y, model.shoulder(side)).sample(rng, batch)
        pos, quat = pos.astype(np.float32), quat_from_pyr(pyr).astype(np.float32)
        q0 = np.repeat(model.rest7(side)[None], batch, 0)
        t0 = time.perf_counter()
        q, it = solver.solve(pos, quat, q0)
        dt = time.perf_counter() - t0
        ok, reason = accept(solver.diagnostics(q, pos, quat), gates, cfg)
        by = {r: int(((reason >> i) & 1).sum()) for i, r in enumerate(REASONS)}
        print(f"{side}: {batch} random in-region targets from rest: {int(ok.sum())} ok, "
              f"{batch / dt:.0f} solves/s (first side includes the compile), fails by reason {by}, "
              f"iters p50 {np.median(it):.0f}")


if __name__ == "__main__":
    main()
