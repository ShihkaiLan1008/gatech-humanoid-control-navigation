"""Scenario definitions and the runner that produces the validation numbers.

The robot walks the nominal route OPEN-LOOP: the commands are read off the map, the
true pose integrates them with drift, and nothing ever corrects the commands.  That
is deliberate.  A closed-loop robot would hide the belief's errors behind a
controller, and the question here is what the BELIEF knows, not how well something
downstream could act on it.

Ground truth and the map differ by a rigid registration offset ``reg_true``: the
world is the drawing, translated.  The robot starts at the drawing's origin, so it
begins a few centimetres off the true corridor centerline -- which is what "the
floorplan is a few centimetres wrong" means when you have to walk down it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .belief import BeliefConfig, ParticleBelief
from .floorplan import (
    AHEAD_FAN,
    LEFT_FAN,
    RIGHT_FAN,
    Floorplan,
    TactileStation,
    benchmark_floorplan,
)
from .geometry import wrap_angle
from .models import MotionConfig, ObservationConfig
from .observations import SensorConfig, simulate_station

__all__ = ["ScenarioConfig", "run_scenario", "SCENARIOS", "build_scenarios", "run_all"]


# Truth drifts slightly LESS than the filter assumes.  A filter that assumes less
# noise than reality is overconfident and will diverge; the reverse merely costs
# sharpness, so this asymmetry is the safe default rather than a tuned one.
TRUTH_DRIFT = MotionConfig(
    sigma_along_per_m=0.050, sigma_cross_per_m=0.030,
    sigma_yaw_per_m=0.010, sigma_yaw_per_rad=0.040, sigma_reg_walk_per_m=0.0,
)
FILTER_DRIFT = MotionConfig(
    sigma_along_per_m=0.060, sigma_cross_per_m=0.040,
    sigma_yaw_per_m=0.015, sigma_yaw_per_rad=0.060, sigma_reg_walk_per_m=0.004,
)


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    question: str
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    truth_drift: MotionConfig = field(default_factory=lambda: TRUTH_DRIFT)
    reg_true: tuple[float, float] = (0.06, -0.04)
    station_spacing_m: float = 1.0
    station_start_m: float = 0.5
    fans: tuple = (LEFT_FAN, RIGHT_FAN, AHEAD_FAN)
    step_m: float = 0.25
    route_end_m: float | None = None
    start_s_m: float = 0.0
    prior_sigma_xy_m: tuple[float, float] = (0.20, 0.20)
    prior_sigma_theta_rad: float = 0.030
    prior_sigma_reg_m: float = 0.100
    prior_offset_xy: tuple[float, float] = (0.0, 0.0)
    prior_modes: tuple | None = None
    """``((xy, theta, edge, weight), ...)`` for a genuinely multi-modal prior.  When
    set, it replaces the unimodal Gaussian prior entirely."""

    particle_sample: int = 0
    """Keep this many particles per logged step, for the map figures.  Zero keeps
    none: the traces are otherwise summary statistics, and a bimodal cloud is the
    one thing summary statistics cannot show."""

    seed: int = 0


# ── scripted motion ──────────────────────────────────────────────────────────────


def command_sequence(plan: Floorplan, step_m: float, start_s_m: float = 0.0,
                     end_s_m: float | None = None):
    """Open-loop walk/turn commands along the nominal route.

    Yields ``(delta_body, delta_yaw, s_after, edge)``.  Turns are separate
    zero-translation commands, matching how a legged robot actually takes a corner
    and keeping the yaw increment out of the translation increment.
    """
    end = plan.route_length_m if end_s_m is None else float(end_s_m)
    s = float(start_s_m)
    heading = plan.route_pose(s)[1]
    while s < end - 1e-9:
        _, edge_heading, edge = plan.route_pose(min(s + 1e-6, end))
        turn = float(wrap_angle(edge_heading - heading))
        if abs(turn) > 1e-9:
            heading = wrap_angle(heading + turn)
            yield np.zeros(2), turn, s, edge
        # do not step past the end of the current edge: the turn belongs to the corner
        edge_end = sum(plan.edge(i).length_m for i in plan.route[:plan.route.index(edge) + 1])
        step = min(step_m, edge_end - s, end - s)
        if step <= 1e-9:
            s = min(edge_end, end) + 1e-9
            continue
        s += step
        yield np.array([step, 0.0]), 0.0, s, edge


def stations_for(plan: Floorplan, spacing_m: float, start_m: float,
                 end_m: float | None = None, fans=(LEFT_FAN, RIGHT_FAN, AHEAD_FAN)
                 ) -> tuple[TactileStation, ...]:
    end = plan.route_length_m if end_m is None else float(end_m)
    out, s = [], float(start_m)
    while s < end:
        out.append(TactileStation(s, tuple(fans), f"s{len(out)}"))
        s += float(spacing_m)
    return tuple(out)


# ── the runner ───────────────────────────────────────────────────────────────────


def _make_prior(plan: Floorplan, scn: ScenarioConfig, rng, xy0, theta0, edge0) -> ParticleBelief:
    if scn.prior_modes is not None:
        return ParticleBelief.initialize_mixture(
            plan, scn.belief, rng, components=scn.prior_modes,
            sigma_xy_m=np.asarray(scn.prior_sigma_xy_m, float),
            sigma_theta_rad=scn.prior_sigma_theta_rad, sigma_reg_m=scn.prior_sigma_reg_m)
    return ParticleBelief.initialize(
        plan, scn.belief, rng,
        xy=np.asarray(xy0, float) + np.asarray(scn.prior_offset_xy, float),
        theta=theta0, edge=edge0,
        sigma_xy_m=np.asarray(scn.prior_sigma_xy_m, float),
        sigma_theta_rad=scn.prior_sigma_theta_rad, sigma_reg_m=scn.prior_sigma_reg_m)


def run_scenario(scn: ScenarioConfig) -> dict:
    plan = benchmark_floorplan()
    reg_true = np.asarray(scn.reg_true, dtype=float).reshape(2)
    world = plan.translated(reg_true)          # ground truth: the drawing, moved
    rng = np.random.default_rng(scn.seed)

    xy0, theta0, edge0 = plan.route_pose(scn.start_s_m)
    truth_xy, truth_theta = np.array(xy0, dtype=float), float(theta0)

    belief = _make_prior(plan, scn, rng, xy0, theta0, edge0)

    stations = list(stations_for(plan, scn.station_spacing_m, scn.station_start_m, scn.route_end_m, scn.fans))
    trace: list[dict] = []
    truth_edge = edge0
    s_now = float(scn.start_s_m)

    def log(station_label: str | None, report: dict | None) -> None:
        row = belief.snapshot({"xy": truth_xy, "theta": truth_theta,
                               "reg": reg_true, "edge": truth_edge})
        row.update({"s_m": s_now, "truth_xy": truth_xy.tolist(), "truth_theta": truth_theta,
                    "truth_edge": int(truth_edge), "station": station_label,
                    "report": report})
        if scn.particle_sample:
            k = min(int(scn.particle_sample), belief.n)
            pick = np.linspace(0, belief.n - 1, k).astype(int)
            row["particles"] = {"xy": belief.xy[pick].tolist(),
                                "w": (belief.weights[pick] * belief.n / k).tolist()}
        trace.append(row)

    log(None, None)
    for delta_body, delta_yaw, s_after, edge in command_sequence(
            plan, scn.step_m, scn.start_s_m, scn.route_end_m):
        # truth executes the command with drift; the belief only ever sees the command
        d = scn.truth_drift
        dist, turn = float(np.linalg.norm(delta_body)), abs(float(delta_yaw))
        noisy = delta_body + rng.standard_normal(2) * np.array(
            [d.sigma_along_per_m * dist, d.sigma_cross_per_m * dist])
        c, s_ = math.cos(truth_theta), math.sin(truth_theta)
        truth_xy = truth_xy + np.array([c * noisy[0] - s_ * noisy[1], s_ * noisy[0] + c * noisy[1]])
        truth_theta = float(wrap_angle(
            truth_theta + delta_yaw
            + math.hypot(d.sigma_yaw_per_m * dist, d.sigma_yaw_per_rad * turn)
            * rng.standard_normal()))
        truth_edge, s_prev, s_now = edge, s_now, s_after

        belief.predict(delta_body, delta_yaw)

        fired = [st for st in stations if s_prev < st.s_m <= s_after]
        if not fired:
            log(None, None)
            continue
        for station in fired:
            contacts, frees = simulate_station(world, truth_xy, truth_theta, station,
                                               scn.sensor, rng)
            report = belief.update(contacts, frees)
            report["contacts"] = list(belief.last_contacts)
            log(station.label, report)

    return {"scenario": scn.name, "question": scn.question,
            "trace": trace, "summary": summarize(trace), "reg_true": reg_true.tolist()}


def summarize(trace: list[dict]) -> dict:
    stations = [r for r in trace if r["station"] is not None]
    err = np.array([r["position_error_m"] for r in trace])
    yaw = np.array([abs(r["heading_error_rad"]) for r in trace])
    return {
        "n_steps": len(trace),
        "n_stations": len(stations),
        "final_position_error_m": float(err[-1]),
        "mean_position_error_m": float(err.mean()),
        "max_position_error_m": float(err.max()),
        "final_heading_error_deg": float(math.degrees(yaw[-1])),
        "final_registration_error_m": float(trace[-1].get("registration_error_m", float("nan"))),
        "final_pos_std_m": float(np.linalg.norm(trace[-1]["pos_std_m"])),
        "final_wrong_turn_probability": float(trace[-1]["wrong_turn_probability"]),
        "max_wrong_turn_probability": float(max(r["wrong_turn_probability"] for r in trace)),
        "final_p_true_edge": float(trace[-1]["p_true_edge"]),
        "min_ess": float(min(r["ess"] for r in trace)),
        "max_modes": int(max(r["n_modes"] for r in trace)),
        "final_modes": int(trace[-1]["n_modes"]),
        "n_outliers": int(trace[-1]["n_outliers"]),
        "n_resamples": int(trace[-1]["n_resamples"]),
    }


# ── the six validation scenarios ─────────────────────────────────────────────────


def build_scenarios(n_particles: int = 2000, seed: int = 0) -> dict[str, ScenarioConfig]:
    """One scenario per line of the validation plan.  The first four are the same
    run with different measurement channels enabled, so their numbers are directly
    comparable -- that is the point of holding everything else fixed."""
    base = BeliefConfig(n_particles=n_particles, motion=FILTER_DRIFT,
                        observation=ObservationConfig())

    def ablation(name: str, question: str, **flags) -> ScenarioConfig:
        return ScenarioConfig(name=name, question=question, seed=seed,
                              belief=replace(base, **flags))

    # The measurement channels are ablated against a HELD-CONSTANT map prior (the
    # structural term), because that prior is not a measurement -- it is the standing
    # fact that a robot is not inside a wall and is in the corridor it claims.
    # ``structural_only`` is the control that makes the other three attributable: it
    # says how far the map prior alone gets you, so that whatever contact_only beats
    # it by is genuinely bought by touch.
    scenarios = {
        "odometry_only": ablation(
            "odometry_only",
            "With no evidence at all, does the belief drift -- and does it KNOW it is drifting?",
            use_contacts=False, use_swept_free=False, use_structural=False),
        "structural_only": ablation(
            "structural_only",
            "How much does the map prior alone buy, with no tactile measurement whatsoever?",
            use_contacts=False, use_swept_free=False, use_structural=True),
        "contact_only": ablation(
            "contact_only",
            "Do realized contacts re-anchor the pose beyond what the map prior already gives?",
            use_contacts=True, use_swept_free=False, use_structural=True),
        "swept_free_only": ablation(
            "swept_free_only",
            "Does negative information alone -- touching NOTHING -- localize?",
            use_contacts=False, use_swept_free=True, use_structural=True),
        "combined": ablation(
            "combined",
            "Do the channels compose, or does one of them dominate?",
            use_contacts=True, use_swept_free=True, use_structural=True),
    }

    # THE AMBIGUITY.  The robot has felt the north wall end.  The map says that
    # happens at x=2.2 and again at x=5.2, and the two are geometrically identical,
    # so the prior is 50/50 over both -- which is not a wide Gaussian centred on
    # x=3.7, where the robot certainly is not.
    #
    # It is really at 5.2, the true junction.  Both hypotheses then take the SAME
    # commands and turn left, so the wrong one drives up the dead-end decoy alcove.
    # The alcove and corridor B are 1.6 m wide and identical to the side probes, so
    # touch cannot separate them; what separates them is the AHEAD probe finding no
    # end wall where the alcove hypothesis insists on one.  Negative information
    # resolves it, roughly 1.2 m into the turn.
    scenarios["ambiguous_landmarks"] = ScenarioConfig(
        name="ambiguous_landmarks",
        question="Given a 50/50 prior over two identical wall termini, does the belief carry "
                 "both hypotheses instead of guessing -- and does the evidence resolve it?",
        belief=replace(base, use_contacts=True, use_swept_free=True, use_structural=True),
        seed=seed, start_s_m=5.2, route_end_m=9.5,
        station_spacing_m=0.5, station_start_m=5.4,
        prior_modes=(((2.20, 0.0), 0.0, 0, 0.5), ((5.20, 0.0), 0.0, 1, 0.5)),
        prior_sigma_xy_m=(0.15, 0.10),
    )

    # The same ambiguity with the resolving measurement WITHHELD.  Above, the
    # forward probe finds corridor A's end wall at 0.83 m and settles it in one
    # station.  Drop the forward fan and the side probes cannot separate the two
    # hypotheses at all -- the decoy alcove and corridor B are both 1.6 m wide, so
    # left and right read identically -- and the ambiguity survives the turn.  That
    # is when it becomes a WRONG TURN rather than a position error, and it is the
    # only scenario in the suite where wrong-turn probability gets to be large.
    scenarios["ambiguous_side_only"] = replace(
        scenarios["ambiguous_landmarks"],
        name="ambiguous_side_only",
        question="With the forward probe withheld, how long does the ambiguity survive, and "
                 "does it show up as WRONG-TURN probability rather than as position error?",
        fans=(LEFT_FAN, RIGHT_FAN),
    )

    # A tactile sensor reporting a signature the map cannot produce anywhere.
    scenarios["bad_association"] = ScenarioConfig(
        name="bad_association",
        question="Does a contact that matches no map feature get rejected rather than "
                 "force-fit onto the nearest plausible one?",
        belief=replace(base, use_contacts=True, use_swept_free=True, use_structural=True),
        seed=seed, route_end_m=6.0,
    )
    return scenarios


SCENARIOS = ("odometry_only", "structural_only", "contact_only", "swept_free_only",
             "combined", "ambiguous_landmarks", "ambiguous_side_only", "bad_association")
ABLATIONS = ("odometry_only", "structural_only", "contact_only", "swept_free_only", "combined")


def run_all(n_particles: int = 2000, seed: int = 0, names=None) -> dict[str, dict]:
    built = build_scenarios(n_particles, seed)
    out = {}
    for name in (names or SCENARIOS):
        if name == "bad_association":
            out[name] = run_bad_association(built[name])
        else:
            out[name] = run_scenario(built[name])
    return out


def run_ablations(seeds=(0, 1, 2, 3, 4), n_particles: int = 2000) -> dict:
    """The measurement-channel ablation, over several seeds.

    Reports the standard error of the mean as well as the spread, because the
    interesting comparisons here are between channels whose gaps are the same size
    as the seed noise.  Without the s.e.m. it is possible to read an ordering off
    this table that five more seeds would reverse -- which is exactly what a first
    pass at five seeds did.
    """
    rows: dict[str, list[dict]] = {name: [] for name in ABLATIONS}
    for seed in seeds:
        built = build_scenarios(n_particles, int(seed))
        for name in ABLATIONS:
            rows[name].append(run_scenario(built[name])["summary"])

    def agg(name: str, key: str) -> dict:
        v = np.array([r[key] for r in rows[name]], dtype=float)
        sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
        return {"mean": float(v.mean()), "std": sd, "sem": sd / math.sqrt(max(v.size, 1)),
                "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}

    keys = ("final_position_error_m", "mean_position_error_m", "final_pos_std_m",
            "final_heading_error_deg", "final_registration_error_m",
            "max_wrong_turn_probability", "min_ess", "n_resamples")
    return {"seeds": list(map(int, seeds)), "n_particles": n_particles,
            "table": {name: {k: agg(name, k) for k in keys} for name in ABLATIONS},
            "raw": rows}


def registration_sweep(magnitudes=(0.0, 0.05, 0.10, 0.20, 0.30, 0.40),
                       seeds=(0, 1, 2), n_particles: int = 1000) -> dict:
    """Is the local registration offset actually observable, and down to what floor?

    A contact on a flat wall constrains ONE linear combination of the pose and the
    registration offset -- "I am 5 cm nearer" and "the wall is drawn 5 cm nearer"
    are the same measurement -- so only odometry separates them, and odometry gets
    worse with distance.  This sweep asks the question that settles it empirically:
    feed in a known offset, and see how much of it comes back.
    """
    base = build_scenarios(n_particles, 0)["combined"]
    out = []
    for mag in magnitudes:
        before, after = [], []
        for seed in seeds:
            scn = replace(base, seed=int(seed),
                          reg_true=(float(mag) * 0.8, -float(mag) * 0.6),
                          prior_sigma_reg_m=0.30, prior_sigma_xy_m=(0.05, 0.05))
            r = run_scenario(scn)
            before.append(float(np.linalg.norm(r["reg_true"])))
            after.append(float(r["trace"][-1]["registration_error_m"]))
        out.append({"true_offset_m": float(np.mean(before)),
                    "residual_error_m": float(np.mean(after)),
                    "residual_std_m": float(np.std(after, ddof=1)) if len(seeds) > 1 else 0.0})
    return {"seeds": list(map(int, seeds)), "n_particles": n_particles, "points": out}


def run_bad_association(scn: ScenarioConfig) -> dict:
    """Same walk, but every third station also reports one impossible contact.

    "Impossible" means a signature the map has no entity for, placed where no
    surface is.  The belief must reject it: an accepted phantom would reweight
    toward whichever hypothesis is least wrong, which is how a filter becomes
    confidently lost rather than merely uncertain.
    """
    from .observations import ContactObs

    plan = benchmark_floorplan()
    reg_true = np.asarray(scn.reg_true, dtype=float).reshape(2)
    world = plan.translated(reg_true)
    rng = np.random.default_rng(scn.seed)
    xy0, theta0, edge0 = plan.route_pose(scn.start_s_m)
    truth_xy, truth_theta = np.array(xy0, dtype=float), float(theta0)

    belief = ParticleBelief.initialize(
        plan, scn.belief, rng, xy=np.asarray(xy0, float), theta=theta0, edge=edge0,
        sigma_xy_m=np.asarray(scn.prior_sigma_xy_m, float),
        sigma_theta_rad=scn.prior_sigma_theta_rad, sigma_reg_m=scn.prior_sigma_reg_m)

    stations = list(stations_for(plan, scn.station_spacing_m, scn.station_start_m, scn.route_end_m, scn.fans))
    trace, truth_edge, s_now, n_injected = [], edge0, float(scn.start_s_m), 0

    for k, (delta_body, delta_yaw, s_after, edge) in enumerate(
            command_sequence(plan, scn.step_m, scn.start_s_m, scn.route_end_m)):
        d = scn.truth_drift
        dist, turn = float(np.linalg.norm(delta_body)), abs(float(delta_yaw))
        noisy = delta_body + rng.standard_normal(2) * np.array(
            [d.sigma_along_per_m * dist, d.sigma_cross_per_m * dist])
        c, s_ = math.cos(truth_theta), math.sin(truth_theta)
        truth_xy = truth_xy + np.array([c * noisy[0] - s_ * noisy[1], s_ * noisy[0] + c * noisy[1]])
        truth_theta = float(wrap_angle(
            truth_theta + delta_yaw
            + math.hypot(d.sigma_yaw_per_m * dist, d.sigma_yaw_per_rad * turn)
            * rng.standard_normal()))
        truth_edge, s_prev, s_now = edge, s_now, s_after
        belief.predict(delta_body, delta_yaw)

        for station in [st for st in stations if s_prev < st.s_m <= s_after]:
            contacts, frees = simulate_station(world, truth_xy, truth_theta, station,
                                               scn.sensor, rng)
            if len(trace) % 3 == 0:
                contacts = list(contacts) + [ContactObs(
                    p_body=(0.35, 0.02), signature="door_handle", normal_body=(-1.0, 0.0),
                    fan="injected", truth={"phantom": True})]
                n_injected += 1
            report = belief.update(contacts, frees)
            report["contacts"] = list(belief.last_contacts)
            row = belief.snapshot({"xy": truth_xy, "theta": truth_theta,
                                   "reg": reg_true, "edge": truth_edge})
            row.update({"s_m": s_now, "truth_xy": truth_xy.tolist(), "truth_theta": truth_theta,
                        "truth_edge": int(truth_edge), "station": station.label, "report": report})
            trace.append(row)

    summary = summarize(trace)
    summary["n_injected_phantoms"] = n_injected
    summary["phantom_rejection_rate"] = (
        float(summary["n_outliers"]) / n_injected if n_injected else float("nan"))
    return {"scenario": scn.name, "question": scn.question, "trace": trace,
            "summary": summary, "reg_true": reg_true.tolist()}
