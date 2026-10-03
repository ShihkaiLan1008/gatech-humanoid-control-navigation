"""One economy for every touch: shared λ prices and the unified IG family.

The wall objective and the landmark utility grew separately and priced the
same things twice:

    wall  J(s)   = IG      + λc·new_coverage − λm·walk − λr·σ_pre
    ate  utility = score   − λm·|Δs| − λb·backtrack

with λm defined in two places, λr missing from the landmark side (a long
visit hop drifts exactly like a long wall walk), and the information terms
computed by two implementations of the SAME formula: the wall's
``0.5·ln(1 + σ_n²/σ_t²)`` is the rank-1 special case of the Gaussian
information gain

    IG = 0.5 · ln det(I + H Σ Hᵀ R⁻¹)

that ``tactile_explore.infogain._pose_score`` already evaluates (in its
``½[logdet Σ − logdet(Σ⁻¹ + HᵀR⁻¹H)⁻¹]`` form -- equal by the matrix
determinant lemma, and asserted equal by the tests).  Feature types differ
only in the observation Jacobian H: a wall contributes its face normal
(rank 1), a cylinder the radial direction of the touch, a corner its two
face normals.  The ATE's hypothesis-discrimination term (``ig_hyp``, a
Monte-Carlo mutual information over the hypothesis label) is NOT part of
this family -- it is the multi-hypothesis correction and stays a separate
additive term.

This module is the single definition of both: ``TouchEconomy`` for the
prices, ``unified_ig`` for the information."""
from __future__ import annotations

import dataclasses
import math

import numpy as np


@dataclasses.dataclass
class TouchEconomy:
    """Prices in nats, one instance shared by every touch decision."""

    lambda_motion: float = 0.5
    """nats per metre of walking to reach a candidate."""
    lambda_backtrack: float = 2.5
    """EXTRA nats per metre of motion against route progress (charged on
    top of lambda_motion; walls never generate backward candidates, so it
    binds only inside visits)."""
    lambda_coverage: float = 0.4
    """Bonus for a candidate on an arc no touch has covered yet."""
    lambda_risk: float = 2.5
    """nats per metre of PREDICTED belief σ at the candidate.  On walls
    this is what makes spacing emerge; on visits it prices predicted σ
    GROWTH over a stance hop (it was missing there before this module).

    16.0 was the wall price, tuned when a drifted arrival meant a missed
    press and a spent search ladder.  Feeler-first removed that failure
    mode (the press hunts the face to a 0.35 m cap and stops on touch),
    and 16 also kills every visit's rank-2 second press, whose measured
    margin is ~0.27 nats.  2.5 keeps the concavity that spaces wall
    touches while leaving required bracketing presses affordable; the
    wall-side effect is validated by the lr3 speed experiment."""
    wall_floor_nats: float = 0.15
    """Below this J the wall press is declined and the base just walks."""
    landmark_floor_nats: float = 0.06
    """The ATE's eig_min: a candidate scoring less is not worth pressing."""


def drift_sigma(distance_m: float, sigma_cross_per_m: float = 0.104) -> float:
    """Predicted belief-σ growth over a walk, from the filter's own motion
    model -- the ONE constant both objectives grow uncertainty with."""
    return float(sigma_cross_per_m) * abs(float(distance_m))


def drift_risk_nats(economy: "TouchEconomy", sigma_now_m: float,
                    distance_m: float,
                    sigma_cross_per_m: float = 0.104) -> float:
    """THE uncertainty cost of an action, for every candidate class.

    INCREMENTAL: what going there COSTS, `sigma_after - sigma_now`, not the
    absolute σ the robot happens to be carrying.  Absolute σ is a property of
    the belief, not of the action being priced -- charging it makes every
    candidate at a station pay the same toll, which cancels out of the
    ranking and then silently moves the accept/decline floor instead.  The
    wall objective charged the absolute form and the landmark utility the
    incremental one; at σ≈0.10 and λr=2.5 that is a 0.25-nat difference
    against a 0.15-nat floor, i.e. the two classes were not comparable.

    The sqrt-growth form, not a linear λr·k·|Δs|: linear is exactly collinear
    with lambda_motion (measured killing the bracketing second press at ANY
    λr >= ~2.3), and the curvature is what makes wall spacing emerge."""
    s0 = float(sigma_now_m)
    grown = math.hypot(s0, drift_sigma(distance_m, sigma_cross_per_m))
    return float(economy.lambda_risk) * (grown - s0)


# canonical implementation lives one layer down, where the ATE scorer can
# reach it without demo_map imports; re-exported here as the economy's
# information side.  The two together are the outcome-marginal score every
# touch decision runs on: unified_ig prices the hit, miss_truncation_ig the
# miss, and the caller weights them by p_hit.
from tactile_explore.infogain import (miss_truncation_ig,  # noqa: E402,F401
                                      unified_ig, unswept_hit_mass)
