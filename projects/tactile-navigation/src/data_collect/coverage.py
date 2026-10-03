"""Running coverage, and the proposal scoring that fills it.

The collector is told what the finished dataset has to look like -- 40/20/40
across route regions, 45-55 % contact, every lateral-offset bin above 15 %,
every feasible action bin above ~150 -- and those are joint constraints on a
process that only ever draws random numbers.  This module is where they are
tracked and where a draw is chosen.

HOW A DRAW IS CHOSEN, AND WHY IT IS STILL RANDOM.  Nothing here optimises a
touch for information.  Several candidates are drawn UNIFORMLY from the
sampling distributions, each is scored purely by how much of the dataset's
remaining DEFICIT it would fill, and the best-scoring one is executed.  Inside
every bin the draw stays uniform; what the score decides is which bin gets the
next sample, which is the only lever that makes joint quotas reachable at all.

Contact balance needs the deficit lever more than anything else, because room2
does not hand it over for free: the base can only stand 0.38 m or more from a
face without its carried hands ploughing it, the tip reaches 0.48 m, so a
contact exists only in a 10 cm band of standoff out of the 40 cm the +-0.20 m
lateral jitter spans.  Drawing uniformly and hoping produces roughly 7 %
contacts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

REGION_SHARE = {"horizontal": 0.40, "corner": 0.20, "vertical": 0.40}
"""Default route-region shares.  A run may override them -- collecting the
corner VERTEX itself adds a fourth region and rescales these."""


def region_share_with_vertex(vertex_share: float) -> dict:
    """The three route regions, rescaled to leave room for `vertex`.

    The vertex is not a stretch of route: the centreline TURNS at the corner,
    so no pose on it stands alongside the vertex on either face.  It is
    sampled from the two faces directly and takes its own share of the
    dataset."""
    v = float(vertex_share)
    if v <= 0.0:
        return dict(REGION_SHARE)
    out = {k: s * (1.0 - v) for k, s in REGION_SHARE.items()}
    out["vertex"] = v
    return out
OUTCOME_SHARE = {"contact": 0.50, "no_contact": 0.50}
LATERAL_MIN_SHARE = 0.15
ACTION_BIN_MIN = 150
N_LATERAL_BINS = 5

W_REGION = 1.0
W_OUTCOME = 4.0
W_REGION_OUTCOME = 3.0
W_LATERAL = 1.0
W_ACTION = 1.5
"""Relative pull of each axis, and a statement about which constraint wins.

Three of the spec's requirements are in genuine tension in this room, and the
arithmetic is worth writing down.  A contact needs the base 0.38-0.48 m off a
face; the centreline stands at 0.50/0.56 and the jitter is +-0.20, so contacts
live in the two lateral bins nearest the wall.  If the other three bins must
hold 15 % each -- 45 % of the data at a standoff nothing can reach, where the
measured contact rate is 2.8 % -- then the two near bins carry 55 % of the data
and have to convert at 88 % for the run to reach 50 % overall.  That is at the
very top of what a capable pose delivers (measured: 89 % at cap 0.2-0.5, 98 %
above), and getting it means the target sampler picking reaching targets nearly
every time in those poses, which starves the near-REACH action bins.

So the three cannot all be maximal, and the weights choose: OUTCOME balance is
the requirement with a stated acceptable range (45-55 %, section 7), the
lateral spread is a FLOOR (no bin below 15 %, section 6), and the action bins
are a floor too (~150 each).  Outcome therefore outweighs both -- at 2.0
against LATERAL's 1.0 the floor won every tie and the contact rate settled
near 42 %."""


def lateral_bin(offset_m: float, span_m: float,
                n: int = N_LATERAL_BINS) -> int:
    """Which of the `n` equal bins of `[-span, +span]` an offset falls in."""
    lo, hi = -float(span_m), float(span_m)
    if hi <= lo:
        return 0
    k = int((float(offset_m) - lo) / (hi - lo) * n)
    return max(0, min(n - 1, k))


def lateral_bin_edges(span_m: float, n: int = N_LATERAL_BINS) -> list:
    step = 2.0 * float(span_m) / n
    return [round(-float(span_m) + i * step, 4) for i in range(n + 1)]


@dataclass
class Coverage:
    """Counts of accepted VALID transitions, and the deficits they leave."""

    target_n: int
    action_bins: tuple
    n_lateral: int = N_LATERAL_BINS
    region_share: dict = field(default_factory=lambda: dict(REGION_SHARE))
    region: Counter = field(default_factory=Counter)
    outcome: Counter = field(default_factory=Counter)
    region_outcome: Counter = field(default_factory=Counter)
    lateral: Counter = field(default_factory=Counter)
    action: Counter = field(default_factory=Counter)
    invalid: Counter = field(default_factory=Counter)
    pending_region: dict = field(default_factory=dict)
    pending_outcome: dict = field(default_factory=dict)
    """Transitions that live segments are EXPECTED to contribute, by class.

    A quota is filled by segments of 10-20 touches, and the batched collector
    commits 128 of them before a single row exists.  Judging the next choice
    on the accepted count alone means 128 identical decisions -- measured, the
    run opened at 91 % contact and then spent thousands of samples paying it
    back.  Counting what is already promised is what turns a quota into a
    schedule."""
    n_valid: int = 0
    n_attempts: int = 0

    # ── recording ────────────────────────────────────────────────────────────

    def add(self, *, region: str, outcome: str, lat_bin: int,
            action_bin: str) -> None:
        self.n_attempts += 1
        if outcome == "invalid_attempt":
            self.invalid["total"] += 1
            return
        self.n_valid += 1
        self.region[region] += 1
        self.outcome[outcome] += 1
        self.region_outcome[(region, outcome)] += 1
        self.lateral[int(lat_bin)] += 1
        self.action[action_bin] += 1

    def add_invalid(self, reason: str) -> None:
        self.n_attempts += 1
        self.invalid["total"] += 1
        self.invalid[reason] += 1

    @property
    def done(self) -> bool:
        return self.n_valid >= self.target_n

    @property
    def invalid_rate(self) -> float:
        return (self.invalid["total"] / self.n_attempts) if self.n_attempts else 0.0

    # ── quotas ───────────────────────────────────────────────────────────────

    def planned(self, outcome: str) -> float:
        """Accepted transitions of this class plus the ones already promised."""
        return self.outcome[outcome] + float(
            self.pending_outcome.get(outcome, 0.0))

    def region_deficit(self, region: str) -> float:
        """How far this region's SHARE is below its target share."""
        tot = max(1.0, sum(self.region[r] + self.pending_region.get(r, 0.0)
                           for r in self.region_share))
        got = (self.region[region]
               + self.pending_region.get(region, 0.0)) / tot
        return max(0.0, self.region_share.get(region, 0.0) - got)

    def lateral_deficit(self, k: int) -> float:
        """How far this lateral bin's SHARE is below the 15 % floor."""
        tot = max(1.0, float(sum(self.lateral.values())))
        return max(0.0, LATERAL_MIN_SHARE - self.lateral[int(k)] / tot)

    def action_deficit(self, key: str) -> float:
        """How far this action bin's SHARE is below an even split."""
        n = max(1, len(self.action_bins))
        tot = max(1.0, float(sum(self.action.values())))
        return max(0.0, (1.0 / n) - self.action[key] / tot)

    def share_deficit(self, outcome: str) -> float:
        """How far this class's SHARE is below its target share.

        The outcome axis has to be judged on share, not on count, and the
        difference is the whole balance.  A raw count deficit is
        `(quota - have) / quota`, which two thirds of the way through a run is
        large for BOTH classes -- 0.69 and 0.58 at one measured point -- so a
        pose that can reach a wall outscored one that cannot by 0.09 out of
        1.2, and the contact rate slid from 59 % to 41 % while every lever was
        pointing the right way.  On share the same state reads
        `0.50 - 0.414 = +0.086` for contact and `-0.086 -> 0` for no-contact:
        one class gets the whole weight, and it is the one behind.

        Zero once a class is at or above its target share, so the pressure
        reverses cleanly instead of merely weakening."""
        total = max(1.0, self.planned("contact") + self.planned("no_contact"))
        got = self.planned(outcome) / total
        return max(0.0, OUTCOME_SHARE.get(outcome, 0.0) - got)

    def regions(self) -> tuple:
        return tuple(self.region_share)

    def quota_region(self, region: str) -> float:
        return self.region_share.get(region, 0.0) * self.target_n

    def quota_outcome(self, outcome: str) -> float:
        return OUTCOME_SHARE.get(outcome, 0.0) * self.target_n

    def quota_region_outcome(self, region: str, outcome: str) -> float:
        return (self.region_share.get(region, 0.0)
                * OUTCOME_SHARE.get(outcome, 0.0) * self.target_n)

    def quota_lateral(self) -> float:
        """A FLOOR, not an even share -- and the difference is the dataset.

        Section 6 asks that no lateral-offset bin hold less than 15 % of the
        final data; it does not ask for 20 % each.  That slack is the only
        room the geometry leaves: a contact needs the base 0.38-0.48 m off a
        face and the centreline stands at 0.50/0.56, so contacts exist only
        in the two bins nearest the wall.  Scoring the lateral axis against
        an even 20 % share would spend a fifth of the dataset on bins that
        cannot contact and cap the contact rate near 40 %.  Scoring it
        against the 15 % floor lets the remaining quarter follow the contact
        deficit into the near bins, which is what lands 45-55 %."""
        return LATERAL_MIN_SHARE * self.target_n

    def action_min(self) -> int:
        """The per-bin floor this run can actually be judged against.

        The spec's 150 assumes the full 12 000; a 200-transition smoke run
        cannot put 150 in each of 45 bins and reporting 45 shortfalls for it
        says nothing.  Below that scale the floor becomes the even share, so
        the smoke run is judged on whether it SPREADS."""
        even = self.target_n // max(1, len(self.action_bins))
        return int(min(ACTION_BIN_MIN, max(1, even)))

    def quota_action(self) -> float:
        """Even share of the target, floored at this run's per-bin minimum."""
        even = self.target_n / max(1, len(self.action_bins))
        return max(float(self.action_min()), even)

    # ── deficits, as fractions in [0, 1] ─────────────────────────────────────

    @staticmethod
    def deficit(count: float, quota: float) -> float:
        if quota <= 0:
            return 0.0
        return max(0.0, (quota - count) / quota)

    def region_share_deficit(self, region: str, outcome: str) -> float:
        """`share_deficit`, asked of one route region's own two classes."""
        tot = max(1.0, sum(self.region_outcome[(region, o)]
                           for o in OUTCOME_SHARE))
        got = self.region_outcome[(region, outcome)] / tot
        return max(0.0, OUTCOME_SHARE.get(outcome, 0.0) - got)

    def pose_score(self, region: str, lat_bin: int,
                   contact_capability: float) -> float:
        """How much a base pose in this cell would help.

        `contact_capability` is the FRACTION of the workspace that a
        ground-truth swept ray says would touch something from this pose --
        the balancing read section 4 allows.  It is multiplied into the
        contact deficit (a pose that cannot reach anything cannot repay a
        contact deficit) and into the complement for the no-contact deficit
        (a pose that touches on every draw cannot repay a miss deficit)."""
        cap = max(0.0, min(1.0, float(contact_capability)))
        s = W_REGION * self.deficit(self.region[region],
                                     self.quota_region(region))
        s += W_LATERAL * self.deficit(self.lateral[int(lat_bin)],
                                       self.quota_lateral())
        s += W_OUTCOME * (
            cap * self.deficit(self.outcome["contact"],
                                self.quota_outcome("contact"))
            + (1.0 - cap) * self.deficit(self.outcome["no_contact"],
                                          self.quota_outcome("no_contact")))
        s += W_REGION_OUTCOME * (
            cap * self.deficit(self.region_outcome[(region, "contact")],
                                self.quota_region_outcome(region, "contact"))
            + (1.0 - cap) * self.deficit(
                self.region_outcome[(region, "no_contact")],
                self.quota_region_outcome(region, "no_contact")))
        return s

    def target_score(self, region: str, action_bin: str,
                     predicted: str) -> float:
        """How much one touch target would help, given its predicted class."""
        s = W_ACTION * self.deficit(self.action[action_bin],
                                     self.quota_action())
        s += W_OUTCOME * self.deficit(self.outcome[predicted],
                                       self.quota_outcome(predicted))
        s += W_REGION_OUTCOME * self.region_share_deficit(region, predicted)
        return s

    # ── reporting ────────────────────────────────────────────────────────────

    def shortfalls(self) -> dict:
        """Every requirement that the dataset does not yet meet."""
        out: dict = {}
        n = max(1, self.n_valid)
        for r, share in self.region_share.items():
            got = self.region[r] / n
            if abs(got - share) > 0.05:
                out[f"region:{r}"] = round(got, 4)
        for o, share in OUTCOME_SHARE.items():
            got = self.outcome[o] / n
            if not (0.45 <= got <= 0.55):
                out[f"outcome:{o}"] = round(got, 4)
        for r in self.region_share:
            tot = sum(self.region_outcome[(r, o)] for o in OUTCOME_SHARE)
            for o in OUTCOME_SHARE:
                if tot and self.region_outcome[(r, o)] / tot < 0.15:
                    out[f"region_outcome:{r}/{o}"] = round(
                        self.region_outcome[(r, o)] / tot, 4)
        for k in range(self.n_lateral):
            got = self.lateral[k] / n
            if got < LATERAL_MIN_SHARE:
                out[f"lateral_bin:{k}"] = round(got, 4)
        floor = self.action_min()
        for b in self.action_bins:
            if self.action[b] < floor:
                out[f"action_bin:{b}"] = int(self.action[b])
        if self.invalid_rate > 0.05:
            out["invalid_rate"] = round(self.invalid_rate, 4)
        return out

    def as_dict(self) -> dict:
        n = max(1, self.n_valid)
        return {
            "target_n": self.target_n,
            "valid": self.n_valid, "attempts": self.n_attempts,
            "contact": self.outcome["contact"],
            "no_contact": self.outcome["no_contact"],
            "contact_frac": round(self.outcome["contact"] / n, 4),
            "invalid": dict(self.invalid),
            "invalid_rate": round(self.invalid_rate, 4),
            "region": dict(self.region),
            "region_outcome": {f"{r}/{o}": c
                               for (r, o), c in sorted(
                                   self.region_outcome.items())},
            "lateral_bin": {str(k): self.lateral[k]
                            for k in range(self.n_lateral)},
            "action_bin": {b: self.action[b] for b in self.action_bins},
            "action_bin_min": self.action_min(),
            "action_bins_under_min": sorted(
                b for b in self.action_bins
                if self.action[b] < self.action_min()),
            "shortfalls": self.shortfalls(),
        }
