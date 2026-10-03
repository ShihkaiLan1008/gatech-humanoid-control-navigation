# Collecting Belief-Transition Data for a Tactile World Model

*Scientific report on the Room 2 data-collection pipeline (run `room2-belief-v3`,
20,001 transitions, 2026-08-25). Draft material for a future paper section.*

---

## 1. What one datum is

The world model to be trained predicts one step of localization-belief
evolution under a touch:

$$p(o_{t+1},\, b_{t+1} \mid b_t,\, c_t,\, M)$$

where $M$ is the 2-D floor plan (a set of convex bodies in the map frame),
$c_t$ is a touch command given in the robot's body frame, $o_{t+1}$ is what
the hand feels, and $b_t$ is the robot's belief over its own SE(2) pose,
represented as a weighted particle set. One datum is therefore a 4-tuple
$(b_t, c_t, o_{t+1}, b_{t+1})$, and the dataset is a set of such one-step
transitions.

Transitions come in **local segments** of 10–20 consecutive touches: one
random base placement, one particle filter initialized once, then each touch
folded into that same filter. Within a segment $b_{t+1}$ of step $t$ is
literally the same particle array as $b_t$ of step $t+1$; this makes the
recursive chain checkable from the files alone. Segments are short on
purpose — the model's job is local belief dynamics, not full-room
trajectories.

Two rules act as a firewall between simulation ground truth and the data:

1. **Labels come from execution.** Whether a touch is a contact is decided by
   the deployed controller's own contact detector running on the simulated
   force sensor — never by querying the simulator's geometry.
2. **Ground truth steers only the sampler.** The simulator's true pose and
   geometry may be used to *choose what to attempt* (balancing expected
   hits/misses, placing the initial belief near the truth) but never enter
   the filter and never become a label. True poses are stored alongside each
   transition as explicitly privileged fields, for evaluation only.

No navigation planner, information-gain criterion, or learned policy selects
any base pose or touch target. Every choice is a random draw under quota
control. This keeps the action distribution broad and free of the
correlations a planner would imprint, which matters because the world model
must later be queried by a planner *off* the collection policy.

## 2. Simulation setup

**Scene.** Room 2 is a mapped scan: a west wall (2 layered panels behind it),
a north wall, and a cylindrical bulge of radius 0.16 m embedded in the north
wall. The map $M$ used by the filter is the same five-body convex plan the
navigation stack uses. The reference route is a frozen polyline hugging the
walls: a 2.8 m vertical leg, a 1.0 m corner elbow, and a 2.0 m horizontal
leg, at the standing standoff of the deployed controller.

**Robot and controller.** A Unitree G1 humanoid with a learned
velocity–force whole-body controller ("vforce"). The controller was trained
to press with a commanded tip position in a bounded body-frame workspace and
a force intent; data collection commands it exactly as deployment does. No
part of the controller was modified or retrained for collection.

**Force sensing.** Contact force is read through the registered sensor model
rather than raw simulator force: saturation at 20 N, additive Gaussian noise
of $\sigma = 1$ N, a 100 Hz sample-and-hold, and 10 ms transport latency. The
controller's contact detector runs on this corrupted channel, so the labels
inherit the sensor's false-positive/false-negative behavior (Section 6).

**Batched worlds.** Collection runs 128 physically independent copies of the
room in one batched MuJoCo (mujoco-warp) simulation on a single GPU. Each
world holds one robot and one room; worlds cannot interact. Randomizing the
base pose is implemented by *moving the room around a fixed robot spawn*
(re-placing the wall bodies by a rigid transform per world) rather than
teleporting the robot, which keeps the controller's state machine and the
robot's standing state untouched between segments. Each world executes its
own independent segment; a scheduler keeps all 128 busy and retires segments
as they finish. Sustained throughput was ≈24,000 valid transitions per hour.
After each world (re)placement the robot runs a short warm-up before the
first touch, because the first press otherwise reads 20–33 N of arm-swing
transient as contact.

## 3. Sampling base poses

Let $r(s)$, $s \in [0, L]$, be the arclength-parameterized route polyline and
$n(s)$ its left-hand unit normal. A base pose is drawn as

$$x_{\text{base}} = r(s) + \delta\, n(s), \qquad
  s \sim \mathcal{U}(0, L), \quad \delta \sim \mathcal{U}(-0.20, 0.20)\ \text{m},$$

with yaw set to face the wall: the pose's heading is the outward normal of
the *nearest wall face*, found by the gradient of the minimum signed distance
to the plan bodies (not the nearest body's average orientation, which is
wrong at corners). Yaw is not additionally randomized; the controller was
trained to press roughly square to a surface, and the lateral offset
$\delta$ already varies the reach geometry by ±0.20 m.

A drawn pose is **rejected** if any of: the robot's standing footprint
penetrates a wall, the pose cannot be installed as a stable standing
configuration, or the parked hand would already be inside a body. To avoid
paying a ~24:1 rejection rate at draw time, feasibility is surveyed once per
run on a grid of (region × lateral-bin) cells (60 probe draws per cell), and
sampling then draws only from cells with nonzero standing probability. The
survey also documents dead zones (e.g. the innermost lateral bin of the
vertical leg is entirely un-standable at the layered panels).

**Region quotas.** The route is partitioned into *vertical*, *corner*, and
*horizontal* regions with planned shares 40/20/40 %, and a fourth **vertex**
class (10 % of the run, carved proportionally from the other three) that
places the base deliberately at the exact north-west corner point, standing
on either face at 0.28–0.45 m along it. The vertex class exists because
uniform sampling along a route polyline gives the single most informative
landmark of the room — the corner point itself — vanishing probability mass:
in a 12k-transition pilot without it, the closest recorded contact to the
vertex was 0.14 m; with it, 0.019 m, with 58 contacts within 0.15 m.

Region, lateral bin, and outcome quotas are enforced by *share-deficit*
control: each proposal is scored by how far each class's realized share lags
its planned share, with in-flight (already dispatched, not yet finished)
segments counted as committed. The in-flight term matters at 128 parallel
worlds: without it, all 128 opening segments see the same deficits and pile
onto the same class, which produced a 91 % → oscillating contact ratio in an
early version.

## 4. Sampling touch commands

A touch command $c_t$ is a target tip position in the body frame, drawn from
the controller's *trained* workspace — the box it was actually commanded in
during training:

$$x \in [0.30, 0.48]\ \text{m}, \qquad
  |y| \in [0.08, 0.35]\ \text{m}, \qquad
  z \in [0.10, 0.25]\ \text{m},$$

with the sign of $y$ selecting the left or right hand. Commands outside this
box are out-of-distribution for the controller and would confound "the world
model is wrong" with "the controller is wrong", so they are excluded by
construction.

To guarantee coverage of the workspace rather than merely of its average,
targets are stratified into 3 height × 5 angular × 3 radial-reach bins
(45 bins after the hand symmetry is folded in). Bin edges are
**equal-probability quantiles** of the reachable set, not equal-width cuts:
equal widths in the angular coordinate leave a middle bin only 0.8° wide that
is geometrically infeasible to fill. Sampling is uniform within a bin, bins
are drawn by deficit against a floor of ≥150 transitions per bin, and the
delivered run reached all 45 bins with a minimum of 431.

**Balancing expected hit/miss.** The dataset targets 45–55 % contact. The
only ground-truth read the sampler is allowed is a 2-D ray cast of the tip's
straight-line path (parked-carry position → target, projected to the plane)
against the plan bodies: it predicts whether a candidate *would* sweep
through a mapped body. Candidate poses and targets are scored so that
whichever outcome class is behind its 50 % share is preferred, run-wide and
per region. The predictor is used with zero inflation padding — an earlier
version padded the ray by a hand radius and mispredicted 16 of 20 misses as
hits at typical standoffs. The prediction is *never* stored as a label; the
label is what the controller's detector reports after execution. Realized
contact share was 48.7 %.

## 5. Executing a touch

Each touch runs the deployed press primitive unmodified: raise the commanded
hand from the carry pose, approach the target, then press along the approach
direction at 0.02 m/s with a 5 N force intent. The primitive terminates on
the first *debounced* sensed contact — the sensed force channel must exceed
2 N on three consecutive 100 Hz samples. With $\sigma = 1$ N sensor noise a
single-sample threshold would false-latch ≈3.4 times per press of pure air;
three consecutive samples reduce the expected false latch to $2\times10^{-3}$
per press. On contact the hand backs off 0.06 m; on no contact the primitive
ends when the target is reached. A safety guard aborts any press whose
sensed force exceeds 12 N sustained.

Every attempt receives exactly one of three labels:

* **`contact`** — debounced sensed contact. Recorded observation: the
  contact point and (when the wrist could measure it) the surface normal in
  the body frame, time-to-contact, and the full sensed-force profile
  (saturating at 20 N; the true peak force is stored separately as a
  privileged field).
* **`no_contact`** — the press completed without a debounced contact.
  Recorded observation: the swept tip path as a sequence of body-frame
  points, i.e. a *negative* measurement ("this arc of space is free").
* **`invalid_attempt`** — IK failure, self-collision, safety abort,
  controller failure, or an unreachable target. Nothing was measured, so the
  filter is *not* updated and `belief_after` is null. These rows are kept
  (the taxonomy is part of the data) but excluded from the transition count.
  The spec bound is <5 %; the delivered run has 0 %.

Because the detector runs on the noisy sensed channel, a small set of
contacts latch on noise with exactly zero true force — 57 of 9,738 contacts
(0.6 %) in the delivered run. These **phantom contacts are deliberately
kept and flagged**, not filtered: they are the detector's real
false-positive rate, and a world model trained for deployment should see the
observation distribution the deployed system produces.

## 6. The belief: initialization from an empirical prior

Each segment needs an initial belief $b_0$. Choosing its covariance by hand
would make the dataset's uncertainty scale an authored constant, so $b_0$ is
bootstrapped from covariances the Room 2 particle filter *has actually
carried*.

**Harvest.** Existing Room 2 localization runs (navigation traces and saved
particle sets) are scanned; at each logged instant the weighted moments

$$\mu = \sum_i w_i x_i, \qquad
  \Sigma = \sum_i w_i\, (x_i - \mu)(x_i - \mu)^{\!\top}, \qquad x = (x, y, \theta)$$

are computed, with the heading averaged on the circle and differenced with
wrapping before the outer product (otherwise a belief straddling $\pm\pi$
reports a spurious variance of order $\pi^2$). The full 3×3 covariance is
kept, including cross terms. Samples are bucketed by the route region the
robot was in, giving a pool of 341 covariances (83 corner / 156 horizontal /
102 vertical).

**Multimodality is carried, not collapsed.** Logged particle sets are
cluster-tested (nearest-center partition; a mode must hold ≥10 % of the mass
and modes must be ≥0.30 m apart). If a logged belief is multimodal, the pool
stores the *mixture* — each mode's own mean offset, mass, and covariance —
never the single second moment of the union. This rule exists because of a
measured failure: summarizing one 3-mode belief (per-mode $\sigma \approx
0.2$ m, modes 5–6 m apart) by its overall covariance produced a Gaussian
with $\sigma \approx 2.2$ m whose mean carried only 4 % of the probability
mass within $0.5\sigma$ — a shape the filter never carried and a
pathological training input.

**Drawing $b_0$.** For each segment a pool entry from the pose's region is
drawn and scaled by a log-normal jitter, $\Sigma' = k\,\Sigma$ with $\ln k
\sim \mathcal{N}(0, 0.15^2)$, so the prior is dense around the observed
scales instead of a finite set of exact repeats. The belief mean is then
placed by drawing the *error* from the belief's own shape:

$$\mu_0 = x^{*} + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \Sigma'),$$

where $x^{*}$ is the true pose. This makes the truth a typical sample under
$b_0$ — a belief is never 0.3 m wrong with a 5 cm sigma, and never centered
on the truth exactly. 600 particles are sampled from
$\mathcal{N}(\mu_0, \Sigma')$; the true pose itself is never given to the
filter, and the median initial error $\lVert\mu_0 - x^{*}\rVert$ in the
delivered run is 0.11 m with the expected heavy tail (p90 = 0.27 m).

For a mixture entry, the dominant mode is placed near the truth as above and
the remaining modes keep their observed offsets, so the geometry of the
ambiguity is the one the filter really carried. Every component must pass a
plausibility gate: ≥0.20 m clear of every mapped body **and** inside the
mapped extent grown by 0.50 m. The second condition was added after a
component was accepted at $y = 8.66$ m in a room whose far wall is at
3.85 m — metres clear of every body, and impossible. If no placement
satisfies the gate for all components (the offsets may simply not fit the
room), the dominant mode stands alone and the fallback is recorded in the
row's provenance. Delivered composition: 1,354 unimodal / 19 mixture / 11
dropped-mixture-fallback segments. The fallback tail is why a few $b_0$
position-covariance traces reach 1.0 m²: those are single modes whose *own
logged* covariance was that wide (along-wall corridor ambiguity), not
sampling noise.

## 7. The belief: one recursive update per touch

The filter is the deployed tactile-localization particle filter, used
unmodified. Each particle carries a pose in SE(2), a discrete route-edge
label, a discrete last-contact association, and a continuous 2-D local map
*registration* offset (the drawn map may be locally displaced from the true
walls; a flat-wall touch cannot distinguish "I am 5 cm closer" from "the
wall is drawn 5 cm too far", and the particle population carries that
ambiguity instead of resolving it prematurely).

Per touch, exactly one predict–update cycle runs; the belief is never reset,
resampled from scratch, or regenerated from ground truth mid-segment.

**Predict.** The odometry increment since the previous touch (millimetres —
the base is held between touches) is applied with the filter's standard
per-distance process noise ($\sigma_\parallel = 0.05$, $\sigma_\perp = 0.03$
m per m, $\sigma_\psi = 0.02$ rad per m + $0.06$ rad per rad; registration
random-walks at 0.004 m per m). Skipping predict for a held base would make
the recorded process noise depend on how the collector was written, so it is
not skipped.

**Measurement construction.** The executed touch is converted to the
filter's observation vocabulary with the registered measurement noise
applied once, at construction: the contact point is corrupted with
$\sigma = 5$ mm isotropic noise and the measured normal rotated by
$\sigma = 5°$; a normal the wrist could not measure (railed against its
±0.12 rad tracking clamp) is passed as *absent* rather than guessed, because
an assumed normal actively confirms whatever yaw error the belief already
carries. A no-contact press becomes a swept-free observation: the tip's
body-frame path points.

**Update.** For a contact at body-frame point $p$, each particle $i$
predicts the world point $\hat p_i$ and is scored against the map. For a
featureless surface contact the log-likelihood is
$-d_i^2 / 2\sigma_c^2$ with $d_i$ the distance of $\hat p_i$ to the nearest
surface (under that particle's registration offset) and $\sigma_c = 3$ cm; a
signature-matched landmark contact is scored against all candidate entities
and marginalized under a uniform association prior,

$$\ell_i = \operatorname{logsumexp}_k\!\left(-\tfrac{r_{ik}^2}{2\sigma_k^2}\right) - \log C .$$

A measured normal adds $-\Delta\phi_i^2 / 2\sigma_n^2$ for the angle between
the predicted and map surface normals. A contact is rejected as an outlier
(counted, not applied) when less than 5 % of the belief *mass* passes a
$\chi^2$ = 25 validation gate — gating on the single best particle would let
a 1-in-$N$ tail "explain" any phantom. A swept-free observation penalizes
each particle by its worst map intrusion along the sweep,
$-\min(\text{pen}_i^2 / 2\sigma_f^2,\ 4)$ nats, capped so one ray degrades
rather than kills; this is the only channel that can rule a hypothesis out
without touching anything. Structural terms (a robot is not inside a wall;
a particle is not far outside its own route corridor) apply every update.

Resampling is systematic, triggered only when
$\mathrm{ESS} = (\sum_i w_i)^2 / \sum_i w_i^2$ falls below $N/2$, followed by
small roughening (1 cm, 0.005 rad) — the standard recipe for preserving
minority modes. In the delivered run the end-of-segment ESS/N median is 0.93
(p10 = 0.59), i.e. the filter stays healthy while still being visibly moved
by evidence: 65 % of accepted contacts and 68 % of free sweeps tighten the
position covariance, and the recorded traces span collapse ratios down to
$2\times10^{-3}$ within a segment.

## 8. What is recorded

Per attempt: the full particle set of $b_t$ and $b_{t+1}$ (600 × $(x, y,
\theta, w)$ each, not just their moments), their Gaussian summaries (mean,
3×3 covariance, eigendecomposition, ESS), the command in body/map/world
frames with its bin labels, the observation as the filter consumed it (noisy
point/normal or swept-free points, plus sensed and true force profiles), the
one-step filter report (accepted/outlier, resampled, associations), and the
privileged true state before and after. In addition the full 36-D
generalized coordinates $q$ of every world are logged at 25 Hz, and each
attempt stores its index range into that log — so any recorded touch can be
replayed kinematically frame-by-frame, which is both a debugging tool and a
zero-ambiguity proof of what the robot actually did (the five published
sample videos are such replays, not re-simulations).

**Splits** are 80/10/10 train/val/test assigned at *segment* level, never at
transition level: the beliefs inside a segment are the same filter at
consecutive instants, and splitting them across sets would leak nearly
identical particle sets between train and test.

## 9. Delivered dataset and validation

Run `room2-belief-v3`: 20,001 valid transitions in 1,384 segments (median 14
touches/segment), 48.7 % contact, 0 % invalid, splits 16,024 / 2,050 / 1,927.
Six checks are computed from the written files (not the collector's
bookkeeping) and all pass:

1. contact share within 45–55 %;
2. invalid rate below 5 %;
3. splits are segment-disjoint;
4. the PF chain is recursive ($b_{t+1}$ row = next $b_t$ row; `n_updates`
   increments by exactly one per valid transition and not at all on invalid
   ones);
5. no ground-truth leak into the filter — tested *statistically*: the median
   belief-to-truth distance must exceed 1 cm and sub-millimetre agreement
   must stay below 1 % of rows. (A "no row within 1 mm" test is a
   coincidence detector: over 20k transitions a correctly converging filter
   produces a smooth tail of near-hits, while a leak puts a spike at exactly
   zero. Delivered: median 8.8 cm, 3 rows of 20,001 under 1 mm.)
6. every valid row is a complete one-step transition (both beliefs present).

Coverage was additionally verified geometrically: all 45 action bins ≥431
transitions; lateral bins within 17–25 % each; the corner vertex reached to
0.019 m.

## 10. Deliberate properties and limitations

* **Collection policy ≠ deployment policy.** Uniform-with-quotas sampling is
  intentionally broader than any planner's visitation distribution; a model
  trained on it must still be *evaluated* under planner-chosen actions.
* **Phantom contacts are in-distribution.** 0.6 % of contact labels have
  zero true force by design (Section 5).
* **The filter's scoring noise is deliberately wider than the injected
  noise** (3 cm vs 5 mm point, 11.5° vs 5° normal): the deployed filter's
  registered conservatism is part of the belief dynamics being modeled, so
  it is left as deployed.
* **Single-room, single-map.** All 20k transitions are Room 2; the map
  enters the data as the conditioning variable $M$, but generalization
  across rooms is untested by this dataset.
* **Simulation stochasticity.** Batched GPU physics is not bitwise
  reproducible across runs (float-atomics ordering); identical seeds diverge
  at the ULP level within one step. Statistics reported here are
  population-level, and no claim rests on replaying a specific trajectory
  through physics (replays are kinematic).
