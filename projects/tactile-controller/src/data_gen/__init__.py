"""Trajectory-optimization reference generators for the DECOUPLED-POLICY line
(2026-08-02 plan of record, branch `decoupled-policy`).

Fresh, minimal rewrites -- deliberately NOT built on the mpc_offline skill stack
(user request: that stack grew hard to read). Each task is one self-contained
subpackage; display reuses mpc_offline.viser_play (the mjviser fixes are not
re-derived).

Tasks:
  foot_placement -- single-step foot-placement references for the lower-body
                    tracking policy (REFINE-DP pi_lower recipe; see
                    docs/refinedp_lowerbody_notes.md).
"""
