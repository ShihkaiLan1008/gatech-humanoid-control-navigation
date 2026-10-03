import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from global_nav.belief import BeliefConfig, ParticleBelief          # noqa: E402
from global_nav.floorplan import benchmark_floorplan               # noqa: E402
from global_nav.models import MotionConfig, ObservationConfig      # noqa: E402


@pytest.fixture(scope="session")
def plan():
    return benchmark_floorplan()


def make_belief(plan, *, n=800, seed=0, xy=(1.0, 0.0), theta=0.0, edge=0,
                sigma_xy_m=0.20, sigma_reg_m=0.10, **flags) -> ParticleBelief:
    cfg = BeliefConfig(n_particles=n, motion=MotionConfig(),
                       observation=ObservationConfig(), **flags)
    return ParticleBelief.initialize(
        plan, cfg, np.random.default_rng(seed), xy=np.asarray(xy, float), theta=theta,
        edge=edge, sigma_xy_m=sigma_xy_m, sigma_theta_rad=0.03, sigma_reg_m=sigma_reg_m)
