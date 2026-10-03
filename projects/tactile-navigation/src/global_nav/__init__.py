"""Map-aligned global tactile localization: belief (Batch 1) + route planner (Batch 2).

Batch 1: a multi-hypothesis SE(2) belief over a KNOWN floorplan -- noisy odometry,
realized contacts, swept-free volumes.  Batch 2 makes it planner-ready (measured
0.45 m reach, registration gating, NORMAL/SUSPECT_LOST/RELOCALIZE) and adds the
belief-aware global route planner with its stressed benchmarks and baselines.
Still no MuJoCo, no G1, no learning, no world model.
"""

from .affordance import REACH_M, AffordanceGraph, Station
from .belief import BeliefConfig, ParticleBelief
from .managed import BeliefState, ManagedBelief, ManagedConfig
from .floorplan import (
    Floorplan,
    MapFeature,
    ProbeFan,
    RouteEdge,
    TactileStation,
    benchmark_floorplan,
    default_stations,
)
from .geometry import ConvexPolygon, raycast, rect, wrap_angle
from .models import MotionConfig, ObservationConfig
from .observations import ContactObs, SensorConfig, SweptFreeObs, simulate_station

__all__ = [
    "BeliefConfig", "ParticleBelief",
    "BeliefState", "ManagedBelief", "ManagedConfig",
    "AffordanceGraph", "Station", "REACH_M",
    "Floorplan", "MapFeature", "RouteEdge", "ProbeFan", "TactileStation",
    "benchmark_floorplan", "default_stations",
    "ConvexPolygon", "rect", "raycast", "wrap_angle",
    "MotionConfig", "ObservationConfig",
    "ContactObs", "SweptFreeObs", "SensorConfig", "simulate_station",
]
