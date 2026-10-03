"""Minimal leg-vs-leg mesh-distance gate.

A single flat-ground step can only self-collide between the two lower legs /
feet (arms and waist are pinned near the reference pose, and the anti-cross
floor keeps the thighs apart), so the checker builds collision pairs ONLY for
lower-leg geometry (links whose parent joint contains 'knee' or 'ankle'),
left side x right side, and sweeps pin.computeDistances over the trajectory.
Run LAST in the gate chain (most expensive); a ::2 frame subsample is fine
against a 5 mm threshold.
"""

import numpy as np
import pinocchio as pin

_KEYWORDS = ("knee", "ankle")


class LegCollisionChecker:
    def __init__(self, robot):
        self.model = robot.model
        self.gmodel = robot.collision_model
        left, right = [], []
        for i, geom in enumerate(self.gmodel.geometryObjects):
            jname = self.model.names[geom.parentJoint]
            if not any(k in jname for k in _KEYWORDS):
                continue
            (left if jname.startswith("left_") else right).append(i)
        self.gmodel.removeAllCollisionPairs()
        for i in left:
            for j in right:
                self.gmodel.addCollisionPair(pin.CollisionPair(i, j))
        self.n_pairs = len(self.gmodel.collisionPairs)
        self.data = self.model.createData()
        self.gdata = pin.GeometryData(self.gmodel)

    def min_gap(self, qs, stride=2):
        """Smallest left-right lower-leg mesh distance [m] over qs[::stride]."""
        if self.n_pairs == 0:
            return np.inf
        best = np.inf
        for q in np.asarray(qs)[::stride]:
            pin.computeDistances(self.model, self.data, self.gmodel, self.gdata, q)
            d = min(r.min_distance for r in self.gdata.distanceResults)
            best = min(best, float(d))
        return best
