"""Named RNG chain.

Every random draw in this package comes from ``named_rng(seed, *names)``.  The
stream is a pure function of (integer seed, name path): no wall-clock, no
process id, no global ``np.random`` state.  Two different name paths under the
same seed give independent streams, so adding a new randomised quantity does
not perturb the ones that already exist.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["named_rng", "stream_tag"]


def stream_tag(*names: str) -> int:
    """Stable 64-bit tag for a name path (order matters, contents matter)."""
    if not names:
        raise ValueError("named_rng requires at least one stream name")
    for n in names:
        if not isinstance(n, str) or not n:
            raise ValueError(f"stream names must be non-empty strings, got {n!r}")
    key = "/".join(names).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def named_rng(seed: int, *names: str) -> np.random.Generator:
    """A ``Generator`` for the stream ``names`` under ``seed``."""
    if int(seed) != seed or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    seq = np.random.SeedSequence(entropy=int(seed), spawn_key=(stream_tag(*names),))
    return np.random.default_rng(seq)
