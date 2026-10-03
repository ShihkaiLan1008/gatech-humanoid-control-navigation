"""The quota bookkeeping, and the one place the spec's numbers live."""

import pytest

from data_collect.coverage import (ACTION_BIN_MIN, LATERAL_MIN_SHARE,
                                   OUTCOME_SHARE, REGION_SHARE, Coverage,
                                   lateral_bin, lateral_bin_edges)

BINS = tuple(f"h{h}a{a}r{r}" for h in range(3) for a in range(5)
             for r in range(3))


def cov(n=12000):
    return Coverage(target_n=n, action_bins=BINS)


def test_the_spec_numbers_are_the_spec_numbers():
    assert REGION_SHARE == {"horizontal": 0.40, "corner": 0.20,
                            "vertical": 0.40}
    assert OUTCOME_SHARE == {"contact": 0.50, "no_contact": 0.50}
    assert LATERAL_MIN_SHARE == 0.15
    assert ACTION_BIN_MIN == 150
    c = cov()
    assert c.quota_region("horizontal") == 4800    # section 6, verbatim
    assert c.quota_region("corner") == 2400
    assert c.quota_region("vertical") == 4800
    assert sum(c.quota_region(r) for r in REGION_SHARE) == 12000


def test_the_lateral_quota_is_a_floor_and_not_an_even_share():
    """15 % x 5 = 75 %, and the free quarter is what pays for 50 % contact.

    Contacts exist only where the base stands 0.38-0.48 m off a face, which
    on a centreline 0.50-0.56 m out is the two bins nearest the wall.  Held
    to an even 20 % each, the contact rate caps near 40 %."""
    c = cov()
    assert c.quota_lateral() == 0.15 * 12000
    assert c.quota_lateral() * 5 < 12000


def test_lateral_bins_tile_the_jitter_range():
    e = lateral_bin_edges(0.20)
    assert e == [-0.2, -0.12, -0.04, 0.04, 0.12, 0.2]
    assert [lateral_bin(v, 0.20) for v in
            (-0.20, -0.13, -0.05, 0.0, 0.05, 0.13, 0.20)] == \
        [0, 0, 1, 2, 3, 4, 4]


def test_the_action_bin_floor_scales_down_for_a_smoke_run():
    """150 per bin assumes the full 12 000; reporting 45 shortfalls on a
    200-transition validation run says nothing about the pipeline."""
    assert cov(12000).action_min() == 150
    assert cov(200).action_min() == 4
    assert cov(12000).quota_action() == pytest.approx(12000 / 45)


def test_a_deficit_falls_to_zero_when_a_quota_is_met():
    c = cov(1000)
    assert c.deficit(0, 100) == 1.0
    assert c.deficit(50, 100) == 0.5
    assert c.deficit(100, 100) == 0.0
    assert c.deficit(400, 100) == 0.0


def test_a_pose_that_cannot_reach_anything_cannot_repay_a_contactdeficit():
    c = cov(1000)
    for _ in range(500):                       # no_contact quota already met
        c.add(region="vertical", outcome="no_contact", lat_bin=2,
              action_bin=BINS[0])
    reachy = c.pose_score("vertical", 2, contact_capability=0.9)
    blind = c.pose_score("vertical", 2, contact_capability=0.0)
    assert reachy > blind


def test_a_target_is_scored_by_the_class_it_would_produce():
    c = cov(1000)
    for _ in range(500):
        c.add(region="corner", outcome="contact", lat_bin=0,
              action_bin=BINS[0])
    assert c.target_score("corner", BINS[1], "no_contact") > \
        c.target_score("corner", BINS[1], "contact")


def test_invalid_attempts_never_count_toward_the_target():
    c = cov(10)
    c.add_invalid("safety_abort")
    c.add_invalid("safety_abort")
    c.add(region="corner", outcome="contact", lat_bin=0, action_bin=BINS[0])
    assert c.n_valid == 1 and c.n_attempts == 3
    assert c.invalid["safety_abort"] == 2
    assert c.invalid_rate == pytest.approx(2 / 3)


def test_shortfalls_name_every_requirement_that_is_not_met():
    c = cov(100)
    for i in range(100):                       # all contact, one region, one bin
        c.add(region="corner", outcome="contact", lat_bin=0,
              action_bin=BINS[0])
    s = c.shortfalls()
    assert "outcome:contact" in s and "outcome:no_contact" in s
    assert "region:horizontal" in s and "region:vertical" in s
    assert "region_outcome:corner/no_contact" in s
    assert "lateral_bin:4" in s
    assert any(k.startswith("action_bin:") for k in s)


def test_a_balanced_dataset_reports_no_shortfalls():
    c = cov(len(BINS) * 20)
    per_region = {"horizontal": 0.40, "corner": 0.20, "vertical": 0.40}
    i = 0
    for region, share in per_region.items():
        for k in range(int(share * c.target_n)):
            c.add(region=region,
                  outcome="contact" if k % 2 else "no_contact",
                  lat_bin=k % 5, action_bin=BINS[i % len(BINS)])
            i += 1
    assert c.shortfalls() == {}
    assert c.done
