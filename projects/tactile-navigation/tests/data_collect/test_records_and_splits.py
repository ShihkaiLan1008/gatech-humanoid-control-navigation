"""The file format, and the split that keeps one base pose out of two sets."""

import json
from collections import Counter

import numpy as np
import pytest

from data_collect.records import (SPLIT_FRACTIONS, TransitionWriter,
                                  config_digest, jsonable, split_of)


def test_the_split_is_a_function_of_the_segment_alone():
    """Section 7: 20 touches from one randomized base pose must stay in one
    split.  Hashing (run, episode) makes that true by construction and
    reproducible from the manifest, independent of the order episodes
    happened to finish in."""
    a = [split_of("run-a", i) for i in range(500)]
    assert a == [split_of("run-a", i) for i in range(500)]
    assert a != [split_of("run-b", i) for i in range(500)]


def test_the_split_is_eighty_ten_ten():
    assert SPLIT_FRACTIONS == (("train", 0.80), ("val", 0.10), ("test", 0.10))
    c = Counter(split_of("r", i) for i in range(20000))
    for name, frac in SPLIT_FRACTIONS:
        assert c[name] / 20000 == pytest.approx(frac, abs=0.02)


def test_nan_becomes_null_and_never_a_number():
    """An empty contact field has to read as empty."""
    assert jsonable(float("nan")) is None
    assert jsonable(np.float64("nan")) is None
    assert jsonable(None) is None
    assert jsonable(np.array([1.0, 2.5])) == [1.0, 2.5]
    assert jsonable({"a": np.int64(3)}) == {"a": 3}
    assert json.dumps(jsonable({"x": float("nan")})) == '{"x": null}'


def test_the_config_digest_is_order_independent():
    assert config_digest({"a": 1, "b": 2}) == config_digest({"b": 2, "a": 1})
    assert config_digest({"a": 1}) != config_digest({"a": 2})


def test_rows_are_on_disk_before_the_run_ends(tmp_path):
    """The demo's contract: a record is appended AS the run happens, because
    the run that gets interrupted is the one whose data you most want."""
    w = TransitionWriter(tmp_path, "r")
    w.write({"uid": "a"})
    w.write({"uid": "b"})
    lines = (tmp_path / "transitions.jsonl").read_text().splitlines()
    assert [json.loads(l)["uid"] for l in lines] == ["a", "b"]
    w.close()


def test_qpos_rows_are_padded_not_truncated(tmp_path):
    w = TransitionWriter(tmp_path, "r", store_qpos=True)
    assert w.stash_qpos(np.arange(5)) == 0
    assert w.stash_qpos(np.arange(8)) == 1
    assert w.stash_qpos(None) is None
    w.close()
    z = np.load(tmp_path / "qpos.npz")["qpos"]
    assert z.shape == (2, 8)
    assert np.isnan(z[0, 5:]).all()
    assert np.allclose(z[1], np.arange(8))


def test_no_qpos_means_no_companion_file(tmp_path):
    w = TransitionWriter(tmp_path, "r", store_qpos=False)
    assert w.stash_qpos(np.arange(5)) is None
    w.write({"uid": "a"})
    w.close()
    assert not (tmp_path / "qpos.npz").exists()


def test_the_manifest_carries_the_map_once(tmp_path):
    """Room 2 is fixed: the geometry is stored once, and a row carries an id."""
    from data_collect.rooms import build_scene, map_identity
    scene = build_scene("room2")
    w = TransitionWriter(tmp_path, "r")
    w.write_manifest({"map": map_identity("room2", scene)})
    w.close()
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["run_id"] == "r" and m["schema_version"].startswith("room2-belief")
    mm = m["map"]
    assert mm["map_id"] == "room2_8_11_scan"
    assert mm["walls_json"].endswith("room2_scan_walls.json")
    assert len(mm["bodies"]) == len(scene.plan.bodies)
    assert {b["name"] for b in mm["bodies"]} == {b.name
                                                for b in scene.plan.bodies}
