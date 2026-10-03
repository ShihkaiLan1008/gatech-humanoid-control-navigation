"""Independently audit headline metrics from raw campaign samples."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary = json.loads((ROOT / "outputs/sky_results.json").read_text())
    checks = []
    for row in summary["evaluation_records"]:
        record = json.loads((ROOT / row["record"]).read_text())
        samples = record["samples"]
        last = samples[-1]["truth"]
        error = math.dist(last[:2], record["goal_xy"])
        reached = error <= record["goal_radius_m"]
        integral, duration = 0., 0.
        for a, b in zip(samples, samples[1:]):
            dt = b["t"] - a["t"]
            if dt <= 0 or any(x[k] is None for x in (a, b) for k in ("truth", "estimate")):
                continue
            squared_errors = [sum((x["truth"][i] - x["estimate"][i]) ** 2 for i in (0, 1)) for x in (a, b)]
            integral += dt * sum(squared_errors) / 2
            duration += dt
        rmse = math.sqrt(integral / duration)
        assert math.isclose(error, row["final_goal_error_m"], abs_tol=1e-10)
        assert math.isclose(rmse, row["position_error_rmse_m"], abs_tol=1e-10)
        assert reached == row["goal_reached"]
        assert (record["completed"] and reached and not record["fallen"]) == row["success"]
        assert sum(bool(p["contact"]) for p in record["probes"]) == row["touch_contacts"]
        assert len(record["probes"]) == row["probe_attempts"]
        replay = json.loads((ROOT / row["record"].replace(".eval.json", ".json")).read_text())
        arrival = [t["arrival"] for t in replay["transitions"] if t.get("arrival") is not None][-1]
        assert arrival["probability"] == row["final_arrival_probability"]
        checks.append(dict(room=row["room"], final_goal_error_m=error, localization_rmse_m=rmse,
                           goal_reached=reached, success=row["success"], arrival_probability=arrival["probability"],
                           samples=len(samples), passed=True))
    assert len(checks) == 4
    assert len({(r["room"], r["seed"]) for r in summary["evaluation_records"]}) == 4
    assert len(summary["interrupted_attempts"]) == 1
    result = dict(completed_trials=4, physical_goals=sum(x["goal_reached"] for x in checks),
                  formal_successes=sum(x["success"] for x in checks), checks=checks,
                  interrupted_attempts_separate=True)
    (ROOT / "outputs/sky_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
