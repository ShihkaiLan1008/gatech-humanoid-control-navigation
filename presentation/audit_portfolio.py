"""Read-only checks of source files and headline claims; write a separate audit.

Run from any directory with Python 3.11+: python presentation/audit_portfolio.py
This does not run simulations or modify archived research source files.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAV = ROOT / "projects/tactile-navigation"


def disk_path(path):
    return Path("\\\\?\\" + str(path.resolve())) if os.name == "nt" else path


def main():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    patterns = [re.compile(p) for p in (
        rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b", rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b",
        rb"\bAKIA[0-9A-Z]{16}\b", rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        rb"\bxox[baprs]-[A-Za-z0-9-]{15,}\b", rb"\bsk-proj-[A-Za-z0-9_-]{40,}\b",
        rb"\bsk-[A-Za-z0-9]{40,}\b")]
    text_extensions = {".py", ".md", ".json", ".jsonl", ".yml", ".yaml", ".toml", ".ps1",
                       ".sh", ".txt", ".log", ".js", ".mjs", ".csv", ".patch", ".xml", ".svg"}
    scanned, findings = set(), []

    def scan(relative):
        path = disk_path(ROOT / relative)
        if relative in scanned:
            return
        if path.suffix.lower() in {".docx", ".pptx"}:
            with zipfile.ZipFile(path) as package:
                found = any(any(p.search(package.read(name)) for p in patterns)
                            for name in package.namelist() if name.endswith(".xml"))
        elif path.suffix.lower() in text_extensions:
            raw = path.read_bytes()
            found = any(p.search(raw) for p in patterns)
        else:
            return
        scanned.add(relative)
        if found:
            findings.append(relative)  # Never print matched credential values.

    for row in manifest:
        path = disk_path(ROOT / row["path"])
        assert path.stat().st_size == row["bytes"], row["path"]
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == row["sha256"], row["path"]
        scan(row["path"])
    extras = list(ROOT.glob("*.md"))
    for directory in ("docs", "presentation", "assets"):
        extras.extend((ROOT / directory).rglob("*"))
    for path in extras:
        if path.is_file():
            scan(path.relative_to(ROOT).as_posix())
    assert not findings, "Credential-pattern matches require review: " + ", ".join(findings)

    summary = json.loads((NAV / "outputs/sky_results.json").read_text())
    checks = []
    for row in summary["evaluation_records"]:
        record = json.loads((NAV / row["record"]).read_text())
        samples = record["samples"]
        error = math.dist(samples[-1]["truth"][:2], record["goal_xy"])
        reached = error <= record["goal_radius_m"]
        integral = duration = 0.0
        for a, b in zip(samples, samples[1:]):
            dt = b["t"] - a["t"]
            if dt <= 0 or any(s[k] is None for s in (a, b) for k in ("truth", "estimate")):
                continue
            sq = [sum((s["truth"][i] - s["estimate"][i]) ** 2 for i in (0, 1)) for s in (a, b)]
            integral += dt * sum(sq) / 2
            duration += dt
        assert duration > 0
        rmse = math.sqrt(integral / duration)
        success = bool(record["completed"] and reached and not record["fallen"])
        replay = json.loads((NAV / row["record"].replace(".eval.json", ".json")).read_text())
        arrival = [t["arrival"] for t in replay["transitions"] if t.get("arrival") is not None][-1]
        assert math.isclose(error, row["final_goal_error_m"], abs_tol=1e-10)
        assert math.isclose(rmse, row["position_error_rmse_m"], abs_tol=1e-10)
        assert reached == row["goal_reached"] and success == row["success"]
        assert arrival["probability"] == row["final_arrival_probability"]
        assert row["required_arrival_probability"] == 0.95
        assert row["fallen"] == record["fallen"]
        assert sum(bool(p["contact"]) for p in record["probes"]) == row["touch_contacts"]
        checks.append(dict(room=row["room"], seed=row["seed"], physical_arrival=reached,
                           formal_success=success, fallen=record["fallen"], goal_error_cm=100 * error,
                           localization_rmse_cm=100 * rmse, arrival_belief_pct=100 * arrival["probability"]))
    assert len(checks) == len({(r["room"], r["seed"]) for r in checks}) == 4
    assert len(summary["interrupted_attempts"]) == 1
    assert sum(r["physical_arrival"] for r in checks) == 4
    assert sum(r["formal_success"] for r in checks) == 1
    assert not any(r["fallen"] for r in checks)
    result = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_files_verified=len(manifest), source_bytes_verified=sum(r["bytes"] for r in manifest),
                  all_source_hashes_match=True, credential_pattern_scanned_files=len(scanned),
                  credential_pattern_findings=findings,
                  scanner_limit="Known credential patterns in selected text and Office XML; excludes PDF/image text, binary models and unknown patterns.",
                  calculation_checks=checks, completed_trials=4, physical_arrivals=4, formal_successes=1,
                  interrupted_attempts_excluded=1,
                  verification_limit="Artifact integrity and calculations do not independently certify personal mastery, mentor endorsement, or sharing permission.")
    (ROOT / "PORTFOLIO_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
