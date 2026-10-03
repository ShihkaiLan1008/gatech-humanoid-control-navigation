#!/usr/bin/env python3
"""Show both clusters; optionally collect their results and aggregate locally."""
import argparse
import json
from pathlib import Path
import sys

from submit_batch import pace, remote


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--collect", action="store_true", help="copy results to campaign/collected; keep remote data")
    a = p.parse_args()
    base = a.campaign.resolve()
    hosts = json.loads((base / "hosts.json").read_text())
    receipt = base / "submission.json"
    if not receipt.exists():
        receipt = base / "smoke_submission.json"
    jobs = json.loads(receipt.read_text())
    for key, config in hosts.items():
        rows = [r for r in jobs if r["host"] == key]
        ids = ",".join(str(r[k]) for r in rows for k in ("job_id", "upload_job_id") if k in r)
        print(key.upper(), flush=True)
        result = remote(config["ssh"], ["sacct", "-X", "-j", ids,
                         "--format=JobID,JobName%30,State,Elapsed,Start,NodeList", "--noheader"], capture_output=True)
        print(result.stdout.strip())
        for row in rows:
            folder = "smoke" if receipt.name.startswith("smoke") else "evaluation"
            directory = config["remote_snapshot"] + "/" + folder + "/" + row["room"]
            command = [config["python"], "-c",
                "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); "
                "print(p.read_text() if p.exists() else 'Results pending')", directory + "/status.json"]
            result = remote(config["ssh"], command, capture_output=True)
            text = result.stdout.strip()
            if text.startswith("{"):
                state = json.loads(text)
                trials = state["trials"]
                text = (f"records={sum(r['evaluation'] for r in trials)}/{len(state['expected_seeds'])}; "
                        f"videos={sum(r['video'] for r in trials)}; "
                        f"verified uploads={sum(r['upload_verified'] for r in trials)}")
            print(row["room"], text)
            if a.collect:
                exists = remote(config["ssh"], [config["python"], "-c",
                    "from pathlib import Path; import sys; print(Path(sys.argv[1]).is_dir())", directory],
                    capture_output=True)
                if exists.stdout.strip() != "True":
                    continue
                target = base / "collected" / key / row["room"]
                target.mkdir(parents=True, exist_ok=True)
                pace.run(["rsync", "-az", "--protect-args", "--exclude=.cache/", "--exclude=*.partial.mp4",
                          "--", config["ssh"] + ":" + directory + "/", str(target) + "/"])
    if a.collect:
        from execution.evaluation import write_report
        write_report(list((base / "collected").glob("*/*/runs/*/*.eval.json")), base / "combined_report")
        print("Combined metrics:", base / "combined_report/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
