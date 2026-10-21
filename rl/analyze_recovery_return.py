"""Read returned v1 evidence and package a validation-selected PPO parent.

No checkpoint execution, training, or archive extraction is required.
"""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow.parquet as pq


def analyze(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        source = json.loads(archive.read("results.json"))
        if source["status"] != "complete":
            raise ValueError("Expected completed v1 experiment")
        refinement = json.loads(archive.read("more_updates.json"))
        validations = []
        for row in refinement["validation"]:
            e = row["evaluation"]
            outcomes = np.asarray(e["episodes"]["info"]["outcome"])
            vy = np.asarray(e["episodes"]["info"]["touchdown_v"])
            vx = np.asarray(e["episodes"]["info"]["touchdown_vx"])
            components = e["episodes"]["reward_components"]
            returns = sum(np.asarray(v) for v in components.values())
            validations.append({
                "steps": row["steps"], "soft_landings": e["soft_landings"],
                "crashed": e["crashed"], "timeouts": e["timeouts"],
                "mean_return": float(returns.mean()),
                "vertical_only_crashes": int(((outcomes == 2) & (vy >= 1) & (vx < .5)).sum()),
                "group_mean_return": {str(code): float(returns[outcomes == code].mean())
                                      if (outcomes == code).any() else None for code in (1, 2, 3)},
            })
        # Selection uses ONLY the supplied validation rows, never frozen outcomes.
        chosen = max(validations, key=lambda r: (r["soft_landings"], -r["steps"]))
        base = f"more_updates/{PurePosixPath(refinement['run_dir']).name}"
        prefix = base if chosen["steps"] == refinement["actual_steps"] else f"{base}/snapshots/{chosen['steps']}"
        parent_manifest = json.loads(archive.read(prefix + "/manifest.json"))
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as package:
            for name in archive.namelist():
                if (name == prefix + "/manifest.json" or name.startswith(prefix + "/checkpoint/")) and not name.endswith("/"):
                    relative = PurePosixPath(name[len(prefix) + 1:])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Unsafe checkpoint member")
                    package.writestr(relative.as_posix(), archive.read(name))
        final = next(r for r in source["frozen"] if r["arm"] == "more_updates")
        traces = []
        for episode in final["episodes"]:
            name = f"trajectories/{final['run_id']}/{episode['seed']}.parquet"
            traces.append(pq.read_table(io.BytesIO(archive.read(name))).to_pydict())
        crossings = {}
        for height in (100, 30, 10, 5, 1, 0):
            indices = [next(i for i, y in enumerate(t["y"]) if y <= height) for t in traces]
            crossings[str(height)] = {
                "abs_vy_p10_median_p90": np.quantile(
                    [abs(t["vy"][i]) for t, i in zip(traces, indices)], [.1, .5, .9]
                ).tolist(),
            }
        report = {
            "input_zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pd_frozen_success_rate": source["preflight"]["pd_w3"]["success_rate"],
            "more_updates_validation": validations,
            "selected_parent": {**chosen, "run_id": parent_manifest["run"]["id"],
                                "selection": "Highest existing validation success; earliest on tie",
                                "zip_prefix": prefix},
            "reload_errors": {r["arm"]: r["max_action_reload_error"] for r in source["runs"]},
            "v1_final_frozen": [{"arm": r["arm"], "success_rate": r["success_rate"],
                                 "by_outcome": r["by_outcome"]} for r in source["frozen"]],
            "more_updates_final_crossings": crossings,
            "more_updates_final_failures": {
                "vertical_only": sum(abs(t["vy"][-1]) >= 1 and abs(t["vx"][-1]) < .5 for t in traces),
                "lateral_only": sum(abs(t["vy"][-1]) < 1 and abs(t["vx"][-1]) >= .5 for t in traces),
                "both": sum(abs(t["vy"][-1]) >= 1 and abs(t["vx"][-1]) >= .5 for t in traces),
            },
            "inference": "Vertical braking is the observed final failure; learning instability and reward tradeoffs remain hypotheses",
        }
        return report, payload.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_zip")
    parser.add_argument("--output", default="outputs/kaggle_ppo_recovery_v2")
    args = parser.parse_args()
    report, checkpoint = analyze(args.input_zip)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "v1_diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "parent_checkpoint.zip").write_bytes(checkpoint)
    print(json.dumps(report, indent=2))
