"""Inspect returned v2 GPU evidence and package the best validation checkpoint."""

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
        result = json.loads(archive.read("results.json"))
        if result["status"] != "complete":
            raise ValueError("Expected a completed v2 GPU run")
        candidates = [{**candidate, "arm": run["arm"]}
                      for run in result["runs"] for candidate in run["validation"]]
        scopes = {tuple(row["evaluation"]["seeds"]) for row in candidates}
        if scopes != {tuple(range(2000, 2100))}:
            raise ValueError("Candidate validation seeds differ from the v2 protocol")
        # Fresh evaluation is deliberately not read until after parent selection.
        selected = max(candidates, key=lambda r: (
            r["evaluation"]["soft_landings"], -r["additional_steps"],
        ))
        stored_path = selected["checkpoint"].removesuffix("/checkpoint")
        manifests = [n for n in archive.namelist() if n.endswith("/manifest.json")
                     and stored_path.endswith("/" + n.removesuffix("/manifest.json"))]
        if len(manifests) != 1:
            raise ValueError("Cannot uniquely locate selected checkpoint in returned ZIP")
        prefix = manifests[0].removesuffix("/manifest.json")
        manifest = json.loads(archive.read(manifests[0]))
        if manifest["evaluation"] != selected["evaluation"]:
            raise ValueError("Selected validation differs from its checkpoint manifest")
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name in archive.namelist():
                if name == manifests[0] or name.startswith(prefix + "/checkpoint/"):
                    relative = PurePosixPath(name[len(prefix) + 1:])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Unsafe checkpoint path")
                    if not name.endswith("/"):
                        bundle.writestr(relative.as_posix(), archive.read(name))
        if not any(n.startswith(prefix + "/checkpoint/") for n in archive.namelist()):
            raise ValueError("Selected checkpoint has no files")

        fresh = []
        for row in result["fresh_evaluation"]:
            e = row["evaluation"]
            fresh.append({"label": row["label"], **{key: e[key] for key in (
                "n_episodes", "n_success", "success_rate", "ci95_low", "ci95_high", "by_outcome",
            )}})
        selected_fresh = next(r["evaluation"] for r in result["fresh_evaluation"]
                              if r["checkpoint"] == selected["checkpoint"])
        failures = {"vertical_only": 0, "lateral_only": 0, "both": 0, "timeouts": 0}
        crossings = {str(h): {"landed": [], "crashed": []} for h in (100, 30, 10, 3, 0)}
        for episode in selected_fresh["episodes"]:
            suffix = f"trajectories/{selected_fresh['run_id']}/{episode['seed']}.parquet"
            trace = pq.read_table(io.BytesIO(archive.read(suffix))).to_pydict()
            code = episode["outcome"]
            if code == 3:
                failures["timeouts"] += 1
                continue
            if code == 2:
                vy_bad, vx_bad = abs(trace["vy"][-1]) >= 1, abs(trace["vx"][-1]) >= .5
                if not (vy_bad or vx_bad):
                    raise ValueError("Crash trajectory does not fail either landing threshold")
                failures["both" if vy_bad and vx_bad else
                         "vertical_only" if vy_bad else "lateral_only"] += 1
            for height, groups in crossings.items():
                index = next((i for i, y in enumerate(trace["y"]) if y <= float(height)), None)
                if index is not None:
                    groups["landed" if code == 1 else "crashed"].append(abs(trace["vy"][index]))
        crossings = {h: {group: {"n": len(values), "vy_p10_median_p90":
                                np.quantile(values, [.1, .5, .9]).tolist() if values else None}
                         for group, values in groups.items()} for h, groups in crossings.items()}
        execution = {}
        for name in archive.namelist():
            if name.endswith("_execution.jsonl"):
                execution[name] = [json.loads(line) for line in archive.read(name).decode().splitlines()
                                   if json.loads(line)["phase"] == "completed"]
        training = [row for rows in execution.values() for row in rows if row["label"] == "ppo_training"]
        gpu_verified = bool(training) and all(
            r.get("gpu_verified") and r["backend"] == "gpu" and r["compiled_devices"] == ["cuda:0"]
            for rows in execution.values() for r in rows
        )
        report = {
            "input_zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selected_parent": {
                "arm": selected["arm"], "additional_steps": selected["additional_steps"],
                "ancestral_steps": (manifest["config"]["actor_initialization"]["parent_steps"]
                                    + manifest["actual_steps"]),
                "validation_soft_landings": selected["evaluation"]["soft_landings"],
                "validation_episodes": selected["evaluation"]["n_episodes"],
                "selection": "highest v2 validation success across both arms; earliest additional step on ties",
                "zip_prefix": prefix, "run_id": manifest["run"]["id"],
            },
            "fresh_evaluation": fresh, "selected_fresh_failures": failures,
            "selected_crossings": crossings, "gpu_workloads_verified": gpu_verified,
            "training_execution": training,
            "execution_scope": "GPU execution times exclude compilation/cache loading, host I/O and startup",
            "reload_errors": {r["arm"]: r["max_action_reload_error"] for r in result["runs"]},
            "next_iteration_rationale": (
                "Retain successful reward and parent; test conservative PPO updates across three "
                "fine-tuning seeds. Remaining failures are chiefly vertical. A smaller learning rate "
                "is a stability hypothesis, not a demonstrated cure."
            ),
        }
        return report, payload.getvalue()
