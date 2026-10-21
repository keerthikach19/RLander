"""Read v3 evidence while honoring its selection frozen before fresh testing."""

import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow.parquet as pq

from rl.kaggle_recovery_v2 import choose_best
from rl.kaggle_recovery_v3 import select_candidate


def analyze(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        result = json.loads(archive.read("results.json"))
        if result["status"] != "complete":
            raise ValueError("Expected completed v3 evidence")
        for run in result["runs"]:
            if run["selected"] != choose_best(run["validation"]):
                raise ValueError("Run selection disagrees with validation")
        selected = select_candidate(result["runs"])
        frozen = result["selection"]
        if not frozen["selection_frozen_before_fresh_evaluation"] or selected != frozen["candidate"]:
            raise ValueError("Frozen candidate does not match validation selection")
        if tuple(selected["evaluation"]["seeds"]) != tuple(range(4000, 4200)):
            raise ValueError("Unexpected v3 validation scope")
        stored = selected["checkpoint"].removesuffix("/checkpoint")
        matches = [n for n in archive.namelist() if n.endswith("/manifest.json")
                   and stored.endswith("/" + n.removesuffix("/manifest.json"))]
        if len(matches) != 1:
            raise ValueError("Cannot locate the selected parent unambiguously")
        prefix = matches[0].removesuffix("/manifest.json")
        manifest = json.loads(archive.read(matches[0]))
        if manifest["evaluation"] != selected["evaluation"]:
            raise ValueError("Parent manifest validation mismatch")
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as package:
            for name in archive.namelist():
                if name == matches[0] or name.startswith(prefix + "/checkpoint/"):
                    relative = PurePosixPath(name[len(prefix) + 1:])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Unsafe checkpoint member")
                    if not name.endswith("/"):
                        package.writestr(relative.as_posix(), archive.read(name))
        if not any(n.startswith(prefix + "/checkpoint/") for n in archive.namelist()):
            raise ValueError("Missing selected checkpoint files")

        fresh = result["fresh_evaluation"]
        initial = next(r["evaluation"] for r in fresh if r["label"] == "initial_parent")
        chosen = next(r["evaluation"] for r in fresh if r["checkpoint"] == selected["checkpoint"])
        before = {r["seed"]: r["outcome"] == 1 for r in initial["episodes"]}
        after = {r["seed"]: r["outcome"] == 1 for r in chosen["episodes"]}
        if set(before) != set(range(40000, 40500)) or before.keys() != after.keys():
            raise ValueError("Unexpected fresh evaluation seed scope")
        paired = {
            "both_landed": sum(before[s] and after[s] for s in before),
            "newly_landed": sum(not before[s] and after[s] for s in before),
            "lost_landings": sum(before[s] and not after[s] for s in before),
            "both_failed": sum(not before[s] and not after[s] for s in before),
        }
        failures = {"vertical_only": 0, "lateral_only": 0, "both": 0, "timeouts": 0}
        crossings = {str(h): {"landed": [], "crashed": []} for h in (30, 10, 3, 0)}
        for episode in chosen["episodes"]:
            name = f"trajectories/{chosen['run_id']}/{episode['seed']}.parquet"
            trace = pq.read_table(io.BytesIO(archive.read(name))).to_pydict()
            if episode["outcome"] == 3:
                failures["timeouts"] += 1
                continue
            vy_bad, vx_bad = abs(trace["vy"][-1]) >= 1., abs(trace["vx"][-1]) >= .5
            if (episode["outcome"] == 2) != (vy_bad or vx_bad):
                raise ValueError("Trajectory outcome disagrees with touchdown thresholds")
            if episode["outcome"] == 2:
                failures["both" if vy_bad and vx_bad else "vertical_only" if vy_bad else "lateral_only"] += 1
            for height, groups in crossings.items():
                index = next((i for i, h in enumerate(trace["y"]) if h <= float(height)), None)
                if index is not None:
                    groups["landed" if episode["outcome"] == 1 else "crashed"].append(abs(trace["vy"][index]))
        summary = [{**{k: r["evaluation"][k] for k in (
            "n_success", "n_episodes", "success_rate", "ci95_low", "ci95_high", "by_outcome")},
            "label": r["label"]} for r in fresh]
        execution = []
        for name in archive.namelist():
            if name.endswith("_execution.jsonl"):
                for line in archive.read(name).decode().splitlines():
                    row = json.loads(line)
                    if row["label"] == "ppo_training" and row["phase"] == "completed":
                        execution.append(row)
        prior = json.loads(archive.read("parent_provenance.json"))["selection"]["ancestral_steps"]
        return {
            "input_zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selected_parent": {
                "arm": selected["arm"], "additional_steps": selected["additional_steps"],
                "ancestral_steps": prior + selected["additional_steps"],
                "validation_soft_landings": selected["evaluation"]["soft_landings"],
                "validation_episodes": selected["evaluation"]["n_episodes"],
                "selection": "honor v3 validation-frozen candidate; never rank by fresh results",
                "zip_prefix": prefix, "run_id": manifest["run"]["id"],
            },
            "fresh_evaluation": summary, "paired_parent_comparison": paired,
            "selected_fresh_failures": failures,
            "selected_crossings": {h: {g: np.quantile(v, [.1, .5, .9]).tolist() if v else None
                                      for g, v in groups.items()} for h, groups in crossings.items()},
            "training_execution": execution,
            "gpu_training_verified": len(execution) == len(result["runs"]) and all(
                e.get("gpu_verified") and e["compiled_devices"] == ["cuda:0"] for e in execution),
            "next_hypothesis": "Earlier slowing and stronger lateral damping near ground may improve safety margins; test against unchanged reward.",
        }, payload.getvalue()
