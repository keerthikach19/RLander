"""Package the validation-frozen v4 winner, never a test-selected alternative."""

import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq

from rl.kaggle_recovery_v2 import choose_best
from rl.kaggle_recovery_v4 import select_candidates


def analyze(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        result = json.loads(archive.read("results.json"))
        if result["status"] != "complete":
            raise ValueError("Expected completed v4 evidence")
        for run in result["runs"]:
            if run["selected"] != choose_best(run["validation"]):
                raise ValueError("Run selection disagrees with validation")
        selection = select_candidates(result["runs"])
        if selection != result["selection"]:
            raise ValueError("Frozen selection disagrees with validation")
        selected = selection["candidate"]
        if selected["evaluation"]["seeds"] != list(range(6000, 6400)):
            raise ValueError("Unexpected v4 validation seeds")
        stored = selected["checkpoint"].removesuffix("/checkpoint")
        matches = [n for n in archive.namelist() if n.endswith("/manifest.json")
                   and stored.endswith("/" + n.removesuffix("/manifest.json"))]
        if len(matches) != 1:
            raise ValueError("Cannot locate the selected checkpoint unambiguously")
        prefix = matches[0].removesuffix("/manifest.json")
        manifest = json.loads(archive.read(matches[0]))
        if manifest["evaluation"] != selected["evaluation"]:
            raise ValueError("Selected checkpoint manifest does not match validation")
        payload = io.BytesIO()
        checkpoint_files = 0
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as package:
            for name in archive.namelist():
                if (name == matches[0] or name.startswith(prefix + "/checkpoint/")) and not name.endswith("/"):
                    relative = PurePosixPath(name[len(prefix) + 1:])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Unsafe checkpoint member")
                    package.writestr(relative.as_posix(), archive.read(name))
                    checkpoint_files += relative.parts[0] == "checkpoint"
        if not checkpoint_files:
            raise ValueError("Missing checkpoint files")
        chosen = next(row["evaluation"] for row in result["fresh_evaluation"]
                      if row["checkpoint"] == selected["checkpoint"])
        if sorted(ep["seed"] for ep in chosen["episodes"]) != list(range(60000, 61000)):
            raise ValueError("Unexpected v4 test seeds")
        failures = {"vertical_only": 0, "lateral_only": 0, "both": 0, "timeouts": 0}
        for ep in chosen["episodes"]:
            if ep["outcome"] == 1:
                continue
            if ep["outcome"] == 3:
                failures["timeouts"] += 1
                continue
            trace = pq.read_table(io.BytesIO(archive.read(
                f"trajectories/{chosen['run_id']}/{ep['seed']}.parquet"
            )), columns=["vy", "vx"]).to_pydict()
            vy_bad, vx_bad = abs(trace["vy"][-1]) >= 1., abs(trace["vx"][-1]) >= .5
            if not (vy_bad or vx_bad):
                raise ValueError("Crash disagrees with touchdown thresholds")
            failures["both" if vy_bad and vx_bad else "vertical_only" if vy_bad else "lateral_only"] += 1
        prior = json.loads(archive.read("parent_provenance.json"))["selection"]["ancestral_steps"]
        return {
            "input_zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selected_parent": {
                "arm": selected["arm"], "additional_steps": selected["additional_steps"],
                "ancestral_steps": prior + selected["additional_steps"],
                "validation_soft_landings": selected["evaluation"]["soft_landings"],
                "validation_episodes": selected["evaluation"]["n_episodes"],
                "selection": "v4 validation-frozen candidate; no test-score reselection",
                "zip_prefix": prefix, "run_id": manifest["run"]["id"],
            },
            "selected_fresh_result": {k: chosen[k] for k in (
                "n_success", "n_episodes", "success_rate", "ci95_low", "ci95_high")},
            "selected_fresh_failures": failures,
            "next_hypothesis": "Lower exploration and smaller, less noisy PPO updates may preserve precise landing control.",
        }, payload.getvalue()
