"""Honor V5's validation-frozen choice and preserve its PPO-only checkpoint."""

import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

from rl.kaggle_recovery_v5 import select_candidate


def analyze(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        report = json.loads(archive.read("results.json"))
        if report["status"] != "complete":
            raise ValueError("Expected completed v5 evidence")
        selected = select_candidate(report["runs"])
        selection = report["selection"]
        if not selection["selection_frozen_before_fresh_evaluation"] or selected != selection["candidate"]:
            raise ValueError("Frozen candidate does not match validation")
        if selected["evaluation"]["seeds"] != list(range(8000, 8800)):
            raise ValueError("Unexpected v5 validation scope")
        stored = selected["checkpoint"].removesuffix("/checkpoint")
        matches = [n for n in archive.namelist() if n.endswith("/manifest.json")
                   and stored.endswith("/" + n.removesuffix("/manifest.json"))]
        if len(matches) != 1:
            raise ValueError("Cannot locate selected checkpoint")
        prefix = matches[0].removesuffix("/manifest.json")
        manifest = json.loads(archive.read(matches[0]))
        if manifest["evaluation"] != selected["evaluation"]:
            raise ValueError("Checkpoint validation does not match selection")
        payload = io.BytesIO()
        count = 0
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as package:
            for name in archive.namelist():
                if (name == matches[0] or name.startswith(prefix + "/checkpoint/")) and not name.endswith("/"):
                    relative = PurePosixPath(name[len(prefix) + 1:])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Unsafe checkpoint member")
                    package.writestr(relative.as_posix(), archive.read(name))
                    count += relative.parts[0] == "checkpoint"
        if not count:
            raise ValueError("Missing checkpoint files")
        prior = json.loads(archive.read("parent_provenance.json"))["selection"]["ancestral_steps"]
        fresh = []
        for row in report["fresh_evaluation"]:
            evaluation = row["evaluation"]
            if sorted(ep["seed"] for ep in evaluation["episodes"]) != list(range(70000, 72000)):
                raise ValueError("Unexpected v5 test seeds")
            fresh.append({"label": row["label"], **{k: evaluation[k] for k in (
                "n_success", "n_episodes", "success_rate", "ci95_low", "ci95_high", "by_outcome")}})
        return {
            "input_zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selected_parent": {
                "arm": selected["arm"], "additional_steps": selected["additional_steps"],
                "ancestral_steps": prior + selected["additional_steps"],
                "validation_soft_landings": selected["evaluation"]["soft_landings"],
                "validation_episodes": selected["evaluation"]["n_episodes"],
                "selection": "v5 validation-frozen PPO-only candidate; no test-score reselection",
                "zip_prefix": prefix, "run_id": manifest["run"]["id"],
            },
            "fresh_evaluation": fresh,
            "paired_comparison": json.loads(archive.read("paired_comparison.json")),
            "target_check": json.loads(archive.read("target_check.json")),
            "next_hypothesis": "Practice late-descent states from parent PPO flights, mixed with full starts; no demonstrations.",
        }, payload.getvalue()
