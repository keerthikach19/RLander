"""Freeze returned Kaggle evidence and validation-selected models; never train."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("v1", "v2_gpu", "v3", "v4", "v5", "v6", "v7")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative_checkpoint(checkpoint):
    return checkpoint.split("/kaggle/working/", 1)[1].split("/", 1)[1]


def preserve(archive, checkpoint, destination):
    prefix = relative_checkpoint(checkpoint).removesuffix("/checkpoint") + "/"
    hashes = {}
    for member in archive.namelist():
        if not member.startswith(prefix) or member.endswith("/"):
            continue
        relative = member[len(prefix):]
        if relative != "manifest.json" and not relative.startswith("checkpoint/"):
            continue
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError("Unsafe checkpoint member")
        payload = archive.read(member)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        hashes[relative] = digest(payload)
    if "manifest.json" not in hashes or len(hashes) < 3:
        raise ValueError(f"Incomplete checkpoint: {checkpoint}")
    write_json(destination / "sha256.json", hashes)
    return hashes


def freeze(downloads, release, demo):
    release.mkdir(parents=True, exist_ok=True)
    demo.mkdir(parents=True, exist_ok=True)
    registry, validation, history = [], [], []
    for version in VERSIONS:
        source = downloads / f"europa_ppo_recovery_{version}.zip"
        with zipfile.ZipFile(source) as archive:
            result = json.loads(archive.read("results.json"))
            if result["status"] != "complete":
                raise ValueError(f"{version} did not complete")
            expected = ROOT / "outputs" / f"kaggle_ppo_recovery_{version}" / "source_manifest.json"
            if json.loads(archive.read("source_manifest.json")) != json.loads(expected.read_text()):
                raise ValueError(f"{version} source mismatch")
            for run in result["runs"]:
                candidates = run["validation"]
                best = min(candidates, key=lambda item: (
                    -item["evaluation"]["soft_landings"],
                    item.get("additional_steps", item.get("steps")),
                ))
                if "selected" in run:
                    if best["checkpoint"] != run["selected"]["checkpoint"]:
                        raise ValueError("Saved selection disagrees with validation")
                    best = run["selected"]
                else:
                    steps = best["steps"]
                    checkpoint = run["checkpoint"] if steps == run["actual_steps"] else (
                        f"{run['run_dir']}/snapshots/{steps}/checkpoint"
                    )
                    best = dict(best, checkpoint=checkpoint, additional_steps=steps)
                local = Path("checkpoints") / version / run["arm"]
                hashes = preserve(archive, best["checkpoint"], release / local)
                evaluation = best["evaluation"]
                registry.append({
                    "version": version, "arm": run["arm"],
                    "additional_steps": best["additional_steps"],
                    "validation_successes": evaluation["soft_landings"],
                    "validation_episodes": evaluation["n_episodes"],
                    "validation_rate": evaluation["soft_landing_rate"],
                    "checkpoint": (local / "checkpoint").as_posix(),
                    "original_checkpoint": best["checkpoint"],
                    "archive_sha256": digest(source.read_bytes()),
                    "checkpoint_files": len(hashes),
                    "selection": "validation success, then earliest step; never test score",
                })
                for item in candidates:
                    ev = item["evaluation"]
                    steps = item.get("additional_steps", item.get("steps"))
                    validation.append({
                        "version": version, "arm": run["arm"], "additional_steps": steps,
                        "episodes": ev["n_episodes"], "landed": ev["soft_landings"],
                        "crashed": ev["crashed"], "timeouts": ev["timeouts"],
                        "success_rate": ev["soft_landing_rate"],
                        "selected_for_run": steps == best["additional_steps"],
                    })
            selected = result.get("selection", {}).get("candidate")
            if selected:
                matches = [item for item in result["fresh_evaluation"]
                           if item.get("checkpoint") == selected["checkpoint"]]
                if matches:
                    ev = matches[0]["evaluation"]
                    history.append({
                        "version": version, "additional_steps": selected["additional_steps"],
                        "test_episodes": ev["n_episodes"], "test_successes": ev["n_success"],
                        "test_success_rate": ev["success_rate"],
                        "seed_start": min(e["seed"] for e in ev["episodes"]),
                        "seed_end": max(e["seed"] for e in ev["episodes"]),
                    })
            evidence = release / "evidence" / version
            evidence.mkdir(parents=True, exist_ok=True)
            for name in ("selection.json", "source_manifest.json", "parent_provenance.json",
                         "paired_comparison.json", "target_check.json"):
                if name in archive.namelist():
                    (evidence / name).write_bytes(archive.read(name))
            if version == "v7":
                make_demo(archive, result, source, release, demo)

    write_json(release / "checkpoint_registry.json", registry)
    write_json(demo / "checkpoint_registry.json", registry)
    pd.DataFrame(validation).to_csv(demo / "validation.csv", index=False)
    pd.DataFrame(history).to_csv(demo / "history.csv", index=False)
    write_json(release / "freeze.json", {
        "training_status": "stopped by user; no further training",
        "release_model": "v7/seed_15/+80000; selection frozen before test",
        "archived_run_winners": len(registry),
        "validation_snapshots": len(validation),
        "selection_note": "One saved controller is used for all flights. No per-flight switching.",
    })
    print(json.dumps({"run_winners": len(registry), "validation_snapshots": len(validation),
                      "release": str(release), "demo": str(demo)}))


def make_demo(archive, result, source, release, demo):
    selection = result["selection"]
    if not selection["selection_frozen_before_fresh_evaluation"]:
        raise ValueError("Selection was not frozen before test")
    selected = selection["candidate"]
    fresh = {item["label"]: item for item in result["fresh_evaluation"]}
    if fresh["selected_candidate"]["checkpoint"] != selected["checkpoint"]:
        raise ValueError("Tested checkpoint differs from selection")
    preserve(archive, selected["checkpoint"], release / "selected")
    source_zip = ROOT / "outputs/kaggle_ppo_recovery_v7/source.zip"
    (release / "training_source.zip").write_bytes(source_zip.read_bytes())
    ppo_episodes = fresh["selected_candidate"]["evaluation"]["episodes"]
    replay_seeds = sorted(
        seed for outcome, count in ((1, 6), (2, 3), (3, 1))
        for seed in sorted(ep["seed"] for ep in ppo_episodes if ep["outcome"] == outcome)[:count]
    )
    summary = {
        "release": "V7 frozen PPO-only", "seeds": [90000, 91999],
        "success_criteria": "Ground contact with |vy| < 1.0 m/s and |vx| < 0.5 m/s",
        "selection": {
            "arm": selected["arm"], "additional_steps": selected["additional_steps"],
            "validation_successes": selected["evaluation"]["soft_landings"],
            "validation_episodes": selected["evaluation"]["n_episodes"],
            "validation_seeds": [11000, 11999], "frozen_before_test": True,
            "checkpoint": selected["checkpoint"],
        },
        "source_archive_sha256": digest(source.read_bytes()),
        "evaluation_backend": "All three policies evaluated on Kaggle GPU",
        "policies": {},
    }
    episode_rows = []
    for key, label, source_label in (
        ("ppo", "PPO · selected V7", "selected_candidate"),
        ("parent", "PPO · V6 parent", "initial_parent"),
        ("pd", "PD baseline", "pd_baseline"),
    ):
        ev = fresh[source_label]["evaluation"]
        seeds = [ep["seed"] for ep in ev["episodes"]]
        if sorted(seeds) != list(range(90000, 92000)) or ev["sensor_noise_scale"] != 1:
            raise ValueError("Mismatched test conditions")
        record = {field: ev[field] for field in (
            "n_episodes", "n_success", "success_rate", "ci95_low", "ci95_high",
            "fuel_mean", "touchdown", "by_outcome", "sensor_noise_scale",
        )}
        record.update(label=label, replays=[])
        for ep in ev["episodes"]:
            member = f"trajectories/{ev['run_id']}/{ep['seed']}.parquet"
            payload = archive.read(member)
            frame = pd.read_parquet(io.BytesIO(payload))
            final = frame.iloc[-1]
            if not np.isfinite(frame.to_numpy()).all() or not (frame.t.diff().iloc[1:] > 0).all():
                raise ValueError("Nonfinite or unordered trajectory")
            contact = final.y <= 0
            physical_outcome = (1 if abs(final.vy) < 1 and abs(final.vx) < .5 else 2) if contact else 3
            if physical_outcome != ep["outcome"]:
                raise ValueError("Outcome disagrees with physical trajectory")
            episode_rows.append({
                "policy": key, "seed": ep["seed"], "outcome": ep["outcome_name"],
                "duration_s": final.t, "final_height_m": final.y,
                "final_vy_mps": final.vy, "final_vx_mps": final.vx,
                "fuel_kg": frame.mass.iloc[0] - final.mass,
            })
            if ep["seed"] in replay_seeds:
                relative = f"trajectories/{key}/{ep['seed']}.parquet"
                target = demo / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                record["replays"].append({
                    "seed": ep["seed"], "outcome": ep["outcome_name"], "file": relative,
                    "sha256": digest(payload),
                })
        if record["n_success"] != sum(ep["outcome"] == 1 for ep in ev["episodes"]):
            raise ValueError("Aggregate count mismatch")
        summary["policies"][key] = record
    for run in result["runs"]:
        name = f"{run['arm']}_training.csv"
        (demo / name).write_bytes(archive.read(name))
    pd.DataFrame(episode_rows).to_csv(demo / "episodes.csv", index=False)
    write_json(demo / "summary.json", summary)
    write_json(release / "summary.json", summary)
    print("Verified all 6,000 test trajectories against physical outcomes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--release", type=Path, default=ROOT / "outputs/final_release")
    parser.add_argument("--demo", type=Path, default=ROOT / "app/release_data")
    args = parser.parse_args()
    freeze(args.downloads, args.release, args.demo)
