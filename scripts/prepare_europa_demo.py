"""Package the frozen v4 winner and existing evaluations; never train or select on test."""

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(archive_path, pd_path, destination, release):
    archive_path, pd_path = Path(archive_path), Path(pd_path)
    destination, release = Path(destination), Path(release)
    with zipfile.ZipFile(archive_path) as archive:
        results = json.loads(archive.read("results.json"))
        assert results["status"] == "complete"
        selection = results["selection"]
        assert selection["selection_frozen_before_fresh_evaluation"]
        candidate = selection["candidate"]
        fresh = {item["label"]: item for item in results["fresh_evaluation"]}
        assert fresh["margin_selected"]["checkpoint"] == candidate["checkpoint"]
        expected_source = Path("outputs/kaggle_ppo_recovery_v4/source_manifest.json")
        assert json.loads(archive.read("source_manifest.json")) == json.loads(
            expected_source.read_text(encoding="utf-8")
        )
        policies = {
            "ppo": ("PPO · selected v4", fresh["margin_selected"]["evaluation"]),
            "parent": ("PPO · parent", fresh["initial_parent"]["evaluation"]),
            "pd": ("PD controller", json.loads(pd_path.read_text(encoding="utf-8"))),
        }
        # Six landings, three crashes, and the timeout, in seed order.
        # Replays illustrate outcomes; the headline rate uses all 1,000 episodes.
        selected_episodes = policies["ppo"][1]["episodes"]
        replay_seeds = []
        for outcome, limit in ((1, 6), (2, 3), (3, 1)):
            replay_seeds.extend(sorted(
                ep["seed"] for ep in selected_episodes if ep["outcome"] == outcome
            )[:limit])
        summary = {
            "title": "RLander · Europa",
            "seeds": [60000, 60999],
            "success_criteria": "Ground contact with |vy| < 1.0 m/s and |vx| < 0.5 m/s",
            "selection": {
                "arm": candidate["arm"],
                "additional_steps": candidate["additional_steps"],
                "validation_successes": candidate["evaluation"]["soft_landings"],
                "validation_episodes": 400,
                "validation_seeds": [6000, 6399],
                "checkpoint": candidate["checkpoint"],
                "frozen_before_test": True,
            },
            "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "policies": {},
        }
        for key, (label, evaluation) in policies.items():
            episodes = {ep["seed"]: ep for ep in evaluation["episodes"]}
            assert sorted(episodes) == list(range(60000, 61000))
            assert evaluation["n_episodes"] == 1000
            assert evaluation["n_success"] == sum(ep["outcome"] == 1 for ep in episodes.values())
            record = {field: evaluation[field] for field in (
                "n_episodes", "n_success", "success_rate", "ci95_low", "ci95_high",
                "fuel_mean", "touchdown", "by_outcome", "sensor_noise_scale",
            )}
            assert record["sensor_noise_scale"] == 1.0
            record.update(label=label, replays=[])
            for seed in replay_seeds:
                ep = episodes[seed]
                if key == "pd":
                    frame = pd.read_parquet(ep["parquet_path"])
                else:
                    member = f"trajectories/{evaluation['run_id']}/{seed}.parquet"
                    frame = pd.read_parquet(io.BytesIO(archive.read(member)))
                filename = f"trajectories/{key}/{seed}.parquet"
                target = destination / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(target, index=False)
                record["replays"].append({
                    "seed": seed, "outcome": ep["outcome_name"], "file": filename,
                })
            summary["policies"][key] = record
        checkpoint = candidate["checkpoint"].split("/europa_ppo_recovery_v4/", 1)[1]
        snapshot = checkpoint.rsplit("/", 1)[0] + "/"
        hashes = {}
        for member in archive.namelist():
            if not member.startswith(snapshot) or member.endswith("/"):
                continue
            relative = member[len(snapshot):]
            target = (release / "selected" / relative).resolve()
            assert target.is_relative_to((release / "selected").resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.read(member)
            target.write_bytes(payload)
            hashes[relative] = hashlib.sha256(payload).hexdigest()
        assert hashes and (release / "selected/checkpoint").is_dir()
        write_json(release / "checkpoint_sha256.json", hashes)
        for name in ("selection.json", "paired_comparison.json", "parent_provenance.json"):
            (release / name).write_bytes(archive.read(name))
        write_json(destination / "summary.json", summary)
        write_json(release / "summary.json", summary)
        shutil.copy2("outputs/kaggle_ppo_recovery_v4/source.zip", release / "training_source.zip")
        print(json.dumps({key: value["success_rate"]
                          for key, value in summary["policies"].items()}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--pd-evaluation", default="outputs/europa_release/pd_baseline/evaluation.json")
    parser.add_argument("--destination", default="app/demo_data")
    parser.add_argument("--release", default="outputs/europa_release")
    args = parser.parse_args()
    prepare(args.archive, args.pd_evaluation, args.destination, args.release)
