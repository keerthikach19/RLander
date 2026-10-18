"""Diagnose saved W3 trajectories without rerunning or selecting a policy.

uv run --with matplotlib python -m rl.diagnose_touchdown
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def distribution(values):
    return dict(zip(
        ("min", "p25", "median", "p75", "p95", "max"),
        np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist(),
    ))


def diagnose(results_path, trajectory_dir, output_dir, training_seed=3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_path, trajectory_dir = Path(results_path), Path(trajectory_dir)
    source = json.loads(results_path.read_text(encoding="utf-8"))
    frozen = next(r for r in source["frozen"] if r["training_seed"] == training_seed)
    if sorted(r["seed"] for r in frozen["episodes"]) != list(range(10000, 10100)):
        raise ValueError("Expected exactly the 100 frozen episode seeds")
    rows, traces = [], []
    for episode in sorted(frozen["episodes"], key=lambda r: r["seed"]):
        path = trajectory_dir / f"{episode['seed']}.parquet"
        data = {k: np.asarray(v) for k, v in pq.read_table(path).to_pydict().items()}
        if data["y"][-1] != 0 or not np.all(data["y"][:-1] > 0):
            raise ValueError(f"Expected first ground contact at final sample: {path}")
        vy, vx = float(data["vy"][-1]), float(data["vx"][-1])
        expected = 1 if abs(vy) < 1 and abs(vx) < .5 else 2
        if expected != episode["outcome"]:
            raise ValueError(f"Contact thresholds disagree with W3: {path}")
        vertical, lateral = abs(vy) >= 1, abs(vx) >= .5
        reason = ("both" if vertical and lateral else "vertical_only" if vertical
                  else "lateral_only" if lateral else "soft")
        # First descending crossing: sampled state, no interpolation or new rollout.
        crossings = {}
        for altitude in (500, 100, 30, 10, 5, 1):
            i = np.flatnonzero(data["y"] <= altitude)[0]
            crossings[str(altitude)] = {
                k: float(data[k][i]) for k in ("t", "y", "vy", "vx", "throttle")
            }
        rows.append({
            "seed": episode["seed"], "outcome": expected, "failure_axis": reason,
            "contact_t_s": float(data["t"][-1]), "contact_vy_mps": vy,
            "contact_vx_mps": vx, "crossings_m": crossings,
            "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        traces.append(data)
    crashes = [r for r in rows if r["outcome"] == 2]
    soft = [r for r in rows if r["outcome"] == 1]
    report = {
        "training_seed": training_seed, "checkpoint_steps": frozen["actual_steps"],
        "protocol": "Existing deterministic W3 frozen-seed evaluation; no new rollouts or tuning",
        "contact_definition": "First terminal RK4 sample, y clamped to ground; not substep-interpolated impact",
        "source_results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "counts": {"soft": len(soft), "crash": len(crashes), "timeout": 0},
        "failure_axes": {axis: sum(r["failure_axis"] == axis for r in crashes)
                         for axis in ("vertical_only", "lateral_only", "both")},
        "groups": {}, "episodes": rows,
    }
    for name, group in (("soft", soft), ("crash", crashes)):
        speeds = [abs(r["contact_vy_mps"]) for r in group]
        report["groups"][name] = {
            "contact_abs_vy_mps": distribution(speeds),
            "contact_abs_vx_mps": distribution([abs(r["contact_vx_mps"]) for r in group]),
            "contact_t_s": distribution([r["contact_t_s"] for r in group]),
            "vertical_speed_bins": {
                "below_1": sum(v < 1 for v in speeds),
                "1_to_2": sum(1 <= v < 2 for v in speeds),
                "2_to_5": sum(2 <= v < 5 for v in speeds),
                "5_or_more": sum(v >= 5 for v in speeds),
            },
            "crossing_abs_vy_mps": {
                str(h): distribution([abs(r["crossings_m"][str(h)]["vy"]) for r in group])
                for h in (500, 100, 30, 10, 5, 1)
            },
        }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "touchdown.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    colors = {1: "#147d72", 2: "#c24c35"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bins = np.linspace(0, max(abs(r["contact_vy_mps"]) for r in rows) * 1.05, 24)
    for label, group, outcome in (("Soft", soft, 1), ("Crash", crashes, 2)):
        axes[0, 0].hist([abs(r["contact_vy_mps"]) for r in group], bins=bins,
                        alpha=.65, color=colors[outcome], label=f"{label} ({len(group)})")
        axes[0, 1].scatter([abs(r["contact_vy_mps"]) for r in group],
                           [abs(r["contact_vx_mps"]) for r in group],
                           color=colors[outcome], alpha=.7, s=20)
    axes[0, 0].axvline(1, color="black", linestyle="--", label="Vertical limit: 1 m/s")
    axes[0, 0].set(xlabel="Contact |vy| (m/s)", ylabel="Episodes", title="Touchdown speed distribution")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axvline(1, color="black", linestyle="--")
    axes[0, 1].axhline(.5, color="black", linestyle="--")
    axes[0, 1].set(xlabel="Contact |vy| (m/s)", ylabel="Contact |vx| (m/s)", title="Both landing thresholds matter")
    for row, data in zip(rows, traces):
        color = colors[row["outcome"]]
        axes[1, 0].plot(data["t"], data["y"], color=color, alpha=.18, linewidth=.8)
        axes[1, 0].scatter(data["t"][-1], 0, color=color, s=8)
        mask = data["t"] >= data["t"][-1] - 20
        axes[1, 1].plot(data["t"][mask] - data["t"][-1], data["vy"][mask],
                        color=color, alpha=.25, linewidth=.8)
    axes[1, 0].set(xlabel="Episode time (s)", ylabel="Altitude (m)", title="Full trajectories; dots mark contact")
    axes[1, 1].axhline(-1, color="black", linestyle="--")
    axes[1, 1].set(xlabel="Seconds relative to contact", ylabel="Vertical velocity (m/s)", title="Final 20 seconds, aligned at contact")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle(f"Europa BC+PPO seed {training_seed}, 500k | deterministic | frozen seeds 10000–10099")
    fig.tight_layout()
    fig.savefig(output_dir / "touchdown.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="rl/validation/stability_20261018/results.json")
    parser.add_argument("--trajectory-dir", default="ckpts/europa_stability_20261018/w3_trajectories/3")
    parser.add_argument("--output-dir", default="rl/validation/stability_20261018/seed3")
    parser.add_argument("--training-seed", type=int, default=3)
    args = parser.parse_args()
    diagnose(args.results, args.trajectory_dir, args.output_dir, args.training_seed)
