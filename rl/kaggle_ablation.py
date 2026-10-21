"""Run only the authorized no-BC 2M ablation, seeds 1-3, on Kaggle GPUs.

python -m rl.kaggle_ablation --dry-run
python -m rl.kaggle_ablation

Final 2M checkpoints are predeclared for W3 frozen evaluation, after all training.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def command(seed, output):
    return [
        sys.executable, "-u", "train.py", "--config-path", "rl/configs",
        "--config-name", "kaggle_europa_2m", f"seed={seed}",
        f"output_root={output}/seed_{seed}",
        f"hydra.run.dir={output}/hydra_seed_{seed}",
    ]


def launch(output, dry_run=False):
    output = Path(output).resolve()
    if dry_run:
        print(json.dumps([command(seed, output) for seed in (1, 2, 3)], indent=2))
        return
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("This full-run launcher is restricted to Kaggle")
    gpu_indices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
    ).splitlines()
    if not gpu_indices:
        raise RuntimeError("A Kaggle GPU accelerator is required")
    gpu_indices = [g.strip() for g in gpu_indices[:2]]
    # Refuse to overwrite or accidentally relaunch a completed experiment.
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "protocol": "PPO-only; shaped reward; seeds 1,2,3; fixed 2M endpoint; deterministic W3 frozen evaluation after all training",
        "smoke_gate": "User-authorized negative ablation waiver; no-BC smoke did not pass",
        "status": "training", "runs": [], "frozen": [],
    }

    def write_report():
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    write_report()
    (output / "packages.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    ))
    (output / "hardware.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    if Path("source_manifest.json").exists():
        (output / "source_manifest.json").write_bytes(Path("source_manifest.json").read_bytes())

    def train_seed(seed, gpu):
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
                       "JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        print(f"Training PPO-only seed {seed} on GPU {gpu}", flush=True)
        with (output / f"seed_{seed}.log").open("w") as log:
            subprocess.run(command(seed, output), env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        manifests = list((output / f"seed_{seed}").glob("*/manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError("Expected one completed checkpoint per training seed")
        manifest = json.loads(manifests[0].read_text())
        if manifest["status"] != "complete" or manifest["actual_steps"] != 2_000_000:
            raise RuntimeError("Incomplete ablation checkpoint")
        return {"seed": seed, "run_dir": str(manifests[0].parent),
                "config": manifest["config"], "evaluation": manifest["evaluation"]}

    try:
        # One process per GPU; seed 3 uses GPU 0 once the first batch finishes.
        for offset in range(0, 3, len(gpu_indices)):
            seeds = (1, 2, 3)[offset:offset + len(gpu_indices)]
            with ThreadPoolExecutor(max_workers=len(gpu_indices)) as pool:
                futures = [pool.submit(train_seed, seed, gpu)
                           for seed, gpu in zip(seeds, gpu_indices)]
                for future in futures:
                    report["runs"].append(future.result())
                    write_report()
        report["status"] = "evaluating"
        write_report()
        for run in report["runs"]:
            result_path = output / f"frozen_seed_{run['seed']}.json"
            subprocess.run(
                [sys.executable, "-u", "-m", "rl.kaggle_ablation", "--evaluate-run",
                 run["run_dir"], "--output", str(output), "--result", str(result_path)],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_indices[0],
                     "JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
                check=True,
            )
            report["frozen"].append(json.loads(result_path.read_text()))
            write_report()
        report["status"] = "complete"
        write_report()
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        write_report()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/kaggle/working/europa_ppo_only_2m")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluate-run")
    parser.add_argument("--result")
    args = parser.parse_args()
    if args.evaluate_run:
        from rl.stability_europa import frozen_evaluation

        result = frozen_evaluation(args.evaluate_run, args.output)
        Path(args.result).write_text(json.dumps(result, indent=2) + "\n")
    else:
        launch(args.output, args.dry_run)
