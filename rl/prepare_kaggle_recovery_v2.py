"""Build v2 notebook with source and the selected v1 parent embedded. No launch."""

import argparse
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

from rl.analyze_recovery_return import analyze
from rl.prepare_kaggle_ablation import source_bundle


def prepare(input_zip, output):
    diagnosis, parent_zip = analyze(input_zip)
    source_zip, source_digest, source, requirements = source_bundle((
        "rl/kaggle_recovery.py", "rl/kaggle_recovery_v2.py",
        "rl/prepare_kaggle_recovery_v2.py", "rl/analyze_recovery_return.py",
        "rl/execution.py", "eval/batched.py",
    ))
    with zipfile.ZipFile(io.BytesIO(parent_zip)) as archive:
        hashes = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()}
    provenance = {"input_zip_sha256": diagnosis["input_zip_sha256"],
                  "selection": diagnosis["selected_parent"], "files": hashes,
                  "parent_zip_sha256": hashlib.sha256(parent_zip).hexdigest()}
    cells = []

    def cell(kind, content):
        item = {"cell_type": kind, "id": f"v2-{len(cells)}", "metadata": {},
                "source": content.splitlines(keepends=True)}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", """# RLander PPO recovery v2 GPU correction
**Fresh Kaggle session · GPU + Internet · Run All. No separate dataset upload.**
This notebook embeds both code and the validation-selected PPO parent from your
returned ZIP. It does not rerun v1. GPU 0 only; 640,000 new transitions total.

**Correction to previous claims:** the old "Starting ... on GPU 0" message only
described configuration. It was not runtime proof. The screenshot was at
evaluation; it cannot establish the training backend. The old W3 evaluator
did synchronize with the host after every step of every individual episode.

This version batches all 200 W3 episodes in one compiled call, reuses validation
compilation across checkpoints, explicitly places workload inputs on the selected
GPU, checks the actual compiled PPO/evaluation executable's device shardings,
and checks completed result arrays before transferring them to the host.
Every worker refuses CPU fallback or disabled JIT. Logs and `*_execution.jsonl`
separate compilation/cache loading from synchronized execution; manifests report
training execution throughput. Live logs and `gpu_telemetry.csv` show progress.
CPU activity during compilation, checkpoint I/O, Parquet writing and plotting is
expected. This small 16-environment PPO workload need not saturate a GPU.
Only GPU 0 is used, even if Kaggle exposes two GPUs.

**Validation scope:** local CPU tests check correctness and failure guards.
Kaggle/CUDA execution and speed have not been tested by the notebook author.
No GPU speedup, completion time, utilization level or landing improvement is
promised. Your run will produce the hardware evidence. Timings synchronize
completion as recommended in the
[JAX profiling documentation](https://docs.jax.dev/en/latest/201/profiling.html).

Your v1 `more_updates` policy reached **49/100 validation landings at 720k**,
then **7/100 at 960k** and **5/100 W3 landings**. All 95 final W3 crashes failed
only the vertical limit. PD landed 100/100; checkpoint reload differences were
below 1e-6. This motivates refinement, not another blind from-scratch restart.

1. Replay the selected parent on validation seeds 2000–2099. Stop if deterministic
   replay differs by more than two landings from its archived result. Also compare
   sampled policies at original and reduced action noise with matched sensor keys.
2. Run two **320k PPO fine-tuning** schedules from the **same parent**, seed 1:
   - `stable_control`: existing continuous reward.
   - `success_aligned`: identical dense costs; contact score becomes
     0.5 × continuous score + 50 × (true success − crash).
3. Both use 16 environments, learning rate 3e-5 annealed to zero, clip .1,
   no entropy bonus, action std .10 constrained to [.05,.15], and frozen parent
   observation normalization. Actor and critic are imported; optimizer,
   environments, RNG, and exploration are reset. This is **not an exact resume**.
4. Save snapshots every 40k and evaluate them after training finishes.
   Select by success count, earliest checkpoint on ties.
   The starting checkpoint is eligible, so a worse final model cannot replace it
   merely because training finished. This protects validation performance, not a
   guarantee of generalization. Retain final checkpoints for regression reporting.
5. Only after both selections, evaluate initial, selected, and final policies on
   **fresh seeds 30000–30199** (200 episodes each; identical checkpoints reused).
   These seeds are never used for checkpoint selection in this notebook.

Success remains |vy| < 1 and |vx| < .5 m/s, full noise and randomization, nominal
2000m starts. There is no BC or controller intervention in PPO actions. The paired
arms isolate the new terminal reward under the same stabilization settings; they
do not isolate individual optimizer changes or measure independent seed variance.
W3 reward totals remain baseline; training manifests contain the actual objective.

**Download `europa_ppo_recovery_v2_gpu.zip` from Output and return it.** It includes
the parent, all snapshots, metrics/curves, sampled diagnostics, fresh evaluations,
SQLite, trajectories and source/checkpoint provenance. Available evidence is
packaged even on failure. No claim of improved PPO performance is made in advance.
""")
    cell("code", f"""import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *{requirements!r}, 'matplotlib'], check=True)
""")
    cell("code", f"""import base64, hashlib, io, json, os, zipfile
from pathlib import Path
source_payload = base64.b64decode({base64.b64encode(source_zip).decode()!r})
parent_payload = base64.b64decode({base64.b64encode(parent_zip).decode()!r})
assert hashlib.sha256(source_payload).hexdigest() == {source_digest!r}
assert hashlib.sha256(parent_payload).hexdigest() == {provenance['parent_zip_sha256']!r}
workspace = Path('/kaggle/working/rlander_recovery_v2_gpu_source')
workspace.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(source_payload)) as archive:
    archive.extractall(workspace)
parent_dir = workspace / 'inputs' / 'ppo_parent'
parent_dir.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(parent_payload)) as archive:
    archive.extractall(parent_dir)
os.chdir(workspace)
source = json.loads(Path('source_manifest.json').read_text())
provenance = json.loads({json.dumps(provenance)!r})
for name, digest in source['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
for name, digest in provenance['files'].items():
    assert hashlib.sha256((parent_dir / name).read_bytes()).hexdigest() == digest, name
Path('parent_provenance.json').write_text(json.dumps(provenance, indent=2))
Path('v1_diagnosis.json').write_text({json.dumps(diagnosis, indent=2)!r})
print('Parent:', provenance['selection'])
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['JAX_DISABLE_JIT'] = 'false'
subprocess.run([sys.executable, '-c', 'from rl.execution import require_gpu; print("Available device (workload verification follows):", require_gpu())'], check=True)
subprocess.run([sys.executable, '-m', 'rl.kaggle_recovery_v2', '--dry-run'], check=True)
""")
    cell("code", """import shutil
from IPython.display import FileLink, display
output = Path('/kaggle/working/europa_ppo_recovery_v2_gpu')
try:
    process = subprocess.run([sys.executable, '-u', '-m', 'rl.kaggle_recovery_v2',
                              '--parent', str(parent_dir / 'checkpoint'), '--output', str(output)])
    print('Launcher exit code:', process.returncode)
finally:
    if output.exists():
        shutil.make_archive(str(output), 'zip', output)
        display(FileLink(str(output) + '.zip'))
if (output / 'results.json').exists():
    report = json.loads((output / 'results.json').read_text())
    print('Status:', report['status'], report.get('error', ''))
    for row in report['runs']:
        print(row['arm'], 'selected additional steps:', row['selected']['additional_steps'],
              'validation success:', row['selected']['evaluation']['soft_landing_rate'])
    for row in report['fresh_evaluation']:
        e = row['evaluation']
        print(row['label'], 'fresh success:', e['success_rate'],
              '95% CI:', (e['ci95_low'], e['ci95_high']), 'contact-only:', e['touchdown'])
if process.returncode:
    raise RuntimeError('Run failed; download the ZIP above for logs and partial evidence.')
""")
    cell("code", """import pandas as pd
import matplotlib.pyplot as plt
for path in sorted(output.glob('*_training.csv')):
    arm = path.name.removesuffix('_training.csv')
    data = pd.read_csv(path)
    completed = data[data.episodes > 0]
    row = json.loads((output / (arm + '.json')).read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(completed.env_steps, completed.episode_return)
    axes[0].set_title('Sampled training return (arm-specific reward)')
    for label in ['throttle_std', 'angle_std']:
        axes[1].plot(data.env_steps, data[label], label=label)
    axes[1].set_title('Exploration standard deviation')
    axes[1].legend()
    axes[2].plot([c['additional_steps'] for c in row['validation']],
                 [c['evaluation']['soft_landing_rate'] for c in row['validation']], marker='o')
    axes[2].axvline(row['selected']['additional_steps'], color='green', linestyle='--', label='selected')
    axes[2].set_title('Deterministic validation success')
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel('Additional steps after PPO warm start')
        ax.grid(alpha=.2)
    fig.suptitle(arm)
    fig.tight_layout()
    fig.savefig(output / (arm + '_curves.png'), dpi=150)
    plt.show()
if output.exists():
    shutil.make_archive(str(output), 'zip', output)
    display(FileLink(str(output) + '.zip'))
""")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "rlander_ppo_recovery_v2_gpu.ipynb"
    path.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "name": "python3", "language": "python"}},
        "nbformat": 4, "nbformat_minor": 5}, indent=2), encoding="utf-8")
    (output / "source.zip").write_bytes(source_zip)
    (output / "parent_checkpoint.zip").write_bytes(parent_zip)
    for filename, value in (("source_manifest.json", source), ("parent_provenance.json", provenance),
                            ("v1_diagnosis.json", diagnosis)):
        (output / filename).write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(f"Prepared {path.resolve()} with parent at {diagnosis['selected_parent']['steps']} steps")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_zip")
    parser.add_argument("--output", default="outputs/kaggle_ppo_recovery_v2_gpu")
    args = parser.parse_args()
    prepare(args.input_zip, args.output)
