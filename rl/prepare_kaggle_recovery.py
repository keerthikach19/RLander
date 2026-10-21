"""Package a fresh self-contained recovery notebook; never execute training."""

import argparse
import base64
import json
from pathlib import Path

from rl.prepare_kaggle_ablation import source_bundle


def prepare(output):
    payload, digest, source, requirements = source_bundle((
        "rl/kaggle_recovery.py", "rl/prepare_kaggle_recovery.py",
    ))

    def cell(kind, content):
        item = {"cell_type": kind, "id": f"recovery-{len(cells)}", "metadata": {},
                "source": content.splitlines(keepends=True)}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cells = []
    cell("markdown", """# RLander PPO recovery v1 — diagnosis, reward pilot, conditional update-frequency pilot
Enable **GPU** and **Internet**, then Run All in a **fresh Kaggle session**.
This notebook is self-contained; no repository upload is required. Uses GPU 0.

1. Run noisy observation-only PD through W3 on seeds 10000–10099. If success is
   below 80%, run a separately labeled noise-off diagnostic and stop before PPO.
2. Check the new reward on PD, zero-thrust and nominal-hover policies using
   validation seeds 2000–2099. Stop if PD's dense-cost/100 ratio is outside .2–1.
3. Run PPO-only **seed 1 for 960,000 steps**, changing only the reward objective
   from the old Kaggle arm (128 environments, 125 steps, same PPO settings).
4. If final validation success is <=10%, run a **separate** 960,000-step pilot
   with 16 environments: 480 optimizer updates instead of 60. Same reward;
   fresh initialization, no BC. This also changes normalization batch statistics.
5. Evaluate the predeclared final checkpoints through W3 after all arm decisions.
   Intermediate checkpoints at 240k/480k/720k use validation seeds only.

The new objective is **landing_profile v1**, not policy-invariant potential shaping:
physical-unit descent tracking, lateral damping, a time cost, and smooth impact
credit/cost. Exact success remains |vy| < 1 and |vx| < .5 m/s. Dynamics, initial
states and training sensor noise are unchanged. No full-run gate waiver is used.
These are bounded pilots, not a three-seed replication or a promise of success.
The 10000–10099 set has already informed development; it is not a pristine holdout.
W3 reward totals remain baseline; use training manifests for the revised reward.

**Return:** download `europa_ppo_recovery_v1.zip` from Kaggle Output. It includes
results, training curves/CSVs, per-update metrics, checkpoint normalization,
validation snapshots, W3 SQLite and trajectories, logs and source provenance.
If a stage fails, the ZIP still contains the available evidence. Do not rerun the
launch cell in the same session; fresh output directories prevent overwrites.
""")
    cell("code", f"""import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *{requirements!r}, 'matplotlib'], check=True)
""")
    cell("code", f"""import base64, hashlib, io, json, os, zipfile
from pathlib import Path
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
workspace = Path('/kaggle/working/rlander_recovery_v1_source')
workspace.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
    bundle.extractall(workspace)
os.chdir(workspace)
source = json.loads(Path('source_manifest.json').read_text())
for name, expected in source['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('Source commit:', source['commit'], 'Dirty:', source['dirty'])
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
subprocess.run([sys.executable, '-c', 'import jax; print(jax.__version__, jax.devices()); assert all(d.platform == "gpu" for d in jax.devices())'], check=True)
subprocess.run([sys.executable, '-m', 'rl.kaggle_recovery', '--dry-run'], check=True)
""")
    cell("code", """import shutil
from IPython.display import FileLink, display
output = Path('/kaggle/working/europa_ppo_recovery_v1')
# A failure is reported visibly, but we keep packaging the evidence.
try:
    process = subprocess.run([sys.executable, '-u', '-m', 'rl.kaggle_recovery', '--output', str(output)])
    print('Launcher exit code:', process.returncode)
finally:
    if output.exists():
        shutil.make_archive(str(output), 'zip', output)
        display(FileLink(str(output) + '.zip'))
if (output / 'results.json').exists():
    report = json.loads((output / 'results.json').read_text())
    print('Status:', report['status'])
    print(report.get('error', report.get('next_step', report.get('preflight', {}).get('decision', ''))))
    for run in report['runs']:
        print(run['arm'], 'validation success:', run['final_validation_rate'],
              'reload action error:', run['max_action_reload_error'])
    for run in report['frozen']:
        print(run['arm'], 'W3 success:', run['success_rate'],
              'by outcome:', run['by_outcome'], 'contact-only:', run['touchdown'])
""")
    cell("code", """# Curves use completed episodes only; a zero-completion update is not a failure rate.
import pandas as pd
import matplotlib.pyplot as plt
for path in sorted(output.glob('*_training.csv')):
    data = pd.read_csv(path)
    completed = data[data.episodes > 0].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(completed.env_steps, completed.episode_return)
    axes[0].set_title('Sampled training episode return')
    window = completed[['landed', 'crashed', 'timeouts', 'episodes']].rolling(10, min_periods=1).sum()
    for name in ['landed', 'crashed', 'timeouts']:
        axes[1].plot(completed.env_steps, window[name] / window.episodes, label=name)
    axes[1].set_title('Training outcomes (10-update weighted window)')
    axes[1].legend()
    arm = path.name.removesuffix('_training.csv')
    evidence = json.loads((output / (arm + '.json')).read_text())
    validation = evidence['validation']
    axes[2].plot([v['steps'] for v in validation],
                 [v['evaluation']['soft_landing_rate'] for v in validation], marker='o')
    axes[2].set_title('Deterministic validation success')
    axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel('Environment steps')
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
    notebook = {"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    path = output / "rlander_ppo_recovery_v1.ipynb"
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    (output / "source.zip").write_bytes(payload)
    (output / "source_manifest.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(f"Prepared {path.resolve()} ({len(source['files'])} source files; SHA256 {digest})")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/kaggle_ppo_recovery_v1")
    prepare(parser.parse_args().output)
