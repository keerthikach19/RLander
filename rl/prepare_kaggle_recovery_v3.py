"""Build the next self-contained Kaggle notebook; never launch training."""

import argparse
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

from rl.analyze_refinement_return import analyze
from rl.prepare_kaggle_ablation import source_bundle


def prepare(input_zip, output, *, analyzer=analyze, version="v3", introduction=None,
            extra_files=(), diagnosis_name="v2_diagnosis.json", validation_episodes=200):
    diagnosis, parent_zip = analyzer(input_zip)
    source_zip, digest, source, requirements = source_bundle((
        "rl/kaggle_recovery.py", "rl/kaggle_recovery_v2.py", "rl/kaggle_recovery_v3.py",
        "rl/prepare_kaggle_recovery_v3.py", "rl/analyze_refinement_return.py",
    ) + tuple(extra_files))
    with zipfile.ZipFile(io.BytesIO(parent_zip)) as archive:
        hashes = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()}
    provenance = {"input_zip_sha256": diagnosis["input_zip_sha256"],
                  "selection": diagnosis["selected_parent"], "files": hashes,
                  "parent_zip_sha256": hashlib.sha256(parent_zip).hexdigest()}
    cells = []

    def cell(kind, text):
        if kind == "code":
            for old, new in (
                ("rl.kaggle_recovery_v3", f"rl.kaggle_recovery_{version}"),
                ("rlander_recovery_v3_source", f"rlander_recovery_{version}_source"),
                ("europa_ppo_recovery_v3", f"europa_ppo_recovery_{version}"),
                ("v2_diagnosis.json", diagnosis_name),
                ("Deterministic validation (200 seeds)", f"Deterministic validation ({validation_episodes} seeds)"),
                ("Additional transitions after v2 parent", "Additional transitions after bundled parent"),
            ):
                text = text.replace(old, new)
        elif introduction is not None:
            text = introduction
        item = {"cell_type": kind, "id": f"{version}-{len(cells)}", "metadata": {},
                "source": text.splitlines(keepends=True)}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", """# RLander PPO v3 — conservative refinement from the best v2 model
**Fresh Kaggle session · GPU accelerator + Internet · Run All.**
Code and the selected checkpoint are embedded. No separate dataset is needed.

The returned v2 result was 82/100 on validation and **157/200 (78.5%)** on
fresh seeds for the selected success-aligned model, compared with 106/200 (53%)
for its parent. Its 43 crashes comprised 42 vertical-only and one lateral-only
failure; no episode timed out. The selected v2 model, not its weaker final
checkpoint, is the starting policy. Selection used v2 validation, not fresh scores.

The returned execution logs verified Tesla T4 / cuda:0 for PPO and evaluation.
The two 320k training calls recorded 3.47s and 3.46s of GPU execution; their
compile/cache-loading times were 68.47s and 7.65s. These are component timings
from that run, **not end-to-end notebook runtimes or a promise for this run**.

## This experiment
1. Replay the archived 100 validation seeds. Stop if the parent differs from
   its archived result by more than two landings. Measure the parent on a
   new 200-seed validation set (4000–4199).
2. Run three **160k PPO refinements**, seeds **2, 3, 4**, from the same parent:
   **480k new transitions total**. Reduce the learning rate to **1e-5**,
   annealed to zero. Keep the success-aligned reward, 16×125 rollouts, clip .1,
   entropy zero, frozen parent observation statistics, and action std reset to
   .10 with bounds [.05,.15]. Actor and critic are imported; optimizer, RNG and
   environments are fresh. This is not exact resumption or BC.
3. Save snapshots every 20k and validate them after each training call. Select
   each seed's best model by success count; earliest step wins ties. The parent
   remains eligible. Choose an overall candidate using the same rule, then
   training seed order 2,3,4 on remaining ties. Freeze that choice before testing.
4. Evaluate the parent and each seed's selected policy on **new seeds
   40000–40499 (500 episodes)**. Identical checkpoints are evaluated once.
   These scores never change the candidate selection. Report both newly gained
   and lost landings versus the parent on these paired seeds. Final checkpoints
   remain in the ZIP with validation results, but receive no extra fresh evaluation.

Previously inspected seeds 2000–2099, 10000–10099 and 30000–30199 are not the
new final test set. Full noise/domain randomization, nominal 2000m starts and
strict |vy|<1 / |vx|<.5 m/s stay unchanged. No PD correction is applied.
The three seeds measure variation in **fine-tuning one shared ancestor**,
not independent learning from scratch. Smaller updates are a hypothesis for
preserving/refining skill; this does not isolate the effect of learning rate,
prove reliability, or promise a gain over 78.5%.

GPU 0 only. Actual compiled workloads and result arrays are checked; CPU fallback
and disabled JIT are rejected. Evaluation stays batched. Logs distinguish
compilation/cache loading, synchronized execution and CPU output writing.
CPU activity and brief or low GPU utilization are expected for some phases.
The author has not executed this v3 notebook or its training.

**Return `europa_ppo_recovery_v3.zip` from Output**, including on a diagnostic stop
or failure. It contains parent/snapshots, metrics, execution logs, provenance,
fresh trajectories, paired comparison and the selection made before testing.
""")
    cell("code", f"""import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *{requirements!r}, 'matplotlib'], check=True)
""")
    cell("code", f"""import base64, hashlib, io, json, os, zipfile
from pathlib import Path
source_payload = base64.b64decode({base64.b64encode(source_zip).decode()!r})
parent_payload = base64.b64decode({base64.b64encode(parent_zip).decode()!r})
assert hashlib.sha256(source_payload).hexdigest() == {digest!r}
assert hashlib.sha256(parent_payload).hexdigest() == {provenance['parent_zip_sha256']!r}
workspace = Path('/kaggle/working/rlander_recovery_v3_source')
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
for name, expected in source['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
for name, expected in provenance['files'].items():
    assert hashlib.sha256((parent_dir / name).read_bytes()).hexdigest() == expected, name
Path('parent_provenance.json').write_text(json.dumps(provenance, indent=2))
Path('v2_diagnosis.json').write_text({json.dumps(diagnosis, indent=2)!r})
print('Selected parent:', provenance['selection'])
os.environ.update(CUDA_VISIBLE_DEVICES='0', JAX_PLATFORMS='cuda',
                  JAX_DISABLE_JIT='false', XLA_PYTHON_CLIENT_PREALLOCATE='false')
subprocess.run([sys.executable, '-c', 'from rl.execution import require_gpu; print("Available device; workload checks follow:", require_gpu())'], check=True)
subprocess.run([sys.executable, '-m', 'rl.kaggle_recovery_v3', '--dry-run'], check=True)
""")
    cell("code", """import shutil
from IPython.display import FileLink, display
output = Path('/kaggle/working/europa_ppo_recovery_v3')
try:
    process = subprocess.run([sys.executable, '-u', '-m', 'rl.kaggle_recovery_v3',
                              '--parent', str(parent_dir / 'checkpoint'), '--output', str(output)])
finally:
    if output.exists():
        shutil.make_archive(str(output), 'zip', output)
        display(FileLink(str(output) + '.zip'))
if (output / 'results.json').exists():
    report = json.loads((output / 'results.json').read_text())
    print('Status:', report['status'], report.get('error', ''))
    if 'selection' in report:
        selected = report['selection']['candidate']
        print('Candidate chosen BEFORE fresh test:', selected['arm'],
              'additional steps:', selected['additional_steps'],
              'validation:', selected['evaluation']['soft_landing_rate'])
    for row in report['fresh_evaluation']:
        e = row['evaluation']
        print(row['label'], 'landed:', e['n_success'], '/', e['n_episodes'],
              'success:', e['success_rate'], 'Wilson 95% CI:', (e['ci95_low'], e['ci95_high']))
    paired = output / 'paired_comparison.json'
    if paired.exists():
        print('Paired comparison:', json.loads(paired.read_text()))
if process.returncode:
    raise RuntimeError('Run failed; download the ZIP above for partial results and logs.')
""")
    cell("code", """import pandas as pd
import matplotlib.pyplot as plt
for path in sorted(output.glob('*_training.csv')):
    arm = path.name.removesuffix('_training.csv')
    summary = output / (arm + '.json')
    if not summary.exists():
        continue
    row = json.loads(summary.read_text())
    data = pd.read_csv(path)
    completed = data[data.episodes > 0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(completed.env_steps, completed.episode_return)
    axes[0].set_title('Sampled training return')
    totals = data.episodes.rolling(10, min_periods=1).sum().replace(0, float('nan'))
    for label in ['landed', 'crashed', 'timeouts']:
        axes[1].plot(data.env_steps, data[label].rolling(10, min_periods=1).sum() / totals, label=label)
    axes[1].set_title('Training outcomes (10-update weighted window)')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[2].plot([c['additional_steps'] for c in row['validation']],
                 [c['evaluation']['soft_landing_rate'] for c in row['validation']], marker='o')
    axes[2].axvline(row['selected']['additional_steps'], color='green', linestyle='--', label='selected')
    axes[2].set_title('Deterministic validation (200 seeds)')
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel('Additional transitions after v2 parent')
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
    path = output / f"rlander_ppo_recovery_{version}.ipynb"
    path.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "name": "python3", "language": "python"}},
        "nbformat": 4, "nbformat_minor": 5}, indent=2), encoding="utf-8")
    (output / "source.zip").write_bytes(source_zip)
    (output / "parent_checkpoint.zip").write_bytes(parent_zip)
    for name, value in (("source_manifest.json", source), ("parent_provenance.json", provenance),
                        (diagnosis_name, diagnosis)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    print("Prepared:", path.resolve())
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_zip")
    parser.add_argument("--output", default="outputs/kaggle_ppo_recovery_v3")
    args = parser.parse_args()
    prepare(args.input_zip, args.output)
