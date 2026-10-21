"""Build a self-contained notebook from tracked source; no push or launch.

python -m rl.prepare_kaggle_ablation --output outputs/kaggle_ppo_only
"""

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import subprocess
import tomllib
import zipfile
from pathlib import Path


def source_bundle(extra_files=()):
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    files = sorted(p for p in tracked if (
        p in ("train.py", "uv.lock")
        or (p.endswith((".py", ".yaml","sql")) and p.split("/")[0] in
            ("env", "rl", "eval", "storage", "configs"))
    ))
    # Runtime dependencies must be bundled even before newly added files are
    # committed; train.py and the batched W3 path import these modules.
    files = sorted(set(files) | set(extra_files) | {"rl/execution.py", "eval/batched.py"})
    versions = {}
    for package in tomllib.loads((root / "uv.lock").read_text())["package"]:
        versions.setdefault(package["name"], set()).add(package["version"])
    names = ("jax", "flax", "optax", "orbax-checkpoint", "hydra-core", "numpy",
             "pandas", "pyarrow", "sqlalchemy", "python-dotenv")
    # uv.lock can contain Python-specific alternatives. Match the tested local
    # environment and require its versions to occur in the lockfile.
    selected = {name: importlib.metadata.version(name) for name in names}
    if any(selected[name] not in versions[name] for name in names):
        raise RuntimeError("Installed dependency version is absent from uv.lock")
    requirements = [f"{name}[cuda12]=={selected[name]}" if name == "jax"
                    else f"{name}=={selected[name]}" for name in names]
    source = {"commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "diff", "HEAD"], cwd=root)),
        "files": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in files:
            bundle.writestr(name, (root / name).read_bytes())
        bundle.writestr("source_manifest.json", json.dumps(source, indent=2))
    payload = archive.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    return payload, digest, source, requirements


def prepare(output):
    payload, digest, source, requirements = source_bundle()

    def cell(kind, text):
        result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            result.update(execution_count=None, outputs=[])
        return result

    cells = [cell("markdown", """# RLander: PPO-only shaped 2M ablation
Seeds 1–3, no BC. User explicitly waived the failed >10% smoke gate on 2026-10-18
for this arm only. Prior 500k validation was 0 soft / 64 crash / 36 timeout.
The shaping magnitude target remains unmet; weights are deliberately unchanged.
Fixed final 2M checkpoints go through W3 frozen seeds 10000–10099 after all training.
This is frozen-seed evaluation, not a pristine holdout. Evaluation is deterministic.
W3 reward totals are baseline and must not be compared to shaped training totals.
Enable GPU and Internet. Run all cells; download the output ZIP when complete.
"""), cell("code", f"""import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *{requirements!r}], check=True)
"""), cell("code", f"""import base64, hashlib, io, json, os, zipfile
from pathlib import Path
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
workspace = Path('/kaggle/working/rlander_ppo_only_source')
workspace.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
    bundle.extractall(workspace)
os.chdir(workspace)
source = json.loads(Path('source_manifest.json').read_text())
for name, expected in source['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('Source commit:', source['commit'], 'Dirty:', source['dirty'])
subprocess.run([sys.executable, '-c', 'import jax; print(jax.__version__, jax.devices()); assert all(d.platform == "gpu" for d in jax.devices())'], check=True)
"""), cell("code", """import shutil
output = '/kaggle/working/europa_ppo_only_2m'
try:
    subprocess.run([sys.executable, '-u', '-m', 'rl.kaggle_ablation', '--output', output], check=True)
finally:
    if Path(output).exists():
        shutil.make_archive(output, 'zip', output)
report = json.loads(Path(output, 'results.json').read_text())
for row in report['frozen']:
    print(row['training_seed'], {k: row[k] for k in ('soft_landings', 'crashed', 'timeouts')})
""")]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "rlander_ppo_only_2m.ipynb").write_text(json.dumps({
        "cells": cells, "metadata": {"kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }, indent=2), encoding="utf-8")
    (output / "source.zip").write_bytes(payload)
    (output / "source_manifest.json").write_text(json.dumps(source, indent=2) + "\n")
    print(f"Prepared {len(source['files'])} source files; archive SHA256 {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/kaggle_ppo_only")
    prepare(parser.parse_args().output)
