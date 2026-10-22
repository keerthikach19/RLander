"""Optional local inference. Never trains or changes the frozen checkpoint."""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=2)
def load_controller(controller):
    from eval.run_eval import load_body_config, load_vehicle_config, make_policy_fn

    body = load_body_config("europa")
    vehicle = load_vehicle_config("europa_lander")
    if controller not in ("ppo", "pd"):
        raise ValueError("Controller must be ppo or pd")
    checkpoint = None
    if controller == "ppo":
        selected = ROOT / "outputs/final_release/selected"
        hashes = json.loads((selected / "sha256.json").read_text())
        for name, expected in hashes.items():
            if hashlib.sha256((selected / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen checkpoint integrity check failed: {name}")
        checkpoint = str(selected / "checkpoint")
    policy = make_policy_fn(
        "ppo" if controller == "ppo" else "pd_fallback",
        checkpoint_path=checkpoint, body_params=body, vehicle_params=vehicle,
    )
    return policy, body, vehicle


def validate_initial_conditions(initial_conditions):
    """Validate physical coordinates/speeds before loading or compiling a model."""
    if initial_conditions is None:
        return None
    if set(initial_conditions) != {"x", "y", "vx", "vy"}:
        raise ValueError("Starting conditions must contain x, y, vx and vy")
    values = {name: float(initial_conditions[name]) for name in ("x", "y", "vx", "vy")}
    if not all(math.isfinite(value) and abs(value) <= 1e6 for value in values.values()):
        raise ValueError("Starting coordinates and speeds must be finite and within ±1,000,000")
    if values["y"] <= 0:
        raise ValueError("Starting altitude must be above the surface")
    return values


def simulate(controller, seed, *, initial_conditions=None):
    import pandas as pd

    from eval.batched import collect
    from eval.metrics import outcome_name

    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must fit an unsigned 32-bit integer")
    requested = validate_initial_conditions(initial_conditions)
    policy, body, vehicle = load_controller(controller)
    batch, execution = collect(
        policy, [seed], body, vehicle,
        initial_conditions=tuple(requested.values()) if requested is not None else None,
    )
    records, _, info = batch[0]
    metadata = {
        "controller": controller, "seed": seed, "outcome": outcome_name(int(info["outcome"])),
        "backend": execution["backend"], "device": execution["device"],
        "dry_mass_kg": float(vehicle["dry_mass_kg"]),
        "app_root": str(ROOT),
        "start_mode": "custom" if requested is not None else "randomized",
        "requested_initial_conditions": requested,
        "initial_state": {name: float(value) for name, value in pd.DataFrame(records).iloc[0].items()},
        "final_state": {name: float(value) for name, value in pd.DataFrame(records).iloc[-1].items()},
    }
    if controller == "ppo":
        selected = ROOT / "outputs/final_release/selected"
        manifest_bytes = (selected / "manifest.json").read_bytes()
        metadata.update(
            checkpoint=str(selected / "checkpoint"),
            model_run_id=json.loads(manifest_bytes)["run"]["id"],
            checkpoint_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        )
    return pd.DataFrame(records), metadata
