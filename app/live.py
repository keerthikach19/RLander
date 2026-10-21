"""Optional local inference. Never trains or changes the frozen checkpoint."""

import hashlib
import json
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


def simulate(controller, seed):
    import pandas as pd

    from eval.batched import collect
    from eval.metrics import outcome_name

    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must fit an unsigned 32-bit integer")
    policy, body, vehicle = load_controller(controller)
    batch, execution = collect(policy, [seed], body, vehicle)
    records, _, info = batch[0]
    return pd.DataFrame(records), {
        "controller": controller, "seed": seed, "outcome": outcome_name(int(info["outcome"])),
        "backend": execution["backend"], "device": execution["device"],
    }
