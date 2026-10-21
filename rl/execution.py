"""Measured execution of the actual compiled workload, with optional GPU enforcement."""

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import jax


def event(label, phase, **details):
    row = {"timestamp": datetime.now(UTC).isoformat(),
           "pid": os.getpid(), "label": label, "phase": phase, **details}
    if os.environ.get("RLANDER_DEVICE_LOG"):
        with Path(os.environ["RLANDER_DEVICE_LOG"]).open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
    return row


def require_gpu():
    devices = jax.devices()
    if jax.default_backend() != "gpu" or len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"GPU execution required: expected one visible CUDA GPU, got {devices}")
    if jax.config.jax_disable_jit:
        raise RuntimeError("GPU execution requires JIT; JAX_DISABLE_JIT is enabled")
    return devices[0]


def execute(fn, args, *, label):
    """Place inputs, compile, inspect executable shardings, run and synchronize.

    Timings separate tracing/compilation from execution. The latter includes
    dispatch and synchronization, but excludes checkpoint/trajectory host I/O.
    """
    strict = os.environ.get("RLANDER_REQUIRE_GPU") == "1"
    device = require_gpu() if strict else jax.devices()[0]
    args = jax.device_put(args, device)
    jax.block_until_ready(args)
    event(label, "compiling_or_loading_cached_executable", device=str(device),
          device_kind=device.device_kind, backend=jax.default_backend())
    start = time.perf_counter()
    compiled = fn.lower(*args).compile()
    compile_seconds = time.perf_counter() - start
    shardings = jax.tree_util.tree_leaves((compiled.input_shardings, compiled.output_shardings))
    targets = {d for sharding in shardings for d in sharding.device_set}
    if not targets or targets != {device}:
        raise RuntimeError(f"{label}: unexpected compiled execution devices: {targets}")
    event(label, "executing", compiled_devices=sorted(map(str, targets)),
          compile_or_cache_seconds=compile_seconds)
    start = time.perf_counter()
    result = compiled(*args)
    jax.block_until_ready(result)
    execution_seconds = time.perf_counter() - start
    leaves = [x for x in jax.tree_util.tree_leaves(result) if isinstance(x, jax.Array)]
    if not leaves or any(x.devices() != {device} for x in leaves):
        raise RuntimeError(f"{label}: result arrays are not all on the selected device")
    evidence = event(label, "completed", backend=jax.default_backend(),
                     device=str(device), device_kind=device.device_kind,
                     compiled_devices=sorted(map(str, targets)),
                     output_array_count=len(leaves),
                     compile_or_cache_seconds=compile_seconds,
                     execution_seconds=execution_seconds,
                     gpu_verified=device.platform == "gpu")
    return result, evidence
