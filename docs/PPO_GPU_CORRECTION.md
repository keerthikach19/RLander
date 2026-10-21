# GPU execution correction and claim audit

The previous message `Starting ... on GPU 0` was unconditional. It described
the subprocess configuration, not measured execution. That was insufficient
evidence for an assertion that the workload had actually executed on the GPU.

The supplied screenshot shows the `evaluate` stage. It does not establish
whether the earlier PPO training ran on CPU. Source inspection found that the
old launcher already set `CUDA_VISIBLE_DEVICES=0` and `JAX_PLATFORMS=cuda`, and
the full PPO rollout/update loop was JIT compiled. The confirmed bottleneck
was W3 evaluation: Python iterated through episodes and converted `done_flag`
to a host boolean after every timestep, synchronizing device work repeatedly.
CPU compilation and checkpoint/trajectory writing were also included in stages
that the old log labeled "on GPU".

## Implemented corrections

- Every Kaggle worker requires exactly one visible GPU and enabled JIT.
- The actual PPO, validation and W3 input arrays are explicitly device-placed.
  The compiled executable's input/output shardings must all target the selected
  device. Completed output arrays must remain on it. A CPU fallback is an error.
- Each workload logs compile/cache-loading time separately from synchronized
  execution time. PPO manifests include execution throughput, excluding compile
  and file writing time. This is one run measurement, not a warmed benchmark.
- W3 compiles the episode batch with `vmap` and `while_loop`. Random-key
  splitting and first-terminal stopping preserve the serial W3 protocol.
  One batch result is transferred to the host for Parquet/SQLite serialization.
- Evaluator weights are dynamic inputs. Saved-checkpoint validation reuses the
  same compiled evaluator. Persistent compilation caching is configured across
  workers, although a cache hit is not guaranteed.
- Live worker logs identify phases. NVIDIA samples are saved in
  `gpu_telemetry.csv`; sampled utilization can miss short GPU activity.
  Mandatory executable/output checks provide the separate device evidence.
- A failed worker now causes the notebook cell to fail after packaging partial
  results, instead of only printing its nonzero exit code.

## Other statements audited

- Prior local test counts describe **CPU correctness tests**, not GPU validation.
  No Kaggle execution, CUDA performance test or full training run was performed
  by the author while preparing this correction.
- The experiment uses **only GPU 0**, not both Kaggle GPUs. Its 16-environment
  PPO configuration is deliberately unchanged to preserve the learning
  experiment. High or sustained GPU utilization is not promised.
- CPU work for compilation, checkpoint serialization, database writes,
  trajectory files, dependency installation and plotting remains expected.
- "Validation every 40k" means snapshots are saved at those steps and evaluated
  **after the compiled training call completes**, not live early stopping.
- The two 320k schedules are PPO fine-tuning from one shared 720k parent,
  with fresh optimizer/RNG/environments and reset exploration. They are not
  exact resumption, BC, independent training seeds, or 640k from-scratch training.
- The archived 49/100 result is validation success, not fresh holdout success.
  Best-checkpoint selection includes the parent and protects only the selection
  metric. It does not guarantee generalization or improvement.
- W3 component rewards use baseline evaluation configuration; training manifests
  describe the training objective. Contact-only touchdown statistics exclude
  timeouts; explicitly labeled legacy SQL velocity columns remain episode-final.
- No speedup factor, completion time or improved landing rate is asserted.

See [JAX benchmarking guidance](https://docs.jax.dev/en/latest/201/profiling.html)
for synchronization and compile-time separation, and
[JAX compiled stages](https://docs.jax.dev/en/latest/jax.stages.html)
for executable sharding inspection.

## Delivery

Notebook: `outputs/kaggle_ppo_recovery_v2_gpu/rlander_ppo_recovery_v2_gpu.ipynb`.
It embeds source and the selected parent; no separate dataset upload is needed.
Use a fresh Kaggle session with a GPU accelerator and Internet enabled.
Return `europa_ppo_recovery_v2_gpu.zip`, including execution logs, telemetry,
checkpoints, trajectories, metrics and provenance.

The historical v1 and v2 notebook artifacts are preserved.

## Completed preparation checks

All **148 local CPU tests passed in 275.48 seconds**. One existing Orbax
sharding-restore warning occurred in a legacy checkpoint stand-in test.
The seven new tests cover CPU rejection in GPU-required mode, executable/output
device evidence, batched-versus-serial W3 parity for PPO/PD/random/timeout
policies, and the evaluation step limit. Parity checks use floating-point
tolerances and do not establish CUDA numerical parity.

All five notebook cells are unexecuted; code cells pass AST parsing. All 40
embedded source files match both their manifest hashes and the working files.
The parent archive and its individual files match recorded hashes. Scoped
lint and `git diff --check` passed; pre-existing broad YAML exception handlers
in `eval/run_eval.py` remain outside the lint-clean scope.
