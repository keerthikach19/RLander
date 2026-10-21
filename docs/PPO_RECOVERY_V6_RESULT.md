# V6 result — improvement, target missed

The returned `europa_ppo_recovery_v6.zip` completed successfully. The
validation-selected candidate is **seed13 at +160,000 PPO transitions**,
with 895/1000 successes on validation seeds9000–9999. This selection was
frozen before the test and matches the archived selection rule and manifest.

All rows below use **2,000 original full-start episodes**, seeds80000–81999,
on GPU, with full noise/randomization and unchanged landing thresholds.

| Policy | Landed | Crashed | Timeout | Success | Wilson 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent PPO | 1746 | 252 | 2 | 87.30% | 85.77–88.69% |
| Selected V6 PPO | 1799 | 196 | 5 | **89.95%** | **88.55–91.19%** |
| PD baseline | 1979 | 21 | 0 | 98.95% | 98.40–99.31% |

Against its parent on these same seeds, V6 gained 127 landings and lost 74:
53 net gains, or **2.65 percentage points**. This comparison uses the matched
parent result, not its 87.7% score on the earlier V5 test set.

## Acceptance

- The observed score is **one landing short of 90%**. Rounding 89.95% to 90%
  does not satisfy an exact 90% minimum.
- The gap to PD is **9.00 percentage points**, outside the requested 7–8-point
  range.
- The predeclared operational target, ≥91% and ≤8 points below PD, was missed.
  With this fixed PD result and denominator, the candidate would need 21 more
  landings to meet both operational conditions. This is arithmetic, not a
  claim that those extra landings can be achieved.
- The confidence interval describes sampling uncertainty; it is not evidence
  that the target has been achieved.

Remaining failures comprise 159 vertical-only crashes, 34 lateral-only crashes,
three failing both speed limits, and five timeouts. Mean fuel consumption was
49.88kg for V6 versus 36.96kg for PD.

## Verification and preservation

The returned source manifest matches the delivered V6 notebook. The recorded
curriculum-bank SHA-256 matches the returned bank. The bank contains 4,096
states from 1,024 parent-PPO flights (324,517 collection transitions).

All 2,000 selected-policy trajectories were checked: their terminal altitude
and velocity agree with the reported outcomes and original success criteria.
The original validation selection, exact test seed sets, outcome counts, and
paired parent comparison were verified.

The policy is PPO-only, with training starts sampled from its own parent's
flights. No PD demonstrations, imitation loss, or controller assistance was
introduced. Training used mixed starts; validation and test used original full
starts. Refinement seeds share a parent and are not independent from-scratch
replications.

Preserved files:

- `outputs/europa_v6_result/europa_v6_selected_policy.zip`: selected snapshot,
  original manifest, file hashes, training source, parent provenance, target
  check, and paired comparisons.
- `outputs/europa_v6_result/summary.json`: verified metrics and failure breakdown.
- `outputs/europa_v6_result/checkpoint_sha256.json`: hashes of all 12 snapshot files.

The previous V4 replay demo is unchanged and still explicitly reports its own
87.5% dataset. No additional training, Kaggle run, or experiment has been
started. This was the final authorized extra session.
