# V7 result and timeout diagnosis

The returned ZIP completed successfully and its source manifest matches the
delivered V7 notebook. Selection before test evaluation chose **seed15 at
+80,000 PPO transitions**, with917/1000 full-start validation landings.
It did not choose the final640k checkpoint.

## Full-start test: seeds90000–91999

| Policy | Landed | Crashed | Timeout | Success | Wilson95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent PPO | 1810 | 186 | 4 | 90.50% | 89.14–91.71% |
| Selected V7 PPO | 1809 | 184 | 7 | **90.45%** | **89.08–91.66%** |
| PD baseline | 1960 | 40 | 0 | 98.00% | 97.29–98.53% |

All rows use GPU, full-height starts, the original physical thresholds, sensor
noise, and randomized conditions. V7 gained41 landings and lost42 versus its
parent. The success difference is effectively flat. Do not describe90.45% on
this new set as an improvement over89.95% on the previous, different test set.

The selected policy's gap to PD is **7.55 percentage points**. It meets the
user's original observed-score requirement (at least90%, within7–8 points of
PD). The assistant's stricter operational91% target was not met; the original
`target_check.json` remains unchanged and false. Confidence intervals describe
sampling uncertainty and do not establish a true population success floor
of90%.

## Where timeouts increased

- Late mixed-start training windows were roughly8–9% timeouts.
- Every seed's final640k snapshot had31/1000 full-start validation timeouts.
  The parent had2/1000.
- The selected seed15/+80k snapshot had3/1000 validation timeouts and7/2000
  test timeouts (0.35%), versus4/2000 for its parent.

Training outcomes contain short curriculum episodes as well as full starts;
their timeout fractions are not directly comparable to the full-start test.
The sustained increase is nevertheless also present in late full-start
validation, so it is not merely a plotting artifact.

All seven selected-policy timeout trajectories end at5.05–15.91m altitude.
They have positive final vertical velocity (climbing), about0.24–0.47m/s,
and roughly1m/s horizontal speed. Their minimum heights were1.40–11.32m.
Over the last30 seconds they were slowly gaining altitude, rather than simply
descending safely and missing the deadline.

This behavior is consistent with over-braking and delaying contact. The added
V7 cost applies even to safe touchdown speeds, making zero vertical speed
preferable to a small safe descent under that term. The timeouts already
existed in the parent; one unpaired training intervention does not prove
the new reward is their sole cause.

## Smallest useful correction

For the delivered policy, use the validation-selected **80k checkpoint** and
do not substitute a later checkpoint. It already meets the original observed
success/gap target with a0.35% timeout rate. The package preserves the exact
selection made before test evaluation; it does not select the slightly higher
scoring parent by inspecting the test.

If another training revision is requested, the focused hypothesis is a
safe-speed deadband in the **additional** vertical contact term:

    excess = max(abs(vy) - 0.5, 0)
    extra_contact_reward = -30 * excess**2 / (excess**2 + 0.5**2)

This gives the added term no preference between0 and0.5m/s, while still
penalizing faster impacts. It leaves the original success definition and
existing lateral costs intact. It would also change the penalty above0.5m/s,
so effects on crashes and timeouts require fresh validation; it is not a
verified fix. No reward code or training notebook has been changed for this
hypothesis, and no further experiment has been launched.

Timeout elimination alone would raise1809/2000 to at most1816/2000 (90.8%)
if all seven became successful landings. The remaining184 crashes dominate:
154 vertical-only,29 lateral-only, and one failing both limits.

## Preserved artifacts and verification

- `outputs/europa_v7_result/europa_v7_selected_policy.zip`: selected checkpoint,
  original manifest, source, provenance, file hashes, evaluation summary and
  unchanged raw target/paired reports.
- `outputs/europa_v7_result/summary.json`: results, explicit distinction between
  original and stricter goals, per-timeout trajectory statistics, and failure counts.

The frozen selection and checkpoint manifest match; all exact test seed sets
and outcome counts were verified. All2,000 selected trajectories were checked
against physical contact/velocity criteria. The curriculum-bank hash and all
12 packaged checkpoint-file hashes were verified. The older replay app remains
explicitly labelled as the V4 snapshot; no unsupported relabelling occurred.
