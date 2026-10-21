"""Flight Replay — pre-recorded evaluation flights with cinematic playback."""

import streamlit as st

from app.ui import COLORS, flight_figure, summary, telemetry_figure, trajectory

st.caption("RECORDED EVALUATION FLIGHTS")
st.title("Flight replay")
st.write(
    "These flights were **recorded during the frozen evaluation** on Kaggle GPU — "
    "they are exact replays of what the policy did during the official test. "
    "Unlike the *Live simulation* page, nothing is re-computed here; you are watching saved data."
)

st.info(
    "💡 **How this differs from Live simulation:** "
    "This page plays back *saved trajectory data* from the 2,000-flight test. "
    "The **Live simulation** page actually runs the guidance policy step-by-step in this process.",
    icon="🎬",
)

policies = summary()["policies"]
controls = st.columns([1, 1, 2])
with controls[0]:
    policy_key = st.selectbox(
        "Controller",
        list(policies),
        format_func=lambda key: policies[key]["label"],
        help="PPO = trained neural network. Parent = previous PPO checkpoint. PD = hand-coded baseline.",
    )
episodes = {e["seed"]: e for e in policies[policy_key]["replays"]}
with controls[1]:
    outcome = st.selectbox(
        "Outcome filter",
        ["All", "landed", "crashed", "timeout"],
        help="Filter to only show a specific flight outcome.",
    )
eligible = [seed for seed, ep in episodes.items() if outcome == "All" or ep["outcome"] == outcome]
if not eligible:
    st.info("No recorded example of this outcome in the illustrative replay set for this controller. See Compare for full-test counts.")
    st.stop()
with controls[2]:
    seed = st.selectbox(
        "Episode (seed · outcome)",
        eligible,
        format_func=lambda s: f"Seed {s} · {episodes[s]['outcome']}",
    )

frame = trajectory(policy_key, seed)
final = frame.iloc[-1]
ep_info = episodes[seed]
is_contact = ep_info["outcome"] != "timeout"

# ── Outcome banner ─────────────────────────────────────────────────────────
outcome_emojis = {"landed": "✅", "crashed": "💥", "timeout": "⏱️"}
outcome_msgs = {
    "landed": f"Safe landing — |vy| = {abs(final.vy):.3f} m/s, |vx| = {abs(final.vx):.3f} m/s (both within limits).",
    "crashed": f"Crash — contact speed |vy| = {abs(final.vy):.3f} m/s exceeded the 1.0 m/s limit.",
    "timeout": "Timeout — episode exceeded the time budget without making contact.",
}
oc = ep_info["outcome"]
if oc == "landed":
    st.success(f"{outcome_emojis[oc]} {outcome_msgs[oc]}")
elif oc == "crashed":
    st.error(f"{outcome_emojis[oc]} {outcome_msgs[oc]}")
else:
    st.warning(f"{outcome_emojis[oc]} {outcome_msgs[oc]}")

# ── Flight stats ───────────────────────────────────────────────────────────
cols = st.columns(5)
cols[0].metric("Outcome", ep_info["outcome"].capitalize())
cols[1].metric("Flight duration", f"{final.t:.1f} s")
cols[2].metric("Fuel consumed", f"{frame.mass.iloc[0] - final.mass:.1f} kg")
cols[3].metric("Final altitude", f"{final.y:.2f} m")
cols[4].metric("Steps recorded", f"{len(frame)}")

st.caption(
    f"{'Touchdown' if is_contact else 'Final (no touchdown)'} speeds: "
    f"|vy| = {abs(final.vy):.3f} m/s · |vx| = {abs(final.vx):.3f} m/s. "
    "Safe contact requires |vy| < 1.0 and |vx| < 0.5 m/s."
)

# ── Cinematic flight figure ────────────────────────────────────────────────
st.subheader("Flight path")
st.caption("Click ▶ Play to watch the descent. Trail colour = throttle level. Flame plume shows active thrust.")
st.plotly_chart(flight_figure(frame, COLORS[policy_key]), width="stretch")

# ── Policy explanation below the chart ───────────────────────────────────
with st.expander("🧠 What was the controller doing?", expanded=False):
    n_firing = int((frame.throttle > 0.05).sum()) if "throttle" in frame.columns else "?"
    n_coast = len(frame) - n_firing if isinstance(n_firing, int) else "?"
    max_thr = float(frame.throttle.max()) if "throttle" in frame.columns else 0
    mean_thr_firing = float(frame.throttle[frame.throttle > 0.05].mean()) if "throttle" in frame.columns and (frame.throttle > 0.05).any() else 0
    fuel = frame.mass.iloc[0] - final.mass

    if policy_key == "ppo":
        st.markdown(f"""
**PPO neural network decisions:**
- Fired engine in **{n_firing}/{len(frame)} steps** ({n_firing/len(frame):.0%} of the flight)
- Coasted (throttle ≈ 0) for **{n_coast} steps** — conserving fuel early
- Peak throttle: **{max_thr:.0%}**, mean when firing: **{mean_thr_firing:.0%}**
- Total fuel burned: **{fuel:.1f} kg** out of 700 kg propellant
- The gimbal angle (`angle` column) shows how it corrected horizontal drift

The PPO policy received **no equation of motion** — it learned this deceleration profile 
purely from reward signals over thousands of training episodes.
""")
    elif policy_key == "pd":
        st.markdown(f"""
**PD (hand-coded) baseline decisions:**
- This controller uses the **known equations of motion** directly
- It fires in **{n_firing}/{len(frame)} steps** and burns **{fuel:.1f} kg** fuel
- Lower fuel use vs PPO ({fuel:.1f} vs ~49 kg mean) because PD can compute the optimal burn exactly
- PD is not a learned policy — it is a deterministic control law written by hand
""")
    else:
        st.markdown(f"""
**PPO parent checkpoint:**
- This is the **V6 checkpoint** that V7 was refined from
- {n_firing}/{len(frame)} steps firing, {fuel:.1f} kg fuel burned
- Nearly identical behaviour to V7 — the refinement showed no clear improvement
""")

# ── Telemetry ──────────────────────────────────────────────────────────────
st.subheader("Telemetry")
st.plotly_chart(telemetry_figure(frame), width="stretch")
st.caption(
    "Ten seeds were deliberately chosen to include PPO successes and failures. "
    "The headline scores use all 2,000 flights."
)

# ── Download ───────────────────────────────────────────────────────────────
st.download_button(
    "Download this flight (CSV)",
    frame.to_csv(index=False),
    file_name=f"europa_{policy_key}_{seed}.csv",
    mime="text/csv",
)
