"""Flight Replay — pre-recorded evaluation flights with cinematic playback."""

import streamlit as st

from app.ui import COLORS, flight_figure, summary, telemetry_figure, trajectory

st.caption("RECORDED EVALUATION FLIGHTS")
st.title("Flight replay")
st.caption("Recorded flights from the frozen evaluation. Choose a controller and episode.")

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
        format_func=lambda s: f"Seed {s} · {episodes[s]['outcome']}" if s in episodes else str(s),
    )

frame = trajectory(policy_key, seed)
final = frame.iloc[-1]
ep_info = episodes[seed]
is_contact = ep_info["outcome"] != "timeout"

# ── Outcome banner ─────────────────────────────────────────────────────────
outcome_emojis = {"landed": "✅", "crashed": "💥", "timeout": "⏱️"}
outcome_msgs = {
    "landed": f"Safe landing — |vy| = {abs(final.vy):.3f} m/s, |vx| = {abs(final.vx):.3f} m/s (both within limits).",
    "crashed": f"Crash — contact |vy| = {abs(final.vy):.3f} m/s; |vx| = {abs(final.vx):.3f} m/s.",
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
st.plotly_chart(flight_figure(frame, COLORS[policy_key]), use_container_width=True)

# ── Telemetry ──────────────────────────────────────────────────────────────
st.subheader("Telemetry")
st.plotly_chart(telemetry_figure(frame), use_container_width=True)
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
