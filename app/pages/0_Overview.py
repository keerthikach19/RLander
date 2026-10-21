"""Mission overview — the 'homepage' for the demo."""

import streamlit as st

from app.ui import evidence_notes, flight_figure, headline, summary, trajectory

# ── Hero ──────────────────────────────────────────────────────────────────────
st.caption("FROZEN EVALUATION · EUROPA")
st.title("A trained controller learns to land")
st.write(
    "A **PPO (Proximal Policy Optimisation) neural network** was trained from scratch — "
    "no handcrafted rules, no demonstrations — and evaluated on **2,000 previously unseen "
    "flight seeds** running on Kaggle GPU. It achieved **90.45% safe landings**."
)

# ── Key stats ─────────────────────────────────────────────────────────────────
headline()
st.caption(
    "PPO Wilson 95% interval: 89.08–91.66%. "
    "Gap uses the same 2,000 test flights for both controllers."
)

# ── Live demo flight + narrative ──────────────────────────────────────────────
st.divider()
left, right = st.columns([1.6, 1])

with left:
    seed = summary()["policies"]["ppo"]["replays"][0]["seed"]
    frame = trajectory("ppo", seed)
    st.plotly_chart(
        flight_figure(frame, animate=True, index=0),
        width="stretch",
    )
    st.caption(
        f"🎬 Click **▶ Play** to watch the PPO agent land — seed {seed}, "
        "recorded from the frozen checkpoint evaluation. "
        "Trail colour shows throttle level; flame plume shows active thrust."
    )

with right:
    st.subheader("What makes this real ML?")
    st.markdown("""
**The PPO agent:**
- Takes a **noisy 8-dimensional observation** (position, velocity, mass, time)
- Outputs **continuous throttle + gimbal angle** every ~0.38 s
- Learned entirely via **reward signals** — no equation of motion was given to it
- Trained through **~80,000 additional transitions** of experience

**Evidence you can check:**
- 📉 Training curves show outcome rates improving over time → *Results* page
- 🔁 The same seeds produce **different decisions** from the PD baseline → *Compare*
- 💾 Checkpoint hashes are frozen; the model shown is the one that was tested
- 🖥️ You can watch the policy run **live, step by step** → *Live simulation*
""")
    st.divider()
    st.subheader("Explore")
    st.page_link("pages/1_Simulator.py", label="Replay a flight (crashed, landed, timeout)", icon="🎬")
    st.page_link("pages/2_Compare.py", label="Compare PPO vs PD vs parent on same seeds", icon="📊")
    st.page_link("pages/3_Results.py", label="See training curves & checkpoint history", icon="📋")
    st.page_link("pages/4_Local.py", label="Watch the policy decide in real time", icon="🖥️")

# ── Why PPO doesn't beat PD ───────────────────────────────────────────────────
st.divider()
st.subheader("Why is PPO below the PD baseline?")
st.markdown("""
The PD controller is **purpose-built** for this exact environment — it uses the **true equations of motion** 
as its control law. That's not a real competitor; it's a sanity-check ceiling.

PPO learned to solve the problem **without knowing the equations**. The 7.55 percentage-point gap 
is the cost of generalisation without a hand-crafted model. In real planetary landing, you don't 
always have a perfect dynamics model — which is exactly where learned policies like PPO have value.

> **The honest claim:** PPO achieves >90% success on a stochastic, noisy, full-height landing 
> problem, trained from scratch in a matter of hours on a single Kaggle GPU.
""")

st.divider()
evidence_notes()
