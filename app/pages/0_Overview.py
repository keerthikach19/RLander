"""Europa mission overview."""
import streamlit as st
from app.ui import flight_figure, headline, summary, trajectory

st.caption("EUROPA · AUTONOMOUS LANDING")
st.title("Learning to land")
st.write("A trained PPO controller guides a lander to Europa's surface using throttle and engine steering.")
with st.container(border=True):
    st.subheader("Ask Gemma 4 to investigate a landing condition")
    st.write("Describe a question about drift, descent speed, or starting altitude. Gemma designs "
             "controlled tests, the frozen PPO flies them, and you inspect measured outcomes.")
    st.page_link("pages/5_Gemma.py", label="Open Gemma Mission Test Lab", icon="🧪")
headline()
st.caption("Results from 2,000 matched test flights per controller.")
left, right = st.columns([1.6, 1])
with left:
    seed = summary()["policies"]["ppo"]["replays"][0]["seed"]
    st.plotly_chart(flight_figure(trajectory("ppo", seed), animate=True, index=0), use_container_width=True)
    st.caption("Recorded evaluation flight · press Play to watch.")
with right:
    st.subheader("Explore the mission")
    st.page_link("pages/4_Local.py", label="Run a simulation", icon="🚀")
    st.page_link("pages/1_Simulator.py", label="Replay test flights", icon="🎬")
    st.page_link("pages/2_Compare.py", label="Compare controllers", icon="📊")
    st.page_link("pages/3_Results.py", label="Training and results", icon="📋")
    st.write("PPO learns through rewards. The PD baseline uses a hand-designed control law.")
    st.caption("Safe contact: vertical speed below 1.0 m/s and horizontal speed below 0.5 m/s.")
with st.expander("About the experiment"):
    st.write("The frozen V7 policy was selected on validation before testing. It achieved 90.45% success, versus 98.00% for PD. Its parent scored 90.50% on the same test.")
    st.write("This is a planar simulator with sensor noise and randomized starts. The observed results do not establish performance beyond these tested conditions.")
    st.caption("PPO 95% confidence interval: 89.08–91.66%. See Training & evidence for the full record.")
