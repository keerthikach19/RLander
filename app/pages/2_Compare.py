import plotly.graph_objects as go
import streamlit as st

from app.ui import (
    COLORS, OUTCOMES, comparison_table, flight_figure, headline, intro,
    style, summary, table, trajectory,
)

intro("MATCHED TEST CONDITIONS", "Compare controllers",
      "All three controllers faced the same 2,000 evaluation seeds, full starts and sensor noise.")
headline()
st.dataframe(comparison_table(), hide_index=True, use_container_width=True)
policies = summary()["policies"]
fig = go.Figure()
for outcome, color in OUTCOMES.items():
    fig.add_bar(name=outcome.capitalize(), x=[p["label"] for p in policies.values()],
                y=[p["by_outcome"][outcome]["n_episodes"] for p in policies.values()],
                marker_color=color)
fig.update_layout(barmode="stack", yaxis_title="Number of test flights")
st.plotly_chart(style(fig, 320), use_container_width=True)
st.caption("V7: 1,809 landings, 184 crashes, 7 timeouts. Its parent: 1,810 landings. The one-flight difference is not evidence of improvement.")
st.subheader("Touchdown speeds")
episodes = table("episodes.csv")
fig = go.Figure()
for key, policy in policies.items():
    contact = episodes[(episodes.policy == key) & (episodes.outcome != "timeout")]
    fig.add_trace(go.Scattergl(
        x=contact.final_vx_mps.abs(), y=contact.final_vy_mps.abs(), mode="markers",
        name=policy["label"], marker={"color": COLORS[key], "size": 5, "opacity": .45},
        customdata=contact.seed, hovertemplate="Seed %{customdata}<br>|vx| %{x:.3f}<br>|vy| %{y:.3f}<extra>%{fullData.name}</extra>",
    ))
fig.add_vline(x=.5, line_dash="dash", line_color="#ffc76c")
fig.add_hline(y=1, line_dash="dash", line_color="#ffc76c")
fig.update_layout(xaxis_title="Horizontal speed at contact |vx| (m/s)",
                  yaxis_title="Vertical speed at contact |vy| (m/s)")
st.plotly_chart(style(fig), use_container_width=True)
st.caption("Safe landings lie below both dashed limits. Timeouts are excluded because they never made contact. Click legend entries to isolate a controller.")
st.subheader("Same starting seed, different decisions")
seed = st.selectbox("Matched replay seed", [e["seed"] for e in policies["ppo"]["replays"]])
other = st.selectbox("Compare PPO against", ["pd", "parent"], format_func=lambda k: policies[k]["label"])
frames = [trajectory(key, seed) for key in ("ppo", other)]
latest = float(max(frame.t.iloc[-1] for frame in frames))
time = st.slider("Shared simulation time (s)", 0., latest, 0., step=.5)
for column, key, frame in zip(st.columns(2), ("ppo", other), frames):
    with column:
        ep = next(e for e in policies[key]["replays"] if e["seed"] == seed)
        index = max(0, int(frame.t.searchsorted(time, side="right")) - 1)
        st.write(f"**{policies[key]['label']}** · {ep['outcome']}")
        st.plotly_chart(flight_figure(frame, COLORS[key], animate=False, index=index),
                        use_container_width=True, key=f"matched_{key}")
        st.caption(f"Displayed t = {frame.t.iloc[index]:.1f} s. Position holds after termination. Axes are independently scaled.")
st.download_button("Download all episode metrics", episodes.to_csv(index=False),
                   file_name="europa_6000_test_flights.csv", mime="text/csv")
