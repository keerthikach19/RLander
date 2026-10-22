import json

import plotly.graph_objects as go
import streamlit as st

from app.ui import DATA, evidence_notes, intro, style, summary, table

intro("TRAINING COMPLETE", "Training & results",
      "Validation performance and training history for the frozen release.")
st.caption("Selected V7 model · 91.7% validation success · +80,000 training steps after its parent.")
validation = table("validation.csv")
version = st.selectbox("Training session", list(validation.version.drop_duplicates()),
                       index=len(validation.version.unique()) - 1)
subset = validation[validation.version == version]
fig = go.Figure()
for arm, rows in subset.groupby("arm"):
    fig.add_scatter(x=rows.additional_steps, y=rows.success_rate, mode="lines+markers", name=arm)
    chosen = rows[rows.selected_for_run]
    fig.add_scatter(x=chosen.additional_steps, y=chosen.success_rate, mode="markers",
                    marker={"size": 14, "symbol": "diamond-open"}, showlegend=False,
                    name=f"{arm} selected")
fig.update_layout(xaxis_title="Transitions in this session (after the parent, where applicable)",
                  yaxis={"title": "Full-start validation success", "range": [0, 1], "tickformat": ".0%"})
st.plotly_chart(style(fig), width="stretch")
st.caption("Diamonds mark each run's selected checkpoint.")
with st.expander("Validation data"):
    st.dataframe(subset.drop(columns="version"), hide_index=True, width="stretch")
if version == "v7":
    arm = st.selectbox("Training trace", ["seed_14", "seed_15", "seed_16"], index=1)
    training = table(f"{arm}_training.csv")
    fig = go.Figure()
    for name in ("landed", "crashed", "timeouts"):
        rate = training[name].rolling(10, min_periods=1).sum() / training.episodes.rolling(10, min_periods=1).sum()
        fig.add_scatter(x=training.env_steps, y=rate, name=name)
    fig.update_layout(xaxis_title="Additional training transitions",
                      yaxis={"title": "Training outcome fraction", "range": [0, 1], "tickformat": ".0%"})
    st.plotly_chart(style(fig, 330), width="stretch")
    st.caption("Training outcomes · rolling 10-update window.")
with st.expander("Checkpoint archive"):
    registry = json.loads((DATA / "checkpoint_registry.json").read_text())
    st.dataframe([{k: entry[k] for k in (
        "version", "arm", "additional_steps", "validation_successes", "validation_episodes", "checkpoint",
    )} for entry in registry], hide_index=True, width="stretch")
    st.caption("Checkpoint paths are relative to the local release bundle. Historical archives retain the other snapshots. The app never chooses a different model for each test flight.")
with st.expander("Historical test results"):
    st.dataframe(table("history.csv"), hide_index=True, width="stretch")
    st.write("Each session used a different fresh test set. These rows are historical records, not a controlled estimate of improvement between sessions.")
with st.expander("Evaluation method and downloads"):
    st.write("Validation sets differ across sessions. Refinements share trained parents. Training mixes partial-flight and full starts, so its outcome rates are not directly comparable to full-start validation. Late timeout growth remains unresolved.")
    evidence_notes()
    st.write("Training stops here. The observed 90.45% and 7.55-point gap meet the requested point-estimate target. The notebook's stricter 91% check was not met and remains recorded as false.")
    st.download_button("Download release evidence", json.dumps(summary(), indent=2),
                       file_name="europa_release_summary.json", mime="application/json")
    st.download_button("Download checkpoint registry", json.dumps(registry, indent=2),
                       file_name="checkpoint_registry.json", mime="application/json")
