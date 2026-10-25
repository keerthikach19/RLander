"""Natural-language experiment design backed by actual frozen-policy rollouts."""

import json

import pandas as pd
import streamlit as st

from app import gemma_lab as lab
from app.ui import flight_figure

st.caption("GEMMA 4 · MISSION TEST LAB")
st.title("Ask a question. Fly the experiment.")
st.write("Gemma designs controlled landing tests from your question. The original PPO flies them, "
         "the simulator measures the outcomes, and Gemma interprets the evidence.")

key = lab.api_key()
if not key:
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        key = None

question = st.text_area("What would you like to investigate?", value=lab.EXAMPLE, max_chars=1200,
                        help="Try sideways drift, faster descent, or a lower starting altitude.")
left, right = st.columns(2)
seed = int(left.number_input("First of three matched flight seeds", min_value=0, max_value=2**32-3,
                             value=1_700_000, step=1))
compare = right.checkbox("Also test the PD baseline", value=True)
run = st.button("Design and run mission tests", type="primary", disabled=not key)
demo = st.button("Open recorded Gemma demo", disabled=not (lab.ROOT / "app/gemma_demo.json").exists())
if not key:
    st.info("For new experiments, set GEMINI_API_KEY in your local .env or Streamlit Secrets. "
            "The recorded demo works without an API key or model runtime.")
st.caption("One nominal reference + three Gemma-designed probes, each flown with the same three seeds. "
           "These exploratory tests find examples; they do not estimate a general success rate.")

if demo:
    st.session_state["gemma_mission_report"] = lab.load_demo()

if run:
    st.session_state.pop("gemma_mission_report", None)
    try:
        with st.status("Gemma is designing the experiment…", expanded=True) as status:
            plan, plan_api = lab.request_plan(question, key)
            if plan["status"] == "unsupported":
                status.update(label="This request needs a different simulator feature", state="complete")
                st.info(plan["reason"])
                st.stop()
            st.write(plan["title"])
            st.write("Hypothesis: " + plan["hypothesis"])
            cases = lab.scenarios_for(plan)
            st.dataframe(pd.DataFrame([
                {"Case": c["label"], **c["initial"], "Why test this": c["rationale"]} for c in cases
            ]), hide_index=True, use_container_width=True)
            progress = st.progress(0., text="Preparing the frozen controllers…")
            def update(done, total, label, controller):
                progress.progress(done/total, text=f"Flight {done}/{total} · {label} · {controller.upper()}")
            report = lab.run_plan(plan, seed, compare_pd=compare, progress=update)
            report.update(question=question, plan_api=plan_api)
            st.session_state["gemma_mission_report"] = report
            status.update(label="Gemma is reviewing measured outcomes…")
            try:
                analysis, analysis_api = lab.interpret(report, key)
                report.update(interpretation=analysis, interpretation_api=analysis_api)
            except (ValueError, RuntimeError) as exc:
                report["interpretation_error"] = str(exc)
            status.update(label="Experiment complete", state="complete", expanded=False)
    except (ValueError, RuntimeError, ImportError, FileNotFoundError) as exc:
        st.error(f"Experiment stopped: {exc}")
        st.info("Actual PPO inference is required. The recorded Gemma demo is available below.")

report = st.session_state.get("gemma_mission_report")
if report is None:
    st.stop()

if report["origin"] == "recorded":
    st.info("Recorded demo · a real Gemma API plan and actual simulator flights. "
            "No new API request or flight has been run by opening it.")
st.subheader(report["plan"]["title"])
st.write("**Gemma’s experiment hypothesis:** " + report["plan"]["hypothesis"])
with st.expander("Gemma’s test plan and starting conditions", expanded=True):
    st.dataframe(pd.DataFrame([
        {"Case": c["label"], "Altitude (m)": c["initial"]["y"],
         "vx (m/s)": c["initial"]["vx"], "vy (m/s)": c["initial"]["vy"],
         "Beyond training start ranges": c["outside_training_starts"], "Why": c["rationale"]}
        for c in report["cases"]
    ]), hide_index=True, use_container_width=True)

st.subheader("Measured flight results")
ppo = [f for f in report["flights"] if f["controller"] == "ppo"]
cols = st.columns(3)
cols[0].metric("PPO flights tested", len(ppo))
cols[1].metric("PPO landings", sum(f["outcome"] == "landed" for f in ppo))
cols[2].metric("PPO crashes / timeouts", sum(f["outcome"] != "landed" for f in ppo))
st.dataframe(pd.DataFrame(report["summary"]).drop(columns=["case_id"]).rename(columns={
    "case": "Case", "controller": "Controller", "tested": "Flights", "landed": "Landed",
    "crashed": "Crashed", "timeout": "Timeout", "fuel_used_mean_kg": "Mean fuel used (kg)",
    "outside_training_starts": "Exploratory start",
}), hide_index=True, use_container_width=True)

if report["interpretation"]:
    analysis = report["interpretation"]
    st.subheader("Gemma’s interpretation")
    st.write(analysis["headline"])
    names = {c["id"]: c["label"] for c in report["cases"]}
    for finding in analysis["findings"]:
        st.write(f"**{names[finding['case_id']]} · {finding['controller'].upper()}** — {finding['hypothesis']}")
    st.caption("Interpretations are hypotheses. The counts above come directly from the simulator.")
    st.write("**Next question:** " + analysis["next_question"])
elif report.get("interpretation_error"):
    st.warning("Gemma’s interpretation was unavailable or failed evidence checks. "
               "The measured results above are still valid.")

st.subheader("Inspect a flight")
flight_index = st.selectbox("Flight", range(len(report["flights"])), format_func=lambda i:
    f"{report['flights'][i]['case_id']} · {report['flights'][i]['controller'].upper()} · "
    f"seed {report['flights'][i]['seed']} · {report['flights'][i]['outcome']}")
flight = report["flights"][flight_index]
st.plotly_chart(flight_figure(pd.DataFrame(flight["trajectory"])), use_container_width=True)
st.caption("Trace sampled for display. Outcome and touchdown measurements use the full flight.")
st.download_button("Download experiment evidence", json.dumps(report, indent=2, allow_nan=False),
                   "gemma_mission_experiment.json", "application/json")
with st.expander("Model and execution evidence"):
    st.write({"planner": report["plan_api"]["returned_model"], "recorded_at": report["created_at"],
              "controller": "Frozen V7 PPO; no action correction or retraining",
              "backend": flight["backend"], "checkpoint_manifest_sha256": flight["checkpoint_manifest_sha256"]})
