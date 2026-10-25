"""Gemma designs controlled flight probes; the frozen controllers execute them."""

from datetime import UTC, datetime
import math
import os
from pathlib import Path
import re

from app.gemma_api import DEFAULT_MODEL, gemma_request, response_text, strict_json

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = {"x": 0., "y": 2000., "vx": 0., "vy": -20.}
LIMITS = {"y": (100., 3000.), "vx": (-15., 15.), "vy": (-60., -1.)}
EXAMPLE = "Find how increasing sideways drift affects landing. Keep altitude and descent speed the same."
PLAN_SCHEMA = {
    "status": "ready", "title": "Short experiment title", "hypothesis": "A testable hypothesis",
    "parameter": "vx", "scenarios": [
        {"label": "A short label", "value": 3., "rationale": "Why this value is informative"},
        {"label": "Another label", "value": 6., "rationale": "Why this value is informative"},
        {"label": "Third label", "value": 9., "rationale": "Why this value is informative"},
    ],
}


def api_key():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def text_field(value, name, limit=1500):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= limit:
        raise ValueError(f"Gemma returned invalid {name}")
    return value.strip()


def model_json(raw):
    """Accept a single JSON code fence, without extracting JSON from arbitrary prose."""
    raw = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", raw, flags=re.IGNORECASE)
    return strict_json(fenced.group(1) if fenced else raw)


def validate_plan(raw):
    plan = model_json(raw)
    if not isinstance(plan, dict):
        raise ValueError("Gemma's plan must be a JSON object")
    if plan.get("status") == "unsupported":
        if set(plan) != {"status", "reason"}:
            raise ValueError("Invalid unsupported-request response")
        text_field(plan["reason"], "reason")
        return plan
    if set(plan) != set(PLAN_SCHEMA) or plan.get("status") != "ready":
        raise ValueError("Gemma returned an incomplete test plan")
    for key in ("title", "hypothesis"):
        text_field(plan[key], key)
    parameter = plan["parameter"]
    if not isinstance(parameter, str) or parameter not in LIMITS:
        raise ValueError("Only altitude, horizontal speed, or vertical speed can be varied")
    scenarios = plan["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != 3:
        raise ValueError("Gemma must propose exactly three probes")
    seen = {REFERENCE[parameter]}
    for scenario in scenarios:
        if not isinstance(scenario, dict) or set(scenario) != {"label", "value", "rationale"}:
            raise ValueError("Invalid scenario fields")
        text_field(scenario["label"], "scenario label", 100)
        text_field(scenario["rationale"], "scenario rationale")
        value = scenario["value"]
        low, high = LIMITS[parameter]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not low <= value <= high or value in seen):
            raise ValueError("Probe values must be distinct, finite, and within simulator limits")
        seen.add(value)
    return plan


def request_plan(question, key):
    import json

    text_field(question, "mission question", 1200)
    if not key:
        raise ValueError("Configure GEMINI_API_KEY to create a new Gemma experiment")
    prompt = (
        "You are Gemma, the experiment designer for RLander's Europa Mission Test Lab. "
        "Turn the user's question into a controlled experiment that varies exactly ONE "
        "starting-state parameter. You propose experiments; do not predict outcomes as facts. "
        "The simulator is a planar point mass over flat terrain on Europa, with fixed vehicle, "
        "sensor noise and frozen PPO/PD controllers. It does NOT model wind, terrain obstacles, "
        "full attitude, other planets, changing fuel capacity, or a target landing pad. "
        "If the request requires any of those, return only "
        '{"status":"unsupported","reason":"Explain the unsupported feature and a supported alternative"}. '
        "Otherwise return JSON ONLY with exactly the fields in the example schema. "
        "Use your own three informative numeric probe values, not necessarily the example values. "
        "The program adds a fixed reference and runs every case with the same three seeds. "
        "Hold other initial conditions fixed. Negative vy is descent. Safe contact needs "
        "abs(vy)<1 m/s and abs(vx)<0.5 m/s. Training starts were y=1800..2200, vx=-2..2, "
        "vy=-24..-16; values outside these ranges are exploratory. "
        "Do not change physics, controller actions, reward weights or success thresholds.\n"
        + json.dumps({"reference": REFERENCE, "allowed_parameter_ranges": LIMITS,
                      "example_response_shape": PLAN_SCHEMA, "user_question": question}, indent=2)
    )
    payload = gemma_request(prompt, DEFAULT_MODEL, key)
    raw = response_text(payload)
    return validate_plan(raw), {"model": DEFAULT_MODEL, "returned_model": payload.get("modelVersion"),
                                "prompt": prompt, "raw_response": payload}


def scenarios_for(plan):
    # Revalidate on execution too: no unsafe values can enter via stored/UI plans.
    import json
    plan = validate_plan(json.dumps(plan, allow_nan=False))
    if plan["status"] != "ready":
        raise ValueError("This mission request is unsupported")
    cases = [{"id": "reference", "label": "Reference", "initial": dict(REFERENCE),
              "rationale": "Nominal starting conditions; the same seeds are used for every case."}]
    for index, probe in enumerate(plan["scenarios"], 1):
        initial = {**REFERENCE, plan["parameter"]: float(probe["value"])}
        cases.append({"id": f"probe_{index}", "label": probe["label"], "initial": initial,
                      "rationale": probe["rationale"]})
    for case in cases:
        p = case["initial"]
        case["outside_training_starts"] = not (
            1800 <= p["y"] <= 2200 and abs(p["vx"]) <= 2 and -24 <= p["vy"] <= -16
        )
    return cases


def run_plan(plan, seed, *, compare_pd=True, progress=None):
    from app.live import simulate

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32-3:
        raise ValueError("Choose a seed that leaves room for three consecutive uint32 seeds")
    cases = scenarios_for(plan)
    controllers = ("ppo", "pd") if compare_pd else ("ppo",)
    report = {"version": 1, "origin": "live", "created_at": datetime.now(UTC).isoformat(),
              "plan": plan, "cases": cases, "seeds": list(range(seed, seed+3)),
              "controllers": list(controllers), "flights": [], "interpretation": None}
    total = len(cases)*len(controllers)*3
    for case in cases:
        for controller in controllers:
            for flight_seed in report["seeds"]:
                frame, meta = simulate(controller, flight_seed, initial_conditions=case["initial"])
                if meta["controller"] != controller or meta["outcome"] not in ("landed", "crashed", "timeout"):
                    raise ValueError("Simulator returned unexpected execution evidence")
                final = frame.iloc[-1]
                # Downsample only the plot/export trace; metrics use the full flight.
                stride = max(1, math.ceil(len(frame)/150))
                indices = sorted(set(range(0, len(frame), stride)) | {len(frame)-1})
                contact = meta["outcome"] in ("landed", "crashed")
                report["flights"].append({
                    "case_id": case["id"], "controller": controller, "seed": flight_seed,
                    "outcome": meta["outcome"], "duration_s": float(final.t),
                    "touchdown_abs_vy_mps": abs(float(final.vy)) if contact else None,
                    "touchdown_abs_vx_mps": abs(float(final.vx)) if contact else None,
                    "fuel_used_kg": float(frame.iloc[0].mass-final.mass),
                    "final_altitude_m": float(final.y),
                    "backend": meta["backend"], "device": meta["device"],
                    "checkpoint_manifest_sha256": meta.get("checkpoint_manifest_sha256"),
                    "requested_initial_conditions": case["initial"],
                    "actual_initial_state": meta["initial_state"],
                    "trajectory": frame.iloc[indices].to_dict(orient="records"),
                })
                if progress:
                    progress(len(report["flights"]), total, case["label"], controller)
    report["summary"] = summarize(report)
    return report


def summarize(report):
    rows = []
    for case in report["cases"]:
        for controller in report["controllers"]:
            flights = [f for f in report["flights"] if f["case_id"] == case["id"]
                       and f["controller"] == controller]
            if len(flights) != len(report["seeds"]):
                raise ValueError("Incomplete experiment; cannot summarize missing flights")
            rows.append({
                "case_id": case["id"], "case": case["label"], "controller": controller,
                "tested": len(flights),
                **{outcome: sum(f["outcome"] == outcome for f in flights)
                   for outcome in ("landed", "crashed", "timeout")},
                "fuel_used_mean_kg": sum(f["fuel_used_kg"] for f in flights)/len(flights),
                "outside_training_starts": case["outside_training_starts"],
            })
    return rows


def interpret(report, key):
    import json

    summary = summarize(report)
    prompt = (
        "You are Gemma interpreting an executed RLander experiment. Return JSON ONLY: "
        '{"headline":"short cautious finding","findings":[{"case_id":"probe_1",'
        '"controller":"ppo","observation":"mixed","hypothesis":"possible explanation"}],'
        '"next_question":"one useful follow-up experiment question"}. '
        "Return one to four findings, each referencing an existing case/controller. "
        "observation must be all_landed, mixed, or none_landed and MUST match measured counts. "
        "Hypotheses are not established causes. Do not invent telemetry, claim certification, "
        "claim a success probability from only three seeds, or say the controller was trained "
        "or improved. PPO and PD were frozen. Every case used the same three seeds. "
        "Outside-training-range starts are exploratory. Do not claim comparison if only PPO ran.\n"
        + json.dumps({"question": report.get("question"), "plan": report["plan"],
                      "cases": report["cases"], "measured_results": summary}, indent=2)
    )
    payload = gemma_request(prompt, DEFAULT_MODEL, key)
    analysis = model_json(response_text(payload))
    validate_interpretation(analysis, summary)
    return analysis, {"model": DEFAULT_MODEL, "returned_model": payload.get("modelVersion"),
                      "prompt": prompt, "raw_response": payload}


def validate_interpretation(analysis, summary):
    if not isinstance(analysis, dict) or set(analysis) != {"headline", "findings", "next_question"}:
        raise ValueError("Gemma returned an incomplete interpretation")
    text_field(analysis["headline"], "headline")
    text_field(analysis["next_question"], "next question")
    findings = analysis["findings"]
    if not isinstance(findings, list) or not 1 <= len(findings) <= 4:
        raise ValueError("Gemma must return one to four grounded findings")
    lookup = {(r["case_id"], r["controller"]): r for r in summary}
    for item in findings:
        if not isinstance(item, dict) or set(item) != {"case_id", "controller", "observation", "hypothesis"}:
            raise ValueError("Invalid finding fields")
        if not isinstance(item["case_id"], str) or not isinstance(item["controller"], str):
            raise ValueError("Invalid evidence reference")
        row = lookup.get((item["case_id"], item["controller"]))
        if row is None:
            raise ValueError("Gemma cited a scenario/controller that was not run")
        observed = "all_landed" if row["landed"] == row["tested"] else (
            "none_landed" if row["landed"] == 0 else "mixed")
        if item["observation"] != observed:
            raise ValueError("Gemma's observation contradicts the measured flight outcomes")
        text_field(item["hypothesis"], "hypothesis")


def load_demo():
    import json

    report = json.loads((ROOT / "app/gemma_demo.json").read_text(encoding="utf-8"))
    report["origin"] = "recorded"
    if summarize(report) != report["summary"]:
        raise ValueError("Recorded demo summary disagrees with its flight evidence")
    if report["interpretation"]:
        validate_interpretation(report["interpretation"], report["summary"])
    return report
