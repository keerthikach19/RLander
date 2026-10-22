"""Generate an actual controller rollout or opted-in approximation, then animate it."""

from __future__ import annotations
import math
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ui import style  # noqa: E402

VY_SAFE = 1.0
VX_SAFE = 0.5

# ── Backend detection ─────────────────────────────────────────────────────────
def _jax_available() -> bool:
    try:
        import jax  # noqa: F401
        return True
    except ImportError:
        return False

HAS_JAX = _jax_available()

# ── Page header ───────────────────────────────────────────────────────────────
st.caption("EUROPA")
st.title("Live simulation")
st.caption("Generate a flight with the selected controller and watch its trajectory.")
with st.expander("Simulation options"):
    allow_approx = st.checkbox(
        "Allow physics approximation if actual inference is unavailable",
        value=False,
        help="Optional hand-coded fallback. It does not run the trained PPO network.",
    )
if not HAS_JAX:
    st.warning("The model runtime is unavailable. Install the project dependencies to run the trained controller.")

# ── Anti-flicker style injection ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Prevent aggressive flashing/blinking on rapid rerun */
    [data-testid="stAppViewContainer"] * {
        transition: none !important;
    }
    .stPlotlyChart {
        transition: none !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        transition: none !important;
    }
    /* Stop Streamlit from dimming content opacity during rerun */
    .stApp [data-testid="stMain"] [data-testid="stVerticalBlock"] {
        opacity: 1 !important;
        transition: none !important;
    }
    /* Hide the top-right running/spinning indicator that flickers */
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Controls ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1.2, 0.8, 0.8, 1.2])
with c1:
    controller = st.selectbox(
        "Controller",
        ["ppo", "pd"],
        format_func=lambda k: {
            "ppo": "🧠 PPO — frozen V7 neural network",
            "pd": "🔧 PD — hand-coded baseline",
        }[k],
    )
with c2:
    seed = st.number_input("Seed", min_value=0, max_value=99999, value=42, step=1)
with c3:
    speed = st.select_slider(
        "Playback speed",
        options=[1, 2, 3, 5, 8, 12],
        value=3,
        format_func=lambda v: f"{v}×",
        help="Frames to advance per screen refresh.",
    )
with c4:
    st.write("")
    st.write("")
    run_btn = st.button("🚀  Run Simulation", type="primary", use_container_width=True,
                        disabled=not HAS_JAX and not allow_approx)

# ── Session-state keys ────────────────────────────────────────────────────────
SIM_KEY  = "live_traj"
SIM_META = "live_meta"
SIM_SRC  = "live_src"
FIDX_KEY = "live_fidx"
PLAY_KEY = "live_playing"

# ── Run full episode on button click ─────────────────────────────────────────
if run_btn:
    for k in (SIM_KEY, SIM_META, SIM_SRC, FIDX_KEY, PLAY_KEY):
        st.session_state.pop(k, None)

    success = False
    if HAS_JAX and controller in ("ppo", "pd"):
        with st.spinner("Preparing flight…"):
            try:
                from app.live import simulate
                frame, meta = simulate(controller, int(seed))
                st.session_state[SIM_KEY]  = frame
                st.session_state[SIM_META] = meta
                st.session_state[SIM_SRC]  = "real"
                success = True
            except Exception as exc:
                st.error(f"Actual {controller.upper()} inference failed: {exc}")
                if not allow_approx:
                    st.info("Restore the frozen checkpoint for PPO or fix the reported error, then retry.")
                    st.stop()

    if not success:
        if not allow_approx:
            st.error("Actual inference is unavailable. No episode was generated.")
            st.stop()
        with st.spinner("Running physics simulation…"):
            from app.sim_canvas import init_state, step as sim_step
            s = init_state(int(seed))
            while not s.done:
                s = sim_step(s)
            frame = pd.DataFrame({
                "t":        s.ts + [s.t],
                "x":        s.xs + [s.x],
                "y":        s.ys + [s.y],
                "vx":       s.vxs + [s.vx],
                "vy":       s.vys + [s.vy],
                "throttle": s.throttles + [s.throttle],
                "angle":    s.angles + [s.angle],
                "mass":     s.masses + [s.mass],
            })
            meta = {"controller": "physics-approx", "requested_controller": controller,
                    "seed": int(seed), "outcome": s.outcome, "backend": "cpu",
                    "device": "python", "dry_mass_kg": 101.13}
            st.session_state[SIM_KEY]  = frame
            st.session_state[SIM_META] = meta
            st.session_state[SIM_SRC]  = "approx"

    st.session_state[FIDX_KEY] = 0
    st.session_state[PLAY_KEY] = True
    st.session_state.pop("frame_scrubber", None)  # clear stale slider state
    st.rerun()

# ── Wait for data ─────────────────────────────────────────────────────────────
if SIM_KEY not in st.session_state:
    st.info("👆 Choose a controller and seed, then click **Run Simulation**.")
    st.stop()

frame: pd.DataFrame = st.session_state[SIM_KEY]
meta  = st.session_state[SIM_META]
src   = st.session_state[SIM_SRC]
outcome = meta["outcome"]
N = len(frame)
final = frame.iloc[-1]

if src == "approx":
    st.warning("Physics approximation · hand-coded fallback, not the trained PPO model.")
else:
    st.caption(f"{meta['controller'].upper()} controller · seed {meta['seed']}")

# ── Playback controls ─────────────────────────────────────────────────────────
fidx: int = st.session_state.get(FIDX_KEY, 0)
playing: bool = st.session_state.get(PLAY_KEY, False)

pb1, pb2, pb3 = st.columns([1, 1, 4])
with pb1:
    if st.button("▶  Play" if not playing else "⏸  Pause", use_container_width=True):
        st.session_state[PLAY_KEY] = not playing
        if not playing:  # was paused, now starting — clear stale slider
            st.session_state.pop("frame_scrubber", None)
        st.rerun()
with pb2:
    if st.button("⏮  Reset", use_container_width=True):
        st.session_state[FIDX_KEY] = 0
        st.session_state[PLAY_KEY] = False
        st.session_state.pop("frame_scrubber", None)
        st.rerun()
with pb3:
    st.progress(fidx / max(N - 1, 1), text=f"Flight time · {frame.t.iloc[fidx]:.1f} s")

# ── Current-frame data ────────────────────────────────────────────────────────
row = frame.iloc[fidx]
rx   = float(row.x)
ry   = float(row.y)
thr  = float(row.throttle) if "throttle" in frame.columns else 0.0
ang  = float(row.angle)    if "angle"    in frame.columns else 0.0

# ── Live HUD ──────────────────────────────────────────────────────────────────
is_done = (fidx == N - 1)

h1, h2, h3, h4, h5, h6 = st.columns(6)
h1.metric("t", f"{row.t:.1f} s")
h2.metric("Altitude", f"{max(ry, 0.0):.0f} m")
h3.metric(
    "vy (vertical)",
    f"{row.vy:+.2f} m/s",
    delta="Safe · < 1.0 m/s" if abs(row.vy) < VY_SAFE else "Not safe · ≥ 1.0 m/s",
    delta_color="normal" if abs(row.vy) < VY_SAFE else "inverse",
)
h4.metric(
    "vx (horiz)",
    f"{row.vx:+.2f} m/s",
    delta="Safe · < 0.5 m/s" if abs(row.vx) < VX_SAFE else "Not safe · ≥ 0.5 m/s",
    delta_color="normal" if abs(row.vx) < VX_SAFE else "inverse",
)
h5.metric("Throttle", f"{thr:.0%}")
h6.metric("Fuel left", f"{max(float(row.mass) - meta['dry_mass_kg'], 0):.0f} kg")

# ── Build the current-frame chart ─────────────────────────────────────────────
def _frame_chart(frame: pd.DataFrame, fidx: int, outcome: str) -> go.Figure:
    xs_all = frame.x.values
    ys_all = frame.y.values
    x_trail = xs_all[: fidx + 1]
    y_trail = [max(float(y), 0.0) for y in ys_all[: fidx + 1]]

    x_min = float(xs_all.min())
    x_max = float(xs_all.max())
    y_max = float(ys_all.max())
    x_pad = max((x_max - x_min) * 0.15, 50)

    # Colour the trail by throttle level (teal = coasting, orange = firing)
    thr_vals = frame.throttle.values if "throttle" in frame.columns else np.zeros(len(frame))

    fig = go.Figure()

    # ── Trail — segmented by throttle colour ─────────────────────────────
    if len(x_trail) > 1:
        for i in range(len(x_trail) - 1):
            t_norm = float(thr_vals[i])
            # Coasting = teal (#46dec7), firing = orange (#ff9500), blend
            r = int(70  + (255 - 70)  * t_norm)
            g = int(222 - (222 - 149) * t_norm)
            b = int(199 - (199 - 0)   * t_norm)
            seg_color = f"rgb({r},{g},{b})"
            fig.add_trace(go.Scatter(
                x=[x_trail[i], x_trail[i+1]],
                y=[y_trail[i], y_trail[i+1]],
                mode="lines",
                line={"color": seg_color, "width": 3},
                showlegend=False, hoverinfo="skip",
            ))

    # ── Ground / surface ──────────────────────────────────────────────────
    fig.add_hrect(y0=-30, y1=0,
                  fillcolor="rgba(110,90,50,0.30)", line_width=0)
    fig.add_hline(y=0, line_color="#ffc76c", line_width=2.5,
                  annotation_text="Europa surface  ",
                  annotation_font_color="#ffc76c",
                  annotation_position="bottom right")

    # ── Rocket marker & camera sizing ────────────────────────────────────
    row = frame.iloc[fidx]
    rx = float(row.x)
    ry_val = float(row.y)
    thr = float(row.throttle) if "throttle" in frame.columns else 0.0
    ang = float(row.angle)    if "angle"    in frame.columns else 0.0
    is_terminal = (fidx >= len(frame) - 1)

    ry_clamped = max(ry_val, 0.0)
    view_height = max(ry_clamped * 1.25, 80)   # zoom in as altitude drops
    x_view = max(x_pad, abs(rx) + 80)

    # Rocket visual sizing: in a 520 px plot with vertical span (view_height + 30)
    meters_per_pixel = (view_height + 30.0) / 520.0
    lander_half_h = 14.0 * meters_per_pixel

    # Keep rocket base exactly on or above y = 0
    marker_y = max(ry_clamped, lander_half_h)
    base_y = marker_y - lander_half_h

    # Gimbal angle: real angle from model/physics
    gimbal_deg = math.degrees(ang)
    marker_angle = -gimbal_deg

    fig.add_trace(go.Scatter(
        x=[rx], y=[marker_y],
        mode="markers",
        marker={
            "symbol":  "triangle-up",
            "size":    36,
            "color":   "#ffffff",
            "angle":   marker_angle,
            "line":    {"color": "#46dec7", "width": 2.5},
        },
        name="Lander",
        hovertemplate=(
            f"<b>Lander</b><br>"
            f"x = {rx:.1f} m<br>h = {ry_clamped:.1f} m<br>"
            f"throttle = {thr:.0%}<br>gimbal = {math.degrees(ang):.1f}°<extra></extra>"
        ),
    ))

    # Direction arrow (tilt indicator)
    if abs(ang) > 0.01 and not is_terminal:
        ax = rx + 30 * math.sin(ang)
        ay = marker_y + 30 * math.cos(ang)
        fig.add_annotation(
            x=ax, y=ay, ax=rx, ay=marker_y,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=2, arrowcolor="#46dec7", opacity=0.7,
        )

    # ── Flame plume (never shoots beneath the surface) ───────────────────
    if thr > 0.05 and not is_terminal and ry_val > 0.5:
        fa = ang + math.pi            # nozzle faces opposite of tilt
        fl = thr * 60
        fx_tip = rx + fl * 0.35 * math.sin(fa)
        fy_tip = max(base_y - fl * math.cos(fa) * 0.5, 0.0)
        for width, color, opacity in [
            (5,  "#ffffff", 0.95),
            (13, "#ff9500", 0.50),
            (22, "#ff3000", 0.20),
        ]:
            fig.add_trace(go.Scatter(
                x=[rx, fx_tip], y=[base_y, fy_tip],
                mode="lines",
                line={"color": color, "width": width},
                opacity=opacity, showlegend=False, hoverinfo="skip",
            ))

    # ── Stars (static) ───────────────────────────────────────────────────
    rng = np.random.RandomState(7)
    n_stars = 60
    sx = rng.uniform(x_min - x_pad, x_max + x_pad, n_stars)
    sy = rng.uniform(max(ry_clamped * 0.6, y_max * 0.4), y_max * 1.05, n_stars)
    sz = rng.uniform(1, 3.5, n_stars)
    fig.add_trace(go.Scatter(
        x=sx, y=sy, mode="markers",
        marker={"color": "white", "size": sz.tolist(), "opacity": 0.28},
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_xaxes(
        range=[-x_view, x_view],
        title_text="Horizontal position (m)",
        gridcolor="rgba(255,255,255,0.04)", zeroline=False,
    )
    fig.update_yaxes(
        range=[-30, view_height],
        title_text="Altitude (m)",
        gridcolor="rgba(255,255,255,0.04)", zeroline=False,
    )
    style(fig, height=520)
    fig.update_layout(showlegend=False)
    return fig


chart_slot = st.empty()
chart_slot.plotly_chart(_frame_chart(frame, fidx, outcome),
                        width="stretch", key="live_main_chart")

# -- Live altitude phase explanation / outcome banner ------------------------
if is_done:
    if outcome == "landed":
        vok = abs(final.vy) < VY_SAFE
        vxok = abs(final.vx) < VX_SAFE
        vy_mark = "< 1.0 ✓" if vok else "≥ 1.0 ✗"
        vx_mark = "< 0.5 ✓" if vxok else "≥ 0.5 ✗"
        st.success(
            f"✅ **Soft landing!** — "
            f"touchdown |vy| = **{abs(final.vy):.3f} m/s** ({vy_mark}), "
            f"|vx| = **{abs(final.vx):.3f} m/s** ({vx_mark}). "
            f"Duration {final.t:.1f} s · "
            f"fuel burned {float(frame.mass.iloc[0]-final.mass):.1f} kg."
        )
    elif outcome == "crashed":
        vy_mark = "✓" if abs(final.vy) < VY_SAFE else "✗ EXCEEDED"
        vx_mark = "✓" if abs(final.vx) < VX_SAFE else "✗ EXCEEDED"
        st.error(
            f"💥 **Crash** — "
            f"contact |vy| = **{abs(final.vy):.3f} m/s** (limit 1.0 {vy_mark}), "
            f"|vx| = **{abs(final.vx):.3f} m/s** (limit 0.5 {vx_mark})."
        )
    else:
        st.warning(
            f"⏱️ **Timeout** — episode budget exhausted. "
            f"Final altitude {final.y:.1f} m."
        )
# -- Live telemetry bar -------------------------------------------------------
tele_slot = st.empty()
# Throttle telemetry chart re-renders during playback to reduce DOM churn and flicker
should_update_tele = (not playing) or is_done or (fidx % 4 == 0)
if fidx > 2 and should_update_tele:
    tele = make_subplots(rows=1, cols=3,
                         subplot_titles=("Altitude (m)", "Velocity (m/s)", "Throttle"),
                         horizontal_spacing=0.10)
    sl = slice(0, fidx + 1)
    tele.add_trace(go.Scatter(x=frame.t.values[sl], y=frame.y.values[sl],
                              line={"color": "#46dec7", "width": 2},
                              showlegend=False), row=1, col=1)
    tele.add_trace(go.Scatter(x=frame.t.values[sl], y=frame.vy.values[sl],
                              name="vy", line={"color": "#46dec7", "width": 2}), row=1, col=2)
    tele.add_trace(go.Scatter(x=frame.t.values[sl], y=frame.vx.values[sl],
                              name="vx", line={"color": "#9bafff", "width": 2, "dash": "dot"}),
                   row=1, col=2)
    tele.add_hrect(y0=-1.0, y1=0, row=1, col=2,
                   fillcolor="#46dec7", opacity=0.07, line_width=0)
    tele.add_trace(go.Scatter(x=frame.t.values[sl], y=frame.throttle.values[sl],
                              line={"color": "#ffc76c", "width": 2},
                              fill="tozeroy", fillcolor="rgba(255,199,108,0.12)",
                              showlegend=False), row=1, col=3)
    tele.update_xaxes(title_text="t (s)", gridcolor="rgba(255,255,255,0.04)")
    tele.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
    style(tele, height=220)
    tele_slot.plotly_chart(tele, width="stretch", key="live_tele_chart")

# -- Advance to next frame (the actual animation loop) ------------------------
if playing:
    next_fidx = fidx + speed
    if next_fidx >= N - 1:
        # Reached the end: park on final frame, clear stale scrubber key
        st.session_state[FIDX_KEY] = N - 1
        st.session_state[PLAY_KEY] = False
        st.session_state.pop("frame_scrubber", None)
        time.sleep(0.05)
        st.rerun()  # one last render: shows outcome banner + stopped Play button
    else:
        st.session_state[FIDX_KEY] = next_fidx
        time.sleep(0.05)
        st.rerun()

# ── Post-flight summary (visible when done) ───────────────────────────────────
if is_done:
    st.divider()
    with st.expander("Flight details"):
        st.caption("A complete episode is generated before animation. Playback speed changes only the presentation.")
        firing = int((frame.throttle > 0.05).sum())
        coast  = N - firing
        fuel   = float(frame.mass.iloc[0] - frame.mass.iloc[-1])
        st.markdown(rf"""
**{meta['controller'].upper()} · seed {meta['seed']} · {outcome.upper()}**

| Metric | Value |
|--------|-------|
| Total steps | {N} |
| Coasting (throttle < 5%) | {coast} steps ({100*coast/N:.0f}%) |
| Engine firing | {firing} steps ({100*firing/N:.0f}%) |
| Peak throttle | {float(frame.throttle.max()):.0%} |
| Fuel burned | {fuel:.1f} kg |
| Final \|vy\| | {abs(final.vy):.3f} m/s (limit 1.0) |
| Final \|vx\| | {abs(final.vx):.3f} m/s (limit 0.5) |

The telemetry above reports the actions and state changes from this flight.
It does not establish that the trajectory is fuel-optimal.
""")

    with st.expander("Export flight"):
        st.download_button(
            f"⬇️ Download trajectory CSV — seed {meta['seed']}",
            frame.to_csv(index=False),
            file_name=f"live_{meta['controller']}_seed{meta['seed']}.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download execution evidence JSON",
            json.dumps(meta, indent=2),
            file_name=f"live_{meta['controller']}_seed{meta['seed']}_evidence.json",
            mime="application/json",
        )
