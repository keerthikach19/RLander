"""Shared views for the frozen Europa release."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA = Path(__file__).resolve().parent / "release_data"
COLORS = {"ppo": "#46dec7", "parent": "#9bafff", "pd": "#ffc76c"}
OUTCOMES = {"landed": "#46dec7", "crashed": "#ff7b87", "timeout": "#ffc76c"}


@st.cache_data
def summary():
    return json.loads((DATA / "summary.json").read_text(encoding="utf-8"))


@st.cache_data
def table(name):
    return pd.read_csv(DATA / name)


@st.cache_data
def trajectory(policy, seed):
    item = next(e for e in summary()["policies"][policy]["replays"] if e["seed"] == seed)
    return pd.read_parquet(DATA / item["file"])


def intro(eyebrow, title, description):
    st.caption(eyebrow)
    st.title(title)
    st.write(description)


def headline():
    data = summary()
    ppo, pd_policy = data["policies"]["ppo"], data["policies"]["pd"]
    cols = st.columns(4)
    cols[0].metric("PPO landing success", f"{ppo['success_rate']:.2%}")
    cols[1].metric("PD landing success", f"{pd_policy['success_rate']:.2%}")
    cols[2].metric("Gap to PD", f"{100 * (pd_policy['success_rate'] - ppo['success_rate']):.2f} pp")
    cols[3].metric("Test flights per controller", f"{ppo['n_episodes']:,}")


def style(fig, height=420):
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif", "color": "#dce5f2"},
        margin={"l": 25, "r": 20, "t": 45, "b": 45},
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def _make_rocket_svg_path(cx: float, cy: float, heading_deg: float, scale: float = 1.0):
    """Return SVG path string for a detailed rocket shape centred at (cx,cy)."""
    # Build a rocket pointing UP then rotate by heading_deg
    w = 14 * scale
    h = 36 * scale
    # Nose cone
    nose_x, nose_y = 0, -h / 2
    # Body corners
    bl_x, bl_y = -w / 2, h * 0.15
    br_x, br_y =  w / 2, h * 0.15
    # Engine bell bottom
    eb_x1, eb_y1 = -w * 0.35, h / 2
    eb_x2, eb_y2 =  w * 0.35, h / 2
    # Fins
    fl_x1, fl_y1 = -w / 2, h * 0.0
    fl_x2, fl_y2 = -w * 1.1, h * 0.5
    fl_x3, fl_y3 = -w / 2, h * 0.45
    fr_x1, fr_y1 =  w / 2, h * 0.0
    fr_x2, fr_y2 =  w * 1.1, h * 0.5
    fr_x3, fr_y3 =  w / 2, h * 0.45

    def rot(px, py):
        rad = math.radians(heading_deg)
        rx = px * math.cos(rad) - py * math.sin(rad)
        ry = px * math.sin(rad) + py * math.cos(rad)
        return cx + rx, cy + ry

    pts = [rot(nose_x, nose_y), rot(br_x, br_y), rot(eb_x2, eb_y2),
           rot(eb_x1, eb_y1), rot(bl_x, bl_y)]
    body_path = "M " + " L ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts) + " Z"

    fin_l = [rot(fl_x1, fl_y1), rot(fl_x2, fl_y2), rot(fl_x3, fl_y3)]
    fin_r = [rot(fr_x1, fr_y1), rot(fr_x2, fr_y2), rot(fr_x3, fr_y3)]
    fin_l_path = "M " + " L ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in fin_l) + " Z"
    fin_r_path = "M " + " L ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in fin_r) + " Z"

    return body_path, fin_l_path, fin_r_path


def flight_figure(frame, color="#46dec7", animate=True, index=0):
    """Cinematic flight path figure with rocket body, flame, and Europa surface."""
    x_min = float(frame.x.min())
    x_max = float(frame.x.max())
    y_min = float(frame.y.min())
    y_max = float(frame.y.max())
    x_pad = max((x_max - x_min) * 0.15, 30)
    y_pad = y_max * 0.08

    # --- Flight path (glowing trail) ---
    fig = go.Figure()

    # Thick glowing trail
    fig.add_trace(go.Scatter(
        x=frame.x, y=frame.y, name="Flight path", mode="lines",
        line={"color": color, "width": 3},
        customdata=np.column_stack([frame.t, frame.throttle, frame.vy]),
        hovertemplate=(
            "x %{x:.1f} m<br>h %{y:.1f} m<br>"
            "t %{customdata[0]:.1f} s<br>"
            "throttle %{customdata[1]:.2f}<br>"
            "vy %{customdata[2]:.2f} m/s<extra></extra>"
        ),
    ))

    # Glow copy (wider, more transparent)
    fig.add_trace(go.Scatter(
        x=frame.x, y=frame.y, name="_glow", mode="lines",
        line={"color": color, "width": 10},
        opacity=0.12,
        hoverinfo="skip", showlegend=False,
    ))

    # Landing zone indicator
    fig.add_hrect(y0=-3, y1=3,
                  fillcolor="#ffc76c", opacity=0.08,
                  line_width=0, annotation_text="")

    # Surface line with Europa colour
    fig.add_hline(y=0, line_color="#ffc76c", line_width=2,
                  annotation_text="Europa surface",
                  annotation_font_color="#ffc76c",
                  annotation_position="bottom right")

    # --- Current rocket position ---
    idx = index
    rx = float(frame.x.iloc[idx])
    ry = float(frame.y.iloc[idx])
    throttle_now = float(frame.throttle.iloc[idx]) if "throttle" in frame.columns else 0.0
    angle_now = float(frame.angle.iloc[idx]) if "angle" in frame.columns else 0.0

    # Rocket body as SVG shapes
    heading_deg = math.degrees(angle_now)  # 0 = pointing up
    body_path, fin_l, fin_r = _make_rocket_svg_path(0, 0, heading_deg, scale=1.0)

    # We use a scatter with a custom symbol for simplicity; detailed SVG goes in layout shapes
    fig.add_trace(go.Scatter(
        x=[rx], y=[ry], name="Lander",
        mode="markers",
        marker={
            "color": "#ffffff",
            "size": 22,
            "symbol": "triangle-up",
            "line": {"color": color, "width": 2},
        },
        hovertemplate=f"<b>Lander</b><br>x={rx:.1f} m<br>h={ry:.1f} m<extra></extra>",
    ))

    # Flame plume (only when throttle > 0)
    if throttle_now > 0.05:
        flame_len = throttle_now * 80  # scale flame to throttle
        # Flame direction: opposite of thrust (downward when angle~0)
        fa = angle_now + math.pi  # flame points away from nozzle
        fx = rx + flame_len * 0.35 * math.sin(fa)
        fy = ry - flame_len * math.cos(fa) * 0.5
        # Inner hot core
        fig.add_trace(go.Scatter(
            x=[rx, fx], y=[ry, fy],
            mode="lines", name="_flame_core",
            line={"color": "#ffffff", "width": 6},
            opacity=0.95, showlegend=False, hoverinfo="skip",
        ))
        # Outer plume (orange)
        fig.add_trace(go.Scatter(
            x=[rx, fx], y=[ry, fy - 5],
            mode="lines", name="_flame_outer",
            line={"color": "#ff9500", "width": 14},
            opacity=0.4, showlegend=False, hoverinfo="skip",
        ))
        # Wide dissipating plume (red)
        fig.add_trace(go.Scatter(
            x=[rx, fx], y=[ry, fy - 10],
            mode="lines", name="_flame_wide",
            line={"color": "#ff4040", "width": 22},
            opacity=0.15, showlegend=False, hoverinfo="skip",
        ))

    fig.update_xaxes(
        title_text="Horizontal position (m)",
        range=[x_min - x_pad, x_max + x_pad],
        gridcolor="rgba(255,255,255,0.04)", zeroline=False,
    )
    fig.update_yaxes(
        title_text="Altitude (m)",
        range=[min(-15., y_min - 20), y_max + y_pad],
        gridcolor="rgba(255,255,255,0.04)", zeroline=False,
    )
    style(fig, height=480)

    if animate:
        indices = np.unique(np.linspace(0, len(frame) - 1, min(200, len(frame)), dtype=int))

        frames = []
        for i in indices:
            rx_i = float(frame.x.iloc[i])
            ry_i = float(frame.y.iloc[i])
            thr_i = float(frame.throttle.iloc[i]) if "throttle" in frame.columns else 0.0
            ang_i = float(frame.angle.iloc[i]) if "angle" in frame.columns else 0.0

            frame_data = [
                go.Scatter(x=[rx_i], y=[ry_i]),   # trace 2 = lander dot
            ]
            # Flame traces (3,4,5) — blank when throttle=0
            if thr_i > 0.05:
                fa_i = ang_i + math.pi
                fl_i = thr_i * 80
                fx_i = rx_i + fl_i * 0.35 * math.sin(fa_i)
                fy_i = ry_i - fl_i * math.cos(fa_i) * 0.5
                frame_data += [
                    go.Scatter(x=[rx_i, fx_i], y=[ry_i, fy_i]),
                    go.Scatter(x=[rx_i, fx_i], y=[ry_i, fy_i - 5]),
                    go.Scatter(x=[rx_i, fx_i], y=[ry_i, fy_i - 10]),
                ]
            else:
                frame_data += [
                    go.Scatter(x=[], y=[]),
                    go.Scatter(x=[], y=[]),
                    go.Scatter(x=[], y=[]),
                ]
            frames.append(go.Frame(name=str(i), data=frame_data, traces=[2, 3, 4, 5]))

        fig.frames = frames
        fig.update_layout(
            margin={"b": 90},
            updatemenus=[{
                "type": "buttons", "direction": "left", "x": 0, "y": -0.22,
                "bgcolor": "#1a2540", "bordercolor": "#46dec7", "borderwidth": 1,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": 55, "redraw": True},
                            "transition": {"duration": 0},
                            "fromcurrent": True,
                        }],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [[None], {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": False},
                        }],
                    },
                ],
            }],
        )
    return fig


def telemetry_figure(frame):
    """4-panel telemetry with cleaner layout and coloured safety bands."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Altitude (m)", "Velocity (m/s)", "Throttle", "Propellant consumed (kg)"),
        vertical_spacing=0.18, horizontal_spacing=0.12,
    )

    # Altitude
    fig.add_trace(go.Scatter(x=frame.t, y=frame.y, name="Altitude",
                             line={"color": "#46dec7", "width": 2}), row=1, col=1)

    # Velocities — both on same panel
    fig.add_trace(go.Scatter(x=frame.t, y=frame.vy, name="vy (vertical)",
                             line={"color": "#46dec7", "width": 2}), row=1, col=2)
    fig.add_trace(go.Scatter(x=frame.t, y=frame.vx, name="vx (horizontal)",
                             line={"color": "#9bafff", "width": 2, "dash": "dot"}), row=1, col=2)

    # Safe zones on velocity panel
    fig.add_hrect(y0=-1.0, y1=0, row=1, col=2,
                  fillcolor="#46dec7", opacity=0.07, line_width=0,
                  annotation_text="Safe vy", annotation_font_size=9,
                  annotation_font_color="#46dec7")

    # Throttle
    fig.add_trace(go.Scatter(x=frame.t, y=frame.throttle, name="Throttle",
                             line={"color": "#ffc76c", "width": 2},
                             fill="tozeroy", fillcolor="rgba(255,199,108,0.12)"), row=2, col=1)

    # Fuel consumed
    fuel = frame.mass.iloc[0] - frame.mass
    fig.add_trace(go.Scatter(x=frame.t, y=fuel, name="Fuel burned",
                             line={"color": "#ff7b87", "width": 2},
                             fill="tozeroy", fillcolor="rgba(255,123,135,0.10)"), row=2, col=2)

    fig.update_xaxes(title_text="Time (s)", gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
    return style(fig, 520)


def comparison_table():
    rows = []
    for policy in summary()["policies"].values():
        rows.append({
            "Controller": policy["label"], "Flights": policy["n_episodes"],
            "Landed": policy["n_success"],
            "Crashed": policy["by_outcome"]["crashed"]["n_episodes"],
            "Timeout": policy["by_outcome"]["timeout"]["n_episodes"],
            "Success": f"{policy['success_rate']:.2%}",
            "Wilson 95% interval": f"{policy['ci95_low']:.2%} – {policy['ci95_high']:.2%}",
            "Mean fuel (kg)": round(policy["fuel_mean"], 2),
        })
    return pd.DataFrame(rows)


def evidence_notes():
    st.write(summary()["success_criteria"] + ". Timeouts are failures and have no touchdown.")
    st.write(
        "All reported release scores use full-height starts, sensor noise, randomized conditions, "
        "and matching test seeds 90000–91999 on Kaggle GPU. PD is a separate hand-designed "
        "controller; PPO receives no PD demonstrations or action assistance."
    )
    st.write(
        "The selected V7 model scored 917/1,000 on validation before the fresh test was run. "
        "The parent scored 90.50% on the same test, versus V7's 90.45%; this refinement "
        "does not demonstrate an overall improvement."
    )
    st.caption(
        "Planar point mass with vectored thrust, fuel consumption and a flat surface. "
        "No full attitude dynamics, terrain avoidance, external benchmark parity or flight qualification. "
        "Confidence intervals reflect sampling uncertainty, not a guaranteed success floor."
    )
