"""Replay dashboard; JAX is only imported by opt-in local simulation."""

import os
from pathlib import Path
import sys

# Ensure repository root is in sys.path so `app.*` imports resolve cleanly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

st.set_page_config(
    page_title="RLander · Europa",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/0_Overview.py", title="Mission overview", icon="🌑", default=True),
    st.Page("pages/1_Simulator.py", title="Flight replay", icon="🎬"),
    st.Page("pages/2_Compare.py", title="Compare controllers", icon="📊"),
    st.Page("pages/3_Results.py", title="Training & evidence", icon="📋"),
    st.Page("pages/4_Local.py", title="Live simulation", icon="🖥️"),
]

with st.sidebar:
    st.markdown("## 🚀 RLander")
    st.caption("EUROPA · TERMINAL DESCENT")
    st.divider()
    st.markdown("**PPO-only training — complete**")
    st.caption("Release: V7 · seed 15 · +80k transitions")
    st.divider()
    st.caption(
        "90.45% landing success over 2,000 test flights on Kaggle GPU. "
        "PD baseline: 98.0%."
    )
    st.divider()
    st.caption(
        "**Live simulation** runs the guidance policy step-by-step. "
        "No JAX required — uses built-in physics approximation."
    )

st.navigation(pages).run()
