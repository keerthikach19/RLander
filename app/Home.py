"""Replay dashboard; JAX is only imported by opt-in local simulation."""

from pathlib import Path
import sys

# Ensure repository root is in sys.path so `app.*` imports resolve cleanly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

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

st.navigation(pages).run()
