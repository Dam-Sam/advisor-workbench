"""Entry point: `streamlit run app.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(page_title="Advisor Portfolio Workbench", page_icon="📊", layout="wide")

nav = st.navigation([
    st.Page("views/home.py", title="Overview", icon="🏠", default=True),
    st.Page("views/portfolio.py", title="Portfolio", icon="📈"),
    st.Page("views/reconciliation.py", title="Reconciliation", icon="🔁"),
    st.Page("views/ai_assistant.py", title="AI assistant", icon="🤖"),
])
nav.run()
