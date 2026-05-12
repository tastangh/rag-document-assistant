from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander("Kaynaklar"):
        for s in sources:
            st.markdown(
                f"- **{s.get('doc_id', '')}** | s{s.get('page', 0)} | `{s.get('chunk_id', '')}`\n"
                f"  - {s.get('text_preview', '')}"
            )


def render_debug(details: Dict[str, Any]) -> None:
    if not details:
        return
    with st.expander("Debug / Details", expanded=False):
        st.json(details)

