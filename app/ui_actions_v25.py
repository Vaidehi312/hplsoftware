"""
Apply a v25 query plan to Streamlit session state (viewer, HPC explorer, highlights).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

from llm_layer_v25 import first_hpc_id


@dataclass
class ViewerContext:
    load_tile_coords_for_slide: Callable
    load_adjacency_for_slide: Callable
    slides_for_hpc_from_kb: Callable
    detect_entity_patterns: Callable
    detect_adjacency_intent: Callable
    detect_single_hpc_adjacency_intent: Callable
    parse_hpc_id: Callable
    parse_inflammation: Callable
    parse_necrosis: Callable
    parse_highlight_mode: Callable


def apply_plan_to_session(
    plan: dict[str, Any],
    prompt: str,
    slide_id: str,
    ctx: ViewerContext,
) -> str:
    """Update session state from plan; return resolved active slide_id."""
    ents = plan.get("entities") or {}
    ui = plan.get("ui_actions") or {}

    slide_id = str(slide_id or "").strip().upper()
    if ents.get("slide"):
        slide_id = str(ents["slide"][0]).strip().upper()
        st.session_state.active_slide = slide_id
    else:
        det = ctx.detect_entity_patterns(prompt)
        if det and det.get("slide"):
            slide_id = str(det["slide"]).strip().upper()
            st.session_state.active_slide = slide_id

    if ui.get("open_slide_viewer"):
        st.session_state.viewer_open = True
    elif plan.get("intent") not in ("greeting", "help", "general_query"):
        if ents.get("slide") or ents.get("hpc") or ents.get("tile"):
            st.session_state.viewer_open = True

    hid = first_hpc_id(plan) or ctx.parse_hpc_id(prompt)
    explore = bool(hid) and (
        ui.get("explore_hpc_slides")
        or plan.get("intent") == "hpc_query"
        or any(
            w in (prompt or "").lower()
            for w in ("which slides", "compare", "across", "most tiles", "rank", "wsi")
        )
    )
    if explore and hid is not None:
        st.session_state.hpc_wsi_explore_id = int(hid)
        for _k in ("hpc_dd_compare_a", "hpc_dd_compare_b"):
            if _k in st.session_state:
                del st.session_state[_k]
        st.session_state.hpc_wsi_ranking_df = ctx.slides_for_hpc_from_kb(int(hid))

    if not prompt:
        return slide_id

    _apply_wsi_highlight_state(plan, prompt, slide_id, ctx)
    return slide_id


def _apply_wsi_highlight_state(
    plan: dict[str, Any],
    prompt: str,
    slide_id: str,
    ctx: ViewerContext,
) -> None:
    slide_id = str(slide_id).strip().upper()
    ui = plan.get("ui_actions") or {}

    st.session_state.query_inflammation = None
    st.session_state.query_necrosis = None
    st.session_state.query_malignant = None
    st.session_state.adj_hpc_a = None
    st.session_state.adj_hpc_b = None
    st.session_state.adj_tile_sets = None

    df = ctx.load_tile_coords_for_slide(slide_id)
    if df is None or df.empty:
        return

    pair = ctx.detect_adjacency_intent(prompt)
    single = ctx.detect_single_hpc_adjacency_intent(prompt)

    if pair or single is not None:
        needed = {"slide_tile", "x_native", "y_native", "hpc_id"}
        if needed - set(df.columns):
            return

    if pair:
        a, b = pair
        st.session_state.highlight_mode = "Adjacency"
        st.session_state.adj_hpc_a = a
        st.session_state.adj_hpc_b = b
        pair_edge_counts, tile_has_neighbor_pair = ctx.load_adjacency_for_slide(slide_id)
        p = (a, b) if a < b else (b, a)
        st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(
            p, {"a_touch": set(), "b_touch": set()}
        )
        return

    if single is not None:
        st.session_state.highlight_mode = "Adjacency"
        pair_edge_counts, tile_has_neighbor_pair = ctx.load_adjacency_for_slide(slide_id)
        candidates = [
            ((a, b), cnt) for (a, b), cnt in pair_edge_counts.items() if a == single or b == single
        ]
        if candidates:
            (a_sel, b_sel), _ = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
            st.session_state.adj_hpc_a = a_sel
            st.session_state.adj_hpc_b = b_sel
            p = (a_sel, b_sel) if a_sel < b_sel else (b_sel, a_sel)
            st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(
                p, {"a_touch": set(), "b_touch": set()}
            )
        else:
            st.session_state.adj_tile_sets = {"a_touch": set(), "b_touch": set()}
        return

    hpc_id = ui.get("selected_hpc") or first_hpc_id(plan) or ctx.parse_hpc_id(prompt)
    infl = ctx.parse_inflammation(prompt)
    nec = ctx.parse_necrosis(prompt)
    mode = ui.get("highlight_mode") or ctx.parse_highlight_mode(prompt)
    st.session_state.query_inflammation = infl
    st.session_state.query_necrosis = nec

    flags = plan.get("flags") or {}
    if mode == "Malignant" or flags.get("malignant") or flags.get("non_malignant"):
        if flags.get("non_malignant") and not flags.get("malignant"):
            st.session_state.query_malignant = "non-malignant"
        else:
            st.session_state.query_malignant = "malignant"

    q0 = (prompt or "").lower()
    if ("heatmap" in q0 or mode == "Heatmap") and hpc_id is not None:
        st.session_state.highlight_mode = "Heatmap"
        st.session_state.heat_hpc = int(hpc_id)
        st.session_state.setdefault("heat_min", 0.0)
        st.session_state.setdefault("heat_alpha", 0.6)
        st.session_state.selected_hpc = None
        return

    if hpc_id is not None:
        st.session_state.selected_hpc = int(hpc_id)
        st.session_state.highlight_mode = mode or "HPC clusters"
    elif mode:
        st.session_state.highlight_mode = mode
