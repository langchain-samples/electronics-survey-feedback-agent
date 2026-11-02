"""Streamlit UI: one chat surface for all three customer-feedback agents.

Pick an agent in the sidebar and chat with it. The same conversation flow,
human approval step, and "agent steps" panel work for every agent, whatever
built or hosts it:

  * Fleet agent         -- no-code, built in LangSmith Fleet, called over its API
  * Google ADK agent    -- built with ADK, runs here, traced to LangSmith
  * Managed Deep Agent  -- survey_agent/ in-process, or any deployed MDA

All three trace to LangSmith, so the conversations below show up in the Agents
view next to each other.

Run it from the workshop root:
    uv run streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from agents.chat_backends import (  # noqa: E402
    AdkBackend,
    ConfigError,
    FleetBackend,
    MdaBackend,
    Turn,
    format_args,
)

st.set_page_config(page_title="Customer Feedback Agents", page_icon="🏷️", layout="wide")

BLUE, YELLOW, NAVY = "#0046BE", "#FFE000", "#001E73"
st.markdown(
    f"""
    <style>
      [data-testid="stSidebar"] {{ background: {NAVY}; }}
      [data-testid="stSidebar"] * {{ color: #FFFFFF; }}
      [data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {{ color: #111 !important; }}
      [data-testid="stSidebar"] [data-testid="stExpander"] {{ border-color: rgba(255,255,255,.25); }}
      .brand-bar {{ background: {BLUE}; color: #fff; padding: 14px 20px; border-radius: 10px;
                    display: flex; align-items: center; gap: 14px; margin-bottom: 8px; }}
      .brand-tag {{ background: {YELLOW}; color: #111; font-weight: 800; padding: 4px 10px;
                    border-radius: 4px; font-size: 1.05rem; letter-spacing: .3px; }}
      .brand-title {{ font-size: 1.35rem; font-weight: 700; }}
      .brand-sub {{ font-size: .9rem; opacity: .85; }}
      .agent-pill {{ display: inline-block; background: {YELLOW}; color: #111; font-weight: 700;
                     padding: 2px 10px; border-radius: 999px; font-size: .8rem; }}
      .approval {{ border: 2px solid {YELLOW}; background: #FFFBE0; border-radius: 10px; padding: 12px 16px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

AGENTS = {
    "fleet": {
        "label": "Fleet agent",
        "icon": "🧩",
        "title": "Survey Insights Agent (Fleet)",
        "blurb": "No-code agent built in LangSmith Fleet · read-only BigQuery over the "
        "thank-you-page survey · `pattern-miner` subagent · called through the Agent Server API.",
        "suggestions": [
            "Where are customers particularly unhappy, and what patterns explain it?",
            "What's the rating distribution across the whole dataset?",
            "What makes 5-star customers happy? Give verbatim examples.",
        ],
        "placeholder": "Ask about ratings, happy/unhappy drivers, or patterns…",
    },
    "adk": {
        "label": "Google ADK agent",
        "icon": "🔷",
        "title": "Survey Feedback Agent (Google ADK)",
        "blurb": "Built with Google ADK · runs in this process · LiteLLM → LangSmith LLM Gateway · "
        "traced with `configure_google_adk()` · posting blocked until you approve.",
        "suggestions": [
            "Summarize the top survey themes for the survey owners.",
            "What is NPS by channel?",
            "Draft the theme summary and post it to #survey-insights.",
        ],
        "placeholder": "Ask about themes, NPS, or channels…",
    },
    "mda": {
        "label": "Managed Deep Agent",
        "icon": "🧠",
        "title": "Survey Feedback Agent (Managed Deep Agent)",
        "blurb": "Deep Agents harness defined with `define_deep_agent` · theme-clustering subagent · "
        "`interrupt_on` pauses before posting · in-process here, or any deployed MDA.",
        "suggestions": [
            "Summarize the top survey themes for the survey owners.",
            "What is NPS by product category?",
            "Analyze the latest survey export and post the summary to #survey-insights.",
        ],
        "placeholder": "Ask about themes, NPS, or channels…",
    },
}

ss = st.session_state
ss.setdefault("histories", {key: [] for key in AGENTS})
ss.setdefault("backends", {})
ss.setdefault("pending", {})
ss.setdefault("queued", None)

# --- sidebar ------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"<div style='display:flex;gap:10px;align-items:center'>"
        f"<span class='brand-tag' style='background:{YELLOW};color:#111'>VOICE OF CUSTOMER</span></div>",
        unsafe_allow_html=True,
    )
    st.title("Feedback Agents")
    st.caption("One UI, three ways to build an agent.")
    choice = st.radio(
        "Agent",
        list(AGENTS),
        format_func=lambda k: f"{AGENTS[k]['icon']}  {AGENTS[k]['label']}",
    )

    with st.expander("Connection settings", expanded=False):
        fleet_url = st.text_input("Fleet API URL", os.environ.get("FLEET_API_URL", ""),
                                  help="Fleet agent → Advanced settings → View code snippets")
        fleet_agent_id = st.text_input("Fleet agent ID", os.environ.get("FLEET_AGENT_ID", ""))
        st.divider()
        mda_url = st.text_input("MDA deployment URL", os.environ.get("MDA_DEPLOYMENT_URL", ""),
                                help="Leave empty to run survey_agent/ in-process.")
        mda_assistant = st.text_input("MDA assistant ID",
                                      os.environ.get("MDA_ASSISTANT_ID", "survey-feedback-agent"),
                                      help="The define_deep_agent name, e.g. survey-feedback-agent")

    settings = {"fleet": (fleet_url, fleet_agent_id), "mda": (mda_url, mda_assistant), "adk": ()}
    if ss.get("settings") != settings:  # settings changed: reconnect on next message
        ss.backends.clear()
        ss.settings = settings

    st.divider()
    if st.button("🧹 New conversation", use_container_width=True):
        ss.histories[choice] = []
        ss.backends.pop(choice, None)
        ss.pending.pop(choice, None)
        st.rerun()
    st.caption("Traces for every agent land in LangSmith → **Agents**.")


def get_backend(key: str):
    if key not in ss.backends:
        if key == "fleet":
            ss.backends[key] = FleetBackend(url=fleet_url.strip(), agent_id=fleet_agent_id.strip())
        elif key == "adk":
            ss.backends[key] = AdkBackend()
        else:
            ss.backends[key] = MdaBackend(url=mda_url.strip(), assistant_id=mda_assistant.strip())
    return ss.backends[key]


def mda_mode() -> str:
    return f"deployed · `{mda_assistant}`" if mda_url.strip() else "in-process · `survey_agent/`"


# --- header -------------------------------------------------------------------

cfg = AGENTS[choice]
st.markdown(
    f"""
    <div class="brand-bar">
      <span class="brand-tag">{cfg['icon']} FEEDBACK</span>
      <div><div class="brand-title">{cfg['title']}</div>
      <div class="brand-sub">Customer survey insights for the store, online, and pickup teams</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
caption = cfg["blurb"] + (f"  \n**Mode:** {mda_mode()}" if choice == "mda" else "")
st.caption(caption)

history = ss.histories[choice]


def render_turn(turn_record: dict) -> None:
    st.markdown(turn_record["text"] or "_(no text response)_")
    if turn_record.get("steps"):
        with st.expander(f"🔧 Agent steps ({len(turn_record['steps'])} tool calls)"):
            for step in turn_record["steps"]:
                st.markdown(f"**`{step['name']}`** — `{format_args(step['args'])}`")


def record(turn: Turn) -> None:
    history.append({"role": "assistant", "text": turn.text,
                    "steps": turn.steps})
    if turn.approval:
        ss.pending[choice] = turn.approval
    else:
        ss.pending.pop(choice, None)


def call_agent(action, spinner: str) -> None:
    with st.chat_message("assistant", avatar=cfg["icon"]):
        with st.spinner(spinner):
            try:
                turn = action(get_backend(choice))
            except ConfigError as exc:
                turn = Turn(text=f"⚙️ **Setup needed:** {exc}")
            except Exception as exc:  # surface gateway/auth errors in the chat
                turn = Turn(text=f"⚠️ **Error:** `{type(exc).__name__}: {exc}`")
        record(turn)
        render_turn(history[-1])


if not history:
    st.markdown("#### Try asking")
    cols = st.columns(len(cfg["suggestions"]))
    for col, suggestion in zip(cols, cfg["suggestions"]):
        if col.button(suggestion, use_container_width=True):
            ss.queued = suggestion
            st.rerun()

for msg in history:
    with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else cfg["icon"]):
        if msg["role"] == "user":
            st.markdown(msg["text"])
        else:
            render_turn(msg)

# --- human approval -----------------------------------------------------------

pending = ss.pending.get(choice)
if pending:
    st.markdown(
        f"<div class='approval'><span class='agent-pill'>Waiting for approval</span> "
        f"The agent wants to call <b><code>{pending.tool}</code></b>.</div>",
        unsafe_allow_html=True,
    )
    if "channel" in pending.args:
        st.markdown(f"**Channel:** `{pending.args['channel']}`")
    if "message" in pending.args:
        st.markdown("**Message to post:**")
        st.info(pending.args["message"])
    else:
        st.json(pending.args)
    feedback = st.text_input("Feedback if rejecting (optional)", key=f"feedback-{choice}")
    approve_col, reject_col, _ = st.columns([1, 1, 4])
    if approve_col.button("✅ Approve", type="primary", use_container_width=True):
        history.append({"role": "user", "text": "✅ Approved"})
        call_agent(lambda b: b.resume(approve=True), "Posting…")
        st.rerun()
    if reject_col.button("❌ Reject", use_container_width=True):
        history.append({"role": "user", "text": f"❌ Rejected{': ' + feedback if feedback else ''}"})
        call_agent(lambda b: b.resume(approve=False, feedback=feedback), "Revising…")
        st.rerun()

# --- chat input -----------------------------------------------------------------

prompt = st.chat_input(cfg["placeholder"], disabled=bool(pending)) or ss.queued
if prompt:
    ss.queued = None
    history.append({"role": "user", "text": prompt})
    with st.chat_message("user", avatar="🙋"):
        st.markdown(prompt)
    call_agent(lambda b: b.send(prompt), f"Thinking with the {cfg['label']}…")
    if ss.pending.get(choice):
        st.rerun()
