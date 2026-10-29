"""Survey-feedback agent built with Google ADK, traced to LangSmith.

This is the "agent not deployed through LangSmith": it runs wherever you host
it (locally, Cloud Run, Vertex AI Agent Engine) and sends traces to LangSmith
via LangSmith's ADK integration, `configure_google_adk()`. It uses the same tools as the
Managed Deep Agent (survey_agent/tools/survey.py), so the same evaluators apply
to both.

Model access: the agent calls models through the LangSmith LLM Gateway (ADK's
LiteLLM wrapper pointed at the Gateway's OpenAI-compatible endpoint), so it
uses the same `openai/gpt-5.6-luna` as the Managed Deep Agent and needs no
Google key. Auth is a LangSmith API key with `gateway:invoke`, read from
LANGSMITH_GATEWAY_API_KEY, LANGSMITH_API_KEY_GATEWAY, or LANGSMITH_API_KEY.
Set ADK_MODEL to any Gateway model ID (e.g. `anthropic/claude-sonnet-5`), or to a
`gemini-...` name to use Gemini natively (then GOOGLE_API_KEY or Vertex AI).

Trace destination, in .env (register the agent in LangSmith first):
  ADK_AGENT_ID=survey-feedback-adk  ADK_AGENT_ENVIRONMENT=development

Run it:
  uv run python -m agents.adk_agent "Summarize the top survey themes."
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from contextlib import nullcontext

import langsmith as ls
from dotenv import load_dotenv

from pathlib import Path

from survey_agent.tools.survey import (
    get_survey_stats,
    post_themes_to_channel,
    search_responses,
    theme_counts,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from google.adk.agents import Agent  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402
from langsmith.integrations.google_adk import configure_google_adk  # noqa: E402

ADK_MODEL = os.environ.get("ADK_MODEL", "openai/gpt-5.6-luna")
GATEWAY_BASE_URL = "https://gateway.smith.langchain.com/v1"  # US region; use your regional gateway elsewhere
APP_NAME = "survey_feedback_adk"


def build_model(model_id: str = ADK_MODEL):
    """Gemini names run natively; anything else is a Gateway model ID via LiteLLM."""
    if model_id.startswith("gemini"):
        return model_id
    api_key = (
        os.environ.get("LANGSMITH_GATEWAY_API_KEY")
        or os.environ.get("LANGSMITH_API_KEY_GATEWAY")
        or os.environ.get("LANGSMITH_API_KEY")
    )
    # "openai/" selects LiteLLM's OpenAI-compatible client; the Gateway receives
    # the rest ("openai/gpt-5.6-luna") as the model ID and routes on it.
    return LiteLlm(model=f"openai/{model_id}", api_base=GATEWAY_BASE_URL, api_key=api_key)


MODEL = build_model()


def _require_approval(tool, args, tool_context):
    """before_tool_callback: block outbound posts unless approval was granted.

    ADK has no built-in Agent Inbox; this guard keeps the same rule as the Deep
    Agent (nothing is posted without a human yes) by requiring the caller to
    set `approved_to_post` in session state.
    """
    if tool.name == "post_themes_to_channel" and not tool_context.state.get("approved_to_post"):
        return {
            "status": "blocked",
            "reason": "Posting requires human approval. Show the draft to the user and ask them to approve.",
        }
    return None


theme_clusterer = Agent(
    name="theme_clustering_agent",
    model=MODEL,
    description="Clusters survey responses into recurring themes backed by verbatim quotes.",
    instruction=(
        "Cluster customer survey responses into 3-6 themes. Use search_responses "
        "with several keyword sets, then theme_counts on each theme's response IDs. "
        "For each theme return: name, one-line description, response IDs, channel "
        "mix, average NPS score, and 2-3 quotes copied verbatim from the comment "
        "field with their response_id. Never paraphrase inside quotes. When done, "
        "transfer back to survey_feedback_agent."
    ),
    tools=[search_responses, theme_counts],
)

root_agent = Agent(
    name="survey_feedback_agent",
    model=MODEL,
    description="Turns customer survey exports into an evidence-backed theme summary.",
    instruction=(
        "You are a survey-feedback analyst for an electronics retailer.\n"
        "1. Call get_survey_stats for every number you report. Never estimate.\n"
        "2. Transfer to theme_clustering_agent to get themes with verbatim quotes.\n"
        "3. Write a summary under 300 words: headline with response count and NPS, "
        "3-5 themes ordered by impact on NPS (most negative first), each with two "
        "verbatim quotes cited by response_id, then suggested next steps.\n"
        "4. Redact emails and phone numbers as [redacted].\n"
        "5. Only call post_themes_to_channel when the user asks to share the summary."
    ),
    tools=[get_survey_stats, search_responses, post_themes_to_channel],
    sub_agents=[theme_clusterer],
    before_tool_callback=_require_approval,
)


_tracing_configured = False
_trace_address = None


def enable_langsmith_tracing(agent_id: str | None = None, environment: str | None = None) -> None:
    """Instrument ADK once per process and pick the LangSmith agent it traces to.

    `configure_google_adk()` wraps ADK's runner, agents, model calls and tools
    in LangSmith runs. Runs are addressed to `agent_id` (or ADK_AGENT_ID) in the
    given environment through `ls.tracing_context(address=...)` inside
    `run_adk_agent`, so the ADK agent traces to its own LangSmith agent even in a
    process (like a notebook) where other code traces to a different agent.
    Env vars aren't used for this: the SDK caches them on first read.
    """
    global _tracing_configured, _trace_address
    agent_id = agent_id or os.environ.get("ADK_AGENT_ID")
    if agent_id:
        _trace_address = ls.agent_address(
            agent_id, environment or os.environ.get("ADK_AGENT_ENVIRONMENT", "development")
        )
    if not _tracing_configured:
        configure_google_adk(
            name="survey-feedback-adk",
            metadata={"framework": "google-adk", "use_case": "survey-feedback"},
            tags=["adk", "survey-feedback"],
        )
        _tracing_configured = True


async def run_adk_agent(prompt: str, *, approved_to_post: bool = False) -> str:
    """Run the ADK agent once and return its final text response."""
    enable_langsmith_tracing()
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id="workshop-user",
        session_id=str(uuid.uuid4()),
        state={"approved_to_post": approved_to_post},
    )
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    final = ""
    with ls.tracing_context(address=_trace_address) if _trace_address else nullcontext():
        async for event in runner.run_async(
            user_id="workshop-user",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final = "".join(p.text or "" for p in event.content.parts)
    return final


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or (
        "Summarize the top themes in the latest customer survey export for the survey owners."
    )
    print(asyncio.run(run_adk_agent(question)))
