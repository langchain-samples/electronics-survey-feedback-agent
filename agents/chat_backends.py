"""One chat interface over the workshop's three survey-feedback agents.

Every backend exposes the same two calls, so the Streamlit UI (app.py) never
needs to know which framework or runtime is behind the agent:

    turn = backend.send("Summarize the top survey themes")
    if turn.approval:                       # the agent wants to post something
        turn = backend.resume(approve=True) # or approve=False, feedback="..."

Backends:
  * FleetBackend  -- a no-code Fleet agent, called over the Agent Server API.
  * AdkBackend    -- the Google ADK agent (agents/adk_agent.py), run in-process.
  * MdaBackend    -- the Managed Deep Agent: survey_agent/ compiled in-process,
                     or any deployed MDA (e.g. one produced by utils/fleet_to_mda.py)
                     when a deployment URL is given.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Approval:
    """A tool call paused for a human decision."""

    tool: str
    args: dict[str, Any]
    description: str = ""


@dataclass
class Turn:
    """The result of one exchange with an agent."""

    text: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    approval: Approval | None = None


class ConfigError(RuntimeError):
    """A backend is missing required settings (shown in the UI, not as a crash)."""


# --- message helpers (work for LangChain message objects and API dicts) -------


def _get(msg: Any, key: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _turn_from_messages(new_messages: list[Any]) -> Turn:
    steps, text = [], ""
    for msg in new_messages:
        if _get(msg, "type") not in ("ai", "AIMessage", "AIMessageChunk"):
            continue
        for call in _get(msg, "tool_calls") or []:
            steps.append({"name": _get(call, "name"), "args": _get(call, "args") or {}})
        msg_text = _text(_get(msg, "content"))
        if msg_text.strip():
            text = msg_text
    return Turn(text=text, steps=steps)


def _approval_from_interrupts(values: list[Any]) -> Approval | None:
    """Read the first HITL action request out of interrupt payloads."""
    for value in values:
        value = _get(value, "value", value)
        for action in (value or {}).get("action_requests", []) if isinstance(value, dict) else []:
            return Approval(
                tool=action.get("name", ""),
                args=action.get("args", {}),
                description=action.get("description", ""),
            )
    return None


def _decision(approve: bool, feedback: str) -> dict[str, Any]:
    if approve:
        return {"decisions": [{"type": "approve"}]}
    return {"decisions": [{"type": "reject", "message": feedback or "Rejected by reviewer."}]}


# --- remote Agent Server (Fleet, deployed MDA) --------------------------------


class _RemoteGraphBackend:
    """Stateful runs on an Agent Server deployment through the LangGraph SDK."""

    def __init__(self, url: str, assistant_id: str, api_key: str, headers: dict[str, str] | None = None):
        if not url or not assistant_id:
            raise ConfigError("Set the deployment URL and agent/assistant ID in the sidebar.")
        if not api_key:
            raise ConfigError("No LangSmith API key found (LANGSMITH_API_KEY).")
        from langgraph_sdk import get_sync_client

        self.assistant_id = assistant_id
        self.client = get_sync_client(url=url, api_key=api_key, headers=headers or {})
        self.thread_id: str | None = None
        self._resolved = False

    def _resolve_assistant(self) -> None:
        """Fail fast (or self-correct) when the assistant ID isn't on this deployment.

        A wrong ID otherwise only surfaces as a bare 404 from /runs/wait, because
        thread creation doesn't involve the assistant.
        """
        if self._resolved:
            return
        try:
            self.client.assistants.get(self.assistant_id)
        except Exception as exc:
            # 404: unknown ID. 422: newer servers only look up UUIDs here, even
            # though runs accept a graph name (the define_deep_agent name).
            if getattr(getattr(exc, "response", None), "status_code", None) not in (404, 422):
                raise
            graphs = sorted({a["graph_id"] for a in self.client.assistants.search(limit=100)})
            if self.assistant_id not in graphs:
                if len(graphs) != 1:
                    raise ConfigError(
                        f"Assistant `{self.assistant_id}` isn't on this deployment. "
                        f"Available: {', '.join(f'`{g}`' for g in graphs) or 'none'}."
                    ) from None
                self.assistant_id = graphs[0]
        self._resolved = True

    def _ensure_thread(self) -> str:
        self._resolve_assistant()
        if self.thread_id is None:
            self.thread_id = self.client.threads.create(metadata={"source": "feedback-agents-ui"})["thread_id"]
        return self.thread_id

    def _message_count(self, thread_id: str) -> int:
        values = self.client.threads.get(thread_id).get("values") or {}
        return len(values.get("messages", [])) if isinstance(values, dict) else 0

    def _run(self, **run_kwargs: Any) -> Turn:
        thread_id = self._ensure_thread()
        before = self._message_count(thread_id)
        self.client.runs.wait(thread_id, self.assistant_id, **run_kwargs)
        thread = self.client.threads.get(thread_id)
        values = thread.get("values") or {}
        turn = _turn_from_messages((values.get("messages") or [])[before:])
        pending = [i for items in (thread.get("interrupts") or {}).values() for i in items]
        turn.approval = _approval_from_interrupts(pending)
        return turn

    def send(self, message: str) -> Turn:
        return self._run(input={"messages": [{"role": "user", "content": message}]})

    def resume(self, approve: bool, feedback: str = "") -> Turn:
        return self._run(command={"resume": _decision(approve, feedback)})


class FleetBackend(_RemoteGraphBackend):
    """A Fleet (no-code) agent. Fleet agents run on Agent Server, so the
    LangGraph SDK calls them like any deployment; Fleet authenticates a
    personal access token with the `X-Auth-Scheme: langsmith-api-key` header."""

    def __init__(self, url: str, agent_id: str, api_key: str | None = None):
        super().__init__(
            url=url,
            assistant_id=agent_id,
            api_key=api_key or os.environ.get("FLEET_API_KEY") or os.environ.get("LANGSMITH_API_KEY", ""),
            headers={"X-Auth-Scheme": "langsmith-api-key"},
        )


# --- Managed Deep Agent ------------------------------------------------------

_LOCAL_MDA = None
_LOCAL_MDA_LOCK = threading.Lock()


def _local_mda_graph():
    """Compile survey_agent/ once per process (shared checkpointer, per-thread state)."""
    global _LOCAL_MDA
    with _LOCAL_MDA_LOCK:
        if _LOCAL_MDA is None:
            from langgraph.checkpoint.memory import MemorySaver

            from agents.local_agent import build_agent

            _LOCAL_MDA = build_agent(checkpointer=MemorySaver())
    return _LOCAL_MDA


class _LocalMdaBackend:
    """survey_agent/'s definition compiled in-process (what `mda dev` runs)."""

    def __init__(self):
        self.graph = _local_mda_graph()
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        agent_id = os.environ.get("MDA_AGENT_ID", "survey-feedback-agent")
        environment = os.environ.get("MDA_AGENT_ENVIRONMENT", "development")
        self._address = None
        try:
            import langsmith as ls

            self._address = ls.agent_address(agent_id, environment)
        except Exception:  # older SDKs: fall back to default tracing
            pass

    def _run(self, payload: Any) -> Turn:
        import langsmith as ls

        before = len((self.graph.get_state(self.config).values or {}).get("messages", []))
        with ls.tracing_context(address=self._address) if self._address else nullcontext():
            self.graph.invoke(payload, self.config)
        state = self.graph.get_state(self.config)
        turn = _turn_from_messages(state.values.get("messages", [])[before:])
        turn.approval = _approval_from_interrupts(list(state.interrupts or []))
        return turn

    def send(self, message: str) -> Turn:
        return self._run({"messages": [{"role": "user", "content": message}]})

    def resume(self, approve: bool, feedback: str = "") -> Turn:
        from langgraph.types import Command

        return self._run(Command(resume=_decision(approve, feedback)))


def MdaBackend(url: str = "", assistant_id: str = "") -> _RemoteGraphBackend | _LocalMdaBackend:
    """A deployed MDA when `url` is set, otherwise survey_agent/ in-process."""
    if url:
        return _RemoteGraphBackend(
            url=url,
            assistant_id=assistant_id or "survey-feedback-agent",
            api_key=os.environ.get("LANGSMITH_API_KEY", ""),
        )
    return _LocalMdaBackend()


# --- Google ADK --------------------------------------------------------------


class _LoopThread:
    """One long-lived event loop: ADK / LiteLLM clients must not hop loops."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


_ADK_LOOP: _LoopThread | None = None


class AdkBackend:
    """The Google ADK agent, in-process, with a persistent ADK session.

    ADK has no interrupt primitive, so its `before_tool_callback` blocks
    `post_themes_to_channel` until approval is set in session state. The UI
    surfaces the blocked call as an Approval; approving re-runs the turn with
    `approved_to_post=True`.
    """

    def __init__(self):
        global _ADK_LOOP
        from agents import adk_agent

        self._adk = adk_agent
        if _ADK_LOOP is None:
            _ADK_LOOP = _LoopThread()
        self.session_id = str(uuid.uuid4())

    def _run(self, prompt: str, approved: bool = False) -> Turn:
        result = _ADK_LOOP.run(
            self._adk.chat_turn(prompt, session_id=self.session_id, approved_to_post=approved)
        )
        turn = Turn(text=result["text"], steps=result["steps"])
        if result["blocked_post"]:
            turn.approval = Approval(
                tool="post_themes_to_channel",
                args=result["blocked_post"],
                description="Blocked by before_tool_callback until a human approves.",
            )
        return turn

    def send(self, message: str) -> Turn:
        return self._run(message)

    def resume(self, approve: bool, feedback: str = "") -> Turn:
        if approve:
            return self._run(
                "Approved: post the summary you drafted to the channel now, unchanged.",
                approved=True,
            )
        return self._run(
            "Rejected: do not post. Reviewer feedback: "
            + (feedback or "none given")
            + ". Revise the draft and show it to me again."
        )


def format_args(args: dict[str, Any], limit: int = 400) -> str:
    """Compact one-line rendering of tool arguments for the steps panel."""
    rendered = json.dumps(args, default=str, ensure_ascii=False)
    return rendered if len(rendered) <= limit else rendered[:limit] + " …"
