# Survey-Feedback Agent Workshop

Hands-on workshop: build, deploy, evaluate and monitor a **survey-feedback agent** in your own LangSmith instance, then bring an agent built elsewhere (Google ADK) into the same view.

The agent reads a customer survey export, measures NPS, clusters comments into themes backed by **verbatim quotes**, drafts a summary, and posts it to Slack or Teams only after a human approves.

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A LangSmith Cloud **agent-based workspace** and an API key with `projects:create` permission (personal access token or service key, `lsv2_...`)
- LangSmith LLM Gateway enabled for the workspace, with `openai/gpt-5.6-luna` available. Every agent and the notebook judge call models through it, so no provider key is needed locally

## Setup

```bash
uv sync
cp .env.example .env      # fill in your keys
uv run jupyter notebook
```

| Key | Used for |
|-----|----------|
| `LANGSMITH_API_KEY` | Everything: tracing, agent registration, deployment |
| `LANGSMITH_WORKSPACE_ID` | Org-scoped keys (agent creation, `mda deploy`) |
| `LANGSMITH_API_KEY_GATEWAY` | Model calls through the LLM Gateway (a LangSmith key with `gateway:invoke`; often the same value as `LANGSMITH_API_KEY`) |
| `ADK_MODEL` (optional) | Gateway model for the ADK agent (default `openai/gpt-5.6-luna`) |
| `SURVEY_WEBHOOK_URL` (optional) | Real Slack/Teams posts. Without it, approved posts are recorded to `survey_outbox.jsonl` in your system temp directory (override with `SURVEY_OUTBOX`) |

> Don't set `LANGSMITH_PROJECT`. In an agent-based workspace traces are addressed to an agent and environment; combining that with a project gives traces two destinations, and the SDK drops them.

## Modules (Part 2, hands-on)

| Agenda module | Notebook | What you do |
|---|---|---|
| 1. First trace | `modules/01_deep_agents.ipynb` | Register the agent, trace to its development environment, mask emails/phone numbers before they leave the app |
| 2. Build the agent | `modules/01_deep_agents.ipynb` | Deep Agents: filesystem for survey exports, tools, theme-clustering subagent, approval before posting; package it as a Managed Deep Agent and see what MDA adds |
| 3. Deploy it | `modules/03_deploy_and_govern.ipynb` | `mda deploy` to development, call it through the API, approve in Agent Inbox, promote to staging |
| 4. Evaluate and review | `modules/04_langsmith.ipynb` | Runs → dataset, quote-grounding evaluators offline and online, low scores → annotation queue |
| 5. Monitor and alert | `modules/04_langsmith.ipynb` | Dashboards; alerts on error rate, latency, evaluator score |
| 6. Bring your own agents | `modules/04_langsmith.ipynb` | Trace the Google ADK agent with LangSmith's ADK integration, score it with the same evaluators |

## The agents

**Managed Deep Agent (pro-code, hosted by LangSmith)**: `survey_agent/`

- `agent.py`: `define_deep_agent(name="survey-feedback-agent", ...)`
- Tools in `tools/survey.py`: `read_survey_export`, `get_survey_stats`, `search_responses`, `theme_counts`, `post_themes_to_channel`
- `theme-clustering-agent` subagent and `interrupt_on` for `post_themes_to_channel` (`subagents.py`): the run pauses for approve / edit / reject (Agent Inbox once deployed)
- Workflow in `instructions.md`, output format in `skills/survey-summary/SKILL.md`, both synced to Context Hub on deploy
- `identity.py` (LangSmith API key auth) and `sandbox/` as scaffolded by `mda init`
- `agents/local_agent.py` compiles the same definition in-process for the notebooks and evals

```bash
cd survey_agent && uv sync
uv run mda dev                                                                   # local Studio
uv run mda deploy --name survey-feedback-agent --agent-environment development   # or staging
```

**Google ADK agent (runs outside LangSmith)**: `agents/adk_agent.py`

Same tools, task, and Gateway model (`openai/gpt-5.6-luna` via ADK's LiteLLM wrapper), built with `google-adk`, traced with LangSmith's ADK integration, `configure_google_adk()`. A `before_tool_callback` blocks posting unless approval is set in session state.

```bash
uv run python -m agents.adk_agent "Summarize the top survey themes."
```

## Project structure

```
├── survey_agent/                  Managed Deep Agents project (deployed with `mda deploy`)
│   ├── agent.py, subagents.py, instructions.md, skills/, tools/survey.py
│   └── data/survey_export.csv     synthetic survey export (swap in your sanitized export)
├── agents/
│   ├── local_agent.py             compiles survey_agent/ in-process (notebooks, evals)
│   └── adk_agent.py               Google ADK agent (same tools)
├── utils/
│   ├── models.py                  model switch (OpenAI, Gateway, Anthropic, Gemini/Vertex, Azure, Bedrock)
│   ├── agent_registry.py          register agents, look up environments
│   └── langsmith_rules.py         online evaluators + annotation-queue rules
└── modules/
    ├── 01_deep_agents.ipynb       Modules 1–2
    ├── 03_deploy_and_govern.ipynb Module 3
    ├── 04_langsmith.ipynb         Modules 4–6
    └── legacy/                    previous modular-workshop notebooks
```

## Switching models

All three default to `gpt-5.6-luna` through the LangSmith LLM Gateway:

- **Survey agent**: `model="langsmith:openai/gpt-5.6-luna"` in `survey_agent/agent.py`, used both deployed and locally. Change the ID after `langsmith:` to any Gateway model, e.g. `langsmith:anthropic/claude-sonnet-5`.
- **ADK agent**: `ADK_MODEL` (default `openai/gpt-5.6-luna`), called through ADK's LiteLLM wrapper at the Gateway's OpenAI-compatible endpoint. A `gemini-*` value calls Gemini natively instead.
- **Notebook judge and helpers**: `model` from `utils/models.py`; uncomment one block there to swap providers.

## Common issues

**`mda deploy` fails on auth or region**: `mda` reads `survey_agent/.env` before the shell. It needs `LANGSMITH_API_KEY`, plus `LANGSMITH_WORKSPACE_ID` for org-scoped keys and `LANGSMITH_ENDPOINT` outside GCP US. No model key is needed because the agent uses the Gateway. Module 3 writes this file for you.

**Model calls return 401/403/404**: the Gateway key needs `gateway:invoke`, and `openai/gpt-5.6-luna` must be available in the workspace's Gateway (an OpenAI provider secret, or a Gateway Credits model).

**`mda deploy` stops on a Context Hub conflict**: someone edited `instructions.md` or a skill in Context Hub since the last deploy. Run it in a terminal to answer the prompt, or pass `--context-strategy overwrite|keep-hub`.

**Traces don't appear**: check that `LANGSMITH_PROJECT` is unset and that the agent exists (`ensure_agent` in Module 1). Agent addressing needs `langsmith>=0.14.9`.

**Notebook can't find `utils` / `agents`**: each setup cell adds the workshop root to `sys.path`; run notebooks from `modules/`.
