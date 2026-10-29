# survey_agent

The workshop's survey-feedback agent as a [Managed Deep Agents](https://docs.langchain.com/langsmith/python/managed-deep-agents-overview) project. It reads a customer survey export, measures NPS, clusters comments into themes backed by verbatim quotes, drafts a summary, and posts it to Slack/Teams after a human approves.

## Project structure

```text
survey_agent/
  agent.py                         define_deep_agent(...): model, tools, subagent, interrupt_on
  instructions.md                  system prompt (synced to Context Hub on deploy)
  skills/survey-summary/SKILL.md   summary format (synced to Context Hub on deploy)
  subagents.py                     theme-clustering-agent + approval rules
  tools/survey.py                  survey tools (shared with the Google ADK agent)
  data/survey_export.csv           synthetic survey export, bundled into the build
  identity.py                      who may call the deployment (LangSmith API key)
  sandbox/__init__.py              managed sandbox declaration from `mda init` (delete sandbox/ to opt out)
  pyproject.toml
  .env                             deploy auth + runtime secrets (gitignored)
```

## Commands

```bash
uv sync
uv run mda build                                    # compile into .mda/build without deploying
uv run mda dev                                      # local Agent Server + Studio, traces to the Local environment
uv run mda deploy --name survey-feedback-agent --agent-environment development
uv run mda deploy --name survey-feedback-agent --agent-environment staging
uv run mda logs                                     # tail deployment logs
```

The model is `langsmith:openai/gpt-5.6-luna` (LangSmith LLM Gateway), so `.env` needs only `LANGSMITH_API_KEY` (with `gateway:invoke`), plus `LANGSMITH_WORKSPACE_ID` for org-scoped keys. `SURVEY_WEBHOOK_URL` is optional: without it, approved posts are recorded to `survey_outbox.jsonl` in the system temp directory (override with `SURVEY_OUTBOX`).

To run the same definition in-process (notebooks, evals), use `agents/local_agent.py` at the workshop root.
