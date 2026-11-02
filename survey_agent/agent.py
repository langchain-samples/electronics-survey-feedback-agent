import os

from langchain.chat_models import init_chat_model
from managed_deepagents import define_deep_agent

from subagents import INTERRUPT_ON, THEME_SUBAGENT
from tools.survey import TOOLS_BY_NAME

# LLM Gateway calls made with an org-scoped LangSmith key must name the workspace
# (X-Tenant-ID), and a bare "langsmith:..." model string doesn't send it, so build
# the model here. GATEWAY_WORKSPACE_ID is checked first because mda may not forward
# the LANGSMITH_* workspace variables to the deployment.
_workspace_id = next(
    (os.environ[k] for k in ("GATEWAY_WORKSPACE_ID", "LANGSMITH_WORKSPACE_ID", "LANGSMITH_TENANT_ID")
     if os.environ.get(k)),
    None,
)
model = init_chat_model(
    "langsmith:openai/gpt-5.6-luna",
    **({"default_headers": {"X-Tenant-ID": _workspace_id}} if _workspace_id else {}),
)

agent = define_deep_agent(
    name="survey-feedback-agent",
    model=model,
    tools=[
        TOOLS_BY_NAME["read_survey_export"],
        TOOLS_BY_NAME["get_survey_stats"],
        TOOLS_BY_NAME["search_responses"],
        TOOLS_BY_NAME["post_themes_to_channel"],
    ],
    subagents=[THEME_SUBAGENT],
    interrupt_on=INTERRUPT_ON,
)
