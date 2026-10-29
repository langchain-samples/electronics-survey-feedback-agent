from managed_deepagents import define_deep_agent

from subagents import INTERRUPT_ON, THEME_SUBAGENT
from tools.survey import TOOLS_BY_NAME

agent = define_deep_agent(
    name="survey-feedback-agent",
    model="langsmith:openai/gpt-5.6-luna",
    tools=[
        TOOLS_BY_NAME["read_survey_export"],
        TOOLS_BY_NAME["get_survey_stats"],
        TOOLS_BY_NAME["search_responses"],
        TOOLS_BY_NAME["post_themes_to_channel"],
    ],
    subagents=[THEME_SUBAGENT],
    interrupt_on=INTERRUPT_ON,
)
