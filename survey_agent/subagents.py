"""Subagents and approval rules for the survey-feedback agent."""

from tools.survey import TOOLS_BY_NAME

THEME_SUBAGENT = {
    "name": "theme-clustering-agent",
    "description": (
        "Clusters survey responses into recurring themes. Give it the survey "
        "question or focus area; it returns themes, each with response IDs and "
        "verbatim quotes."
    ),
    "system_prompt": (
        "You cluster customer survey responses into themes.\n"
        "1. Use search_responses with several keyword sets to find candidate themes.\n"
        "2. Group responses into 3-6 themes. A theme needs at least 2 responses.\n"
        "3. Use theme_counts on each theme's response IDs.\n"
        "4. Return each theme as: name, one-line description, response IDs, "
        "channel mix, average NPS score, and 2-3 quotes copied verbatim from the "
        "`comment` field with their response_id. Never paraphrase inside quotes."
    ),
    "tools": [TOOLS_BY_NAME["search_responses"], TOOLS_BY_NAME["theme_counts"]],
}

# Pause before anything leaves the agent. On a deployment, the paused thread is
# what reviewers approve, edit, or reject in Agent Inbox.
INTERRUPT_ON = {
    "post_themes_to_channel": {"allowed_decisions": ["approve", "edit", "reject"]},
}
