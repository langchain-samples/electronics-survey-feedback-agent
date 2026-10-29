"""Helpers for LangSmith agents and their environments.

In an agent-based workspace every trace belongs to one agent and one of its four
environments (local, development, staging, production). These helpers register
an agent once and look up an environment's trace storage, which some APIs
(run queries, run rules) still address by project/session.
"""

from __future__ import annotations

from langsmith import Client, NotFoundError
from langsmith.types.agent import Agent, Environment


def ensure_agent(client: Client, agent_id: str, name: str) -> Agent:
    """Return the agent with this identifier, creating it if it doesn't exist.

    Creating an agent needs a personal access token or service key with the
    `projects:create` permission.
    """
    try:
        return client.agents.retrieve(agent_id)
    except NotFoundError:
        return client.agents.create(id=agent_id, name=name, source_type="EXTERNAL")


def get_environment(client: Client, agent_id: str, environment: str) -> Environment:
    """Return one environment of an agent (case-insensitive name)."""
    agent = client.agents.retrieve(agent_id)
    wanted = environment.upper()
    for env in agent.environments:
        if env.environment == wanted:
            return env
    raise ValueError(
        f"Agent {agent_id!r} has no {environment!r} environment. "
        f"Available: {[e.environment.lower() for e in agent.environments]}"
    )


def environment_project_name(client: Client, agent_id: str, environment: str) -> str:
    """The environment's `tracer_session_name`, for APIs that take a project name."""
    return get_environment(client, agent_id, environment).tracer_session_name


def environment_session_id(client: Client, agent_id: str, environment: str) -> str:
    """The environment's `tracer_session_id`, for APIs that take a project ID."""
    return get_environment(client, agent_id, environment).tracer_session_id
