"""Run the Managed Deep Agent project (survey_agent/) locally, in-process.

`survey_agent/agent.py` exports a `define_deep_agent(...)` *definition*. On
`mda dev` / `mda deploy`, the managed runtime compiles it with
`create_deep_agent`, supplying the backend, checkpointer, system prompt
(instructions.md) and skills. This module does the same compile in a notebook
kernel so you can step through the agent, run human-in-the-loop locally, and run
evaluations, all from the same definition that gets deployed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SURVEY_AGENT_DIR = PROJECT_ROOT / "survey_agent"

# The MDA project imports its own modules as top-level names (`tools`,
# `subagents`), exactly as the managed build does.
if str(SURVEY_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(SURVEY_AGENT_DIR))

from agent import agent as definition  # noqa: E402

DEFINITION = definition.config
INSTRUCTIONS_PATH = SURVEY_AGENT_DIR / "instructions.md"
SKILLS_DIR = SURVEY_AGENT_DIR / "skills"


def build_agent(*, interrupts: bool = True, checkpointer=None):
    """Compile survey_agent/'s definition locally.

    Args:
        interrupts: Keep the definition's `interrupt_on` approval gate.
        checkpointer: Required for interrupts (the managed runtime supplies its own).
    """
    cfg = DEFINITION
    return create_deep_agent(
        name=cfg["name"],
        model=cfg["model"],
        tools=cfg.get("tools", []),
        subagents=cfg.get("subagents", []),
        interrupt_on=cfg.get("interrupt_on", {}) if interrupts else {},
        system_prompt=INSTRUCTIONS_PATH.read_text(),
        skills=["/skills/"],
        backend=CompositeBackend(
            default=StateBackend(),
            routes={"/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)},
        ),
        checkpointer=checkpointer,
    )
