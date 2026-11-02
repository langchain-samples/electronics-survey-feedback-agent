"""Convert a LangSmith Fleet agent export into a Managed Deep Agents project.

Fleet's "Export to code" produces a zip (or folder) with:

    AGENTS.md                      system prompt
    config.json                    name, model, workspace metadata
    tools.json                     MCP tools + per-tool approval (interrupt_config)
    skills/<name>/SKILL.md         skills
    subagents/<name>/AGENTS.md     subagent prompt (description in front matter)
    subagents/<name>/tools.json    subagent tools

This script maps each piece onto an MDA project (`mda build` / `mda deploy`):

    Fleet                          Managed Deep Agent
    -----------------------------  ------------------------------------------------
    AGENTS.md                      instructions.md (synced to Context Hub)
    skills/                        skills/ (synced to Context Hub)
    config.json model              define_deep_agent(model=...), via LLM Gateway by default
    subagents/<name>/              subagents.py -> define_deep_agent(subagents=[...])
    tools.json interrupt_config    define_deep_agent(interrupt_on={...})
    tools.json known built-ins     authored tools in tools/<name>.py (see TOOL_ADAPTERS)
    tools.json other MCP tools     tools/mcp.py -> define_mcp(servers={...})
    backend / sandbox snapshot     sandbox/__init__.py (only if Fleet used a sandbox)
    (Fleet auth)                   identity.py (LangSmith API key)

Usage (from the workshop root):

    uv run python -m utils.fleet_to_mda My_Fleet_Agent
    uv run python -m utils.fleet_to_mda My_Fleet_Agent.zip --out my_agent --force
    uv run python -m utils.fleet_to_mda My_Fleet_Agent --no-gateway --name my-agent

Then:  cd <out> && uv sync && uv run mda build
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import textwrap
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

FLEET_BUILTIN_HOSTS = {"tools.langchain.com"}
DEFAULT_PYTHON = "3.14"
SKILL_NAME = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")


# --- export model ------------------------------------------------------------------


@dataclass
class FleetTool:
    name: str
    server_url: str
    server_name: str
    interrupt: bool = False


@dataclass
class FleetSubagent:
    name: str
    description: str
    prompt: str
    tools: list[FleetTool]


@dataclass
class FleetExport:
    root: Path
    name: str
    description: str
    model: str
    prompt: str
    tools: list[FleetTool]
    subagents: list[FleetSubagent]
    skills: list[Path]
    sandbox_snapshot_id: str = ""
    backend_type: str = "default"
    ignored_settings: dict[str, object] = field(default_factory=dict)


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("'\"")
    return meta, text[match.end():]


def _read_tools(path: Path) -> list[FleetTool]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    interrupts = data.get("interrupt_config", {}) or {}
    tools = []
    for entry in data.get("tools", []):
        url, server, name = entry.get("mcp_server_url", ""), entry.get("mcp_server_name", ""), entry["name"]
        flag = interrupts.get(f"{url}::{name}::{server}", False)
        tools.append(FleetTool(name=name, server_url=url, server_name=server, interrupt=bool(flag)))
    return tools


def load_export(source: Path) -> FleetExport:
    root = source
    if source.suffix == ".zip":
        root = Path(tempfile.mkdtemp(prefix="fleet-export-"))
        with zipfile.ZipFile(source) as zf:
            zf.extractall(root)
    if not (root / "AGENTS.md").exists():  # zips sometimes wrap everything in one folder
        nested = [p.parent for p in root.glob("*/AGENTS.md")]
        if len(nested) != 1:
            raise SystemExit(f"{source}: no AGENTS.md found; is this a Fleet export?")
        root = nested[0]

    config = json.loads((root / "config.json").read_text()) if (root / "config.json").exists() else {}
    configurable = (config.get("config") or {}).get("configurable") or {}
    model = ((configurable.get("llm_model_config") or {}).get("modelId")) or "anthropic:claude-sonnet-4-6"
    ignored = {
        k: v for k, v in configurable.items()
        if k not in {"llm_model_config", "backend_type", "sandbox_snapshot_id"} and v not in ("", None)
    }

    subagents = []
    for sub_dir in sorted(p for p in (root / "subagents").glob("*") if p.is_dir()):
        meta, body = _split_front_matter((sub_dir / "AGENTS.md").read_text())
        subagents.append(FleetSubagent(
            name=sub_dir.name,
            description=meta.get("description", f"The {sub_dir.name} subagent."),
            prompt=body.strip() + "\n",
            tools=_read_tools(sub_dir / "tools.json"),
        ))

    return FleetExport(
        root=root,
        name=config.get("name") or root.name,
        description=config.get("description") or "",
        model=model,
        prompt=_split_front_matter((root / "AGENTS.md").read_text())[1].strip() + "\n",
        tools=_read_tools(root / "tools.json"),
        subagents=subagents,
        skills=sorted(p.parent for p in (root / "skills").glob("*/SKILL.md")),
        sandbox_snapshot_id=configurable.get("sandbox_snapshot_id") or "",
        backend_type=configurable.get("backend_type") or "default",
        ignored_settings=ignored,
    )


# --- tool adapters -------------------------------------------------------------------
# Fleet built-in tools run on LangChain's hosted tool server with Fleet-managed
# OAuth. An adapter re-implements a built-in as an authored MDA tool, keeping
# the same tool name so the exported prompts work unchanged. Built-ins without
# an adapter fall back to an MCP declaration in tools/mcp.py.


@dataclass
class AdapterOutput:
    module: str                       # tools/<module>.py
    symbol: str                       # name exported by the module
    source: str
    dependencies: list[str]
    env: dict[str, str]               # .env.example entries (name -> default/comment)
    notes: list[str]


_GCP_REF = re.compile(r"`([a-z][a-z0-9-]{4,28}[a-z0-9])\.([A-Za-z_][A-Za-z0-9_]*)`")
_GCP_PROJECT_DATASET = re.compile(
    r"project\s+`([a-z][a-z0-9-]{4,28}[a-z0-9])`\s*,\s*dataset\s+`([A-Za-z_][A-Za-z0-9_]*)`", re.I
)


def _bigquery_adapter(export: FleetExport) -> AdapterOutput:
    texts = [("AGENTS.md", export.prompt)] + [(f"subagents/{s.name}", s.prompt) for s in export.subagents]
    found: dict[str, list[str]] = defaultdict(list)
    for where, text in texts:
        for project, dataset in _GCP_REF.findall(text) + _GCP_PROJECT_DATASET.findall(text):
            ref = f"{project}.{dataset}"
            if where not in found[ref]:
                found[ref].append(where)
    allowed = ",".join(found)
    notes = []
    if len({ref.split(".")[0] for ref in found}) > 1:
        detail = "; ".join(f"`{ref}` ({', '.join(w)})" for ref, w in found.items())
        notes.append(
            "The prompts reference more than one BigQuery project: "
            f"{detail}. All are allowlisted in `BIGQUERY_ALLOWED_DATASETS`; "
            "fix the prompts and trim the allowlist to the project you actually query."
        )
    notes.append(
        "`bigquery_query_read_only` is now an authored tool (`tools/bigquery.py`). Read-only is "
        "enforced in code with a BigQuery dry run (single `SELECT` only, allowlisted datasets only), "
        "not just by the prompt. It authenticates with a service account "
        "(`GOOGLE_SERVICE_ACCOUNT_JSON`), Application Default Credentials locally, or each "
        "caller's own Google account via a user-owned connection (`BIGQUERY_CONNECTION`)."
    )
    return AdapterOutput(
        module="bigquery",
        symbol="bigquery_query_read_only",
        source=BIGQUERY_TOOL_SOURCE,
        dependencies=["google-cloud-bigquery>=3.25"],
        env={
            "BIGQUERY_ALLOWED_DATASETS": allowed or "<project>.<dataset>",
            "BIGQUERY_PROJECT": "# billing project for query jobs; defaults to the credentials' project",
            "GOOGLE_SERVICE_ACCOUNT_JSON": "# service-account key JSON (one line); omit locally to use gcloud ADC",
            "BIGQUERY_CONNECTION": "# optional user-owned OAuth connection slug (mda connections create ...)",
            "BIGQUERY_MAX_ROWS": "500",
            "BIGQUERY_MAX_BYTES_BILLED": "1000000000",
        },
        notes=notes,
    )


TOOL_ADAPTERS = {
    "bigquery_query_read_only": _bigquery_adapter,
}

BIGQUERY_TOOL_SOURCE = '''\
"""Read-only BigQuery SQL, replacing Fleet's built-in `bigquery_query_read_only`.

Generated by utils/fleet_to_mda.py. Guardrails that were prompt-only in Fleet
are enforced here:

* Every query is dry-run first; anything but a single SELECT is refused.
* Every referenced table must sit in BIGQUERY_ALLOWED_DATASETS
  (comma-separated `project.dataset`).
* Results are capped at BIGQUERY_MAX_ROWS rows and BIGQUERY_MAX_BYTES_BILLED.

Credentials, first match wins:
  1. BIGQUERY_CONNECTION: slug of a user-owned Google OAuth connection, so each
     caller queries as themselves (as in Fleet). Create it with
     `mda connections create <slug> --oauth google --scope https://www.googleapis.com/auth/bigquery.readonly ...`
  2. GOOGLE_SERVICE_ACCOUNT_JSON: a service-account key (deployment secret).
  3. Application Default Credentials (`gcloud auth application-default login`).
"""

from __future__ import annotations

import asyncio
import json
import os

from langchain_core.tools import tool

READ_ONLY_SCOPE = "https://www.googleapis.com/auth/bigquery.readonly"


def _allowed_datasets() -> set[str]:
    raw = os.environ.get("BIGQUERY_ALLOWED_DATASETS", "")
    return {d.strip().lower() for d in raw.split(",") if d.strip()}


async def _credentials():
    slug = os.environ.get("BIGQUERY_CONNECTION")
    if slug:
        from google.oauth2.credentials import Credentials
        from managed_deepagents import connections

        return Credentials(token=await connections.get(slug, {"type": "user"}))
    key = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if key:
        from google.oauth2 import service_account

        return service_account.Credentials.from_service_account_info(
            json.loads(key), scopes=[READ_ONLY_SCOPE, "https://www.googleapis.com/auth/cloud-platform"]
        )
    return None  # Application Default Credentials


def _run(sql: str, credentials) -> dict:
    from google.cloud import bigquery

    client = bigquery.Client(project=os.environ.get("BIGQUERY_PROJECT") or None, credentials=credentials)

    dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    if dry.statement_type != "SELECT":
        return {"error": f"Refused: only a single SELECT statement is allowed (got {dry.statement_type})."}
    allowed = _allowed_datasets()
    if not allowed:
        return {"error": "Refused: BIGQUERY_ALLOWED_DATASETS is not configured."}
    outside = sorted(
        f"{t.project}.{t.dataset_id}.{t.table_id}"
        for t in dry.referenced_tables
        if f"{t.project}.{t.dataset_id}".lower() not in allowed
    )
    if outside:
        return {"error": f"Refused: query references tables outside the allowed datasets: {outside}"}

    max_rows = int(os.environ.get("BIGQUERY_MAX_ROWS", "500"))
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            maximum_bytes_billed=int(os.environ.get("BIGQUERY_MAX_BYTES_BILLED", "1000000000"))
        ),
    )
    rows = job.result(max_results=max_rows + 1)
    records = [dict(row.items()) for row in rows]
    return {
        "columns": [f.name for f in rows.schema],
        "rows": json.loads(json.dumps(records[:max_rows], default=str)),
        "row_count": len(records[:max_rows]),
        "truncated": len(records) > max_rows,
        "bytes_processed": job.total_bytes_processed,
    }


@tool(parse_docstring=True)
async def bigquery_query_read_only(query: str) -> str:
    """Run one read-only BigQuery SQL SELECT and return the rows as JSON.

    Use fully qualified table names (`project.dataset.table`). Only a single
    SELECT statement is accepted; DDL, DML, scripts, and tables outside the
    allowed datasets are refused.

    Args:
        query: A single BigQuery Standard SQL SELECT statement.
    """
    try:
        result = await asyncio.to_thread(_run, query, await _credentials())
    except Exception as exc:  # return errors to the model so it can fix the SQL
        result = {"error": f"{type(exc).__name__}: {exc}"}
    return json.dumps(result, default=str)
'''


GATEWAY_MODEL_BLOCK = '''
# LLM Gateway calls made with an org-scoped LangSmith key must name the workspace
# (X-Tenant-ID), and a bare "langsmith:" model string doesn't send it, so build
# the model here. GATEWAY_WORKSPACE_ID is checked first because mda may treat the
# LANGSMITH_*/LANGCHAIN_* workspace variables as reserved and not forward them
# to the deployment.
_workspace_id = next(
    (os.environ[k] for k in ("GATEWAY_WORKSPACE_ID", "LANGSMITH_WORKSPACE_ID", "LANGSMITH_TENANT_ID")
     if os.environ.get(k)),
    None,
)
model = init_chat_model(
    %s,
    **({"default_headers": {"X-Tenant-ID": _workspace_id}} if _workspace_id else {}),
)
'''


# --- generation -------------------------------------------------------------------


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return slug if re.match(r"[a-z]", slug or "") else f"agent-{slug}"


def gateway_model(model: str) -> str:
    """`openai:gpt-5.5` -> `langsmith:openai/gpt-5.5` (LLM Gateway model ID)."""
    if model.startswith("langsmith:") or ":" not in model:
        return model
    provider, name = model.split(":", 1)
    provider = {"google_genai": "google", "google-genai": "google"}.get(provider, provider)
    return f"langsmith:{provider}/{name}"


def _server_key(tool: FleetTool) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", tool.server_name or "mcp").strip("_").lower()
    return key or "mcp"


def _mcp_url(tool: FleetTool) -> str:
    url = tool.server_url.rstrip("/")
    host = re.sub(r"^https?://", "", url).split("/")[0]
    if host in FLEET_BUILTIN_HOSTS and not url.endswith("/mcp"):
        url += "/mcp"
    return url


def _py(value: object) -> str:
    return json.dumps(value, indent=4, ensure_ascii=False)


def _mda_version() -> str:
    try:
        return metadata.version("managed-deepagents")
    except metadata.PackageNotFoundError:
        return "0.9.5"


def convert(export: FleetExport, out: Path, *, name: str, model: str) -> list[str]:
    """Write the MDA project. Returns the conversion report lines."""
    report: list[str] = []
    out.mkdir(parents=True)
    (out / "tools").mkdir()
    (out / "tools" / "__init__.py").write_text("")

    # Tools: adapters first, MCP fallback for the rest.
    all_tools = export.tools + [t for s in export.subagents for t in s.tools]
    adapters: dict[str, AdapterOutput] = {}
    mcp_servers: dict[str, dict] = {}
    for tool in all_tools:
        if tool.name in adapters or any(tool.name in s.get("include_tools", []) for s in mcp_servers.values()):
            continue
        if tool.name in TOOL_ADAPTERS:
            adapters[tool.name] = TOOL_ADAPTERS[tool.name](export)
            continue
        server = mcp_servers.setdefault(_server_key(tool), {"transport": "http", "url": _mcp_url(tool), "include_tools": []})
        server["include_tools"].append(tool.name)

    def mda_tool_name(tool: FleetTool) -> str:
        return tool.name if tool.name in adapters else f"{_server_key(tool)}_{tool.name}"

    for adapter in adapters.values():
        (out / "tools" / f"{adapter.module}.py").write_text(adapter.source)
        report += adapter.notes

    if mcp_servers:
        (out / "tools" / "mcp.py").write_text(textwrap.dedent('''\
            """Remote MCP servers carried over from the Fleet export (tools.json).

            Generated by utils/fleet_to_mda.py. MDA exposes each tool as
            `{server}_{tool}`. Servers that need credentials take a connection,
            e.g. "connection": connections.get("<slug>", {"type": "user"}); see
            https://docs.langchain.com/langsmith/python/managed-deep-agents-mcp-connectors
            """

            from managed_deepagents import define_mcp

            mcp = define_mcp(servers=%s)
            ''') % _py(mcp_servers))
        for key, server in mcp_servers.items():
            builtin = any(h in server["url"] for h in FLEET_BUILTIN_HOSTS)
            report.append(
                f"MCP server `{key}` ({server['url']}) carries {', '.join(server['include_tools'])}."
                + (" It is a Fleet built-in tool server whose auth Fleet manages: add a `connection` "
                   "in `tools/mcp.py` or write a tool adapter in utils/fleet_to_mda.py."
                   if builtin else "")
            )

    # Approvals.
    interrupt_on = {
        mda_tool_name(t): {"allowed_decisions": ["approve", "edit", "reject"]}
        for t in all_tools if t.interrupt
    }

    # Subagents.
    sub_blocks, sub_names = [], []
    for sub in export.subagents:
        const = re.sub(r"[^A-Za-z0-9]+", "_", sub.name).upper() + "_SUBAGENT"
        authored = [t.name for t in sub.tools if t.name in adapters]
        uses_mcp = any(t.name not in adapters for t in sub.tools)
        lines = [
            f"{const} = {{",
            f'    "name": {json.dumps(sub.name)},',
            f'    "description": {json.dumps(sub.description, ensure_ascii=False)},',
            f'    "system_prompt": (PROMPTS_DIR / "{sub.name}.md").read_text(),',
        ]
        if authored and not uses_mcp:
            lines.append(f'    "tools": [{", ".join(adapters[n].symbol for n in authored)}],')
        elif uses_mcp:
            report.append(
                f"Subagent `{sub.name}` uses MCP tools, which load at runtime, so it inherits the "
                "main agent's tools instead of a fixed list."
            )
        lines.append("}")
        sub_blocks.append("\n".join(lines))
        sub_names.append(const)

    if export.subagents:
        (out / "subagent_prompts").mkdir()
        for sub in export.subagents:
            (out / "subagent_prompts" / f"{sub.name}.md").write_text(sub.prompt)
        imports = sorted({f"from tools.{adapters[t.name].module} import {adapters[t.name].symbol}"
                          for s in export.subagents for t in s.tools if t.name in adapters})
        (out / "subagents.py").write_text(
            '"""Subagents carried over from the Fleet export (subagents/<name>/).\n\n'
            "Prompts live in subagent_prompts/ and ship with the build.\n"
            '"""\n\nfrom pathlib import Path\n\n'
            + ("\n".join(imports) + "\n\n" if imports else "")
            + 'PROMPTS_DIR = Path(__file__).resolve().parent / "subagent_prompts"\n\n'
            + "\n\n".join(sub_blocks)
            + f"\n\nSUBAGENTS = [{', '.join(sub_names)}]\n"
        )

    # Instructions and skills.
    instructions = export.prompt
    renamed = {t.name: mda_tool_name(t) for t in all_tools if mda_tool_name(t) != t.name}
    if renamed:
        instructions += "\n## Tool names in this deployment\n\n" + "".join(
            f"- `{old}` is available as `{new}`.\n" for old, new in renamed.items()
        )
    (out / "instructions.md").write_text(instructions)
    for skill in export.skills:
        if not SKILL_NAME.match(skill.name):
            report.append(f"Skill `{skill.name}` skipped: MDA skill names must match [a-z][a-z0-9-]*.")
            continue
        shutil.copytree(skill, out / "skills" / skill.name)
        if "placeholder" in (skill / "SKILL.md").read_text().lower():
            report.append(f"Skill `{skill.name}` still contains placeholder text from Fleet; finish it before deploying.")

    # Agent entry.
    main_imports = sorted({f"from tools.{adapters[t.name].module} import {adapters[t.name].symbol}"
                           for t in export.tools if t.name in adapters})
    main_tools = sorted({adapters[t.name].symbol for t in export.tools if t.name in adapters})
    gateway = model.startswith("langsmith:")
    body = [f"    name={json.dumps(name)},", f"    model={'model' if gateway else json.dumps(model)},"]
    if main_tools:
        body.append(f"    tools=[{', '.join(main_tools)}],")
    if export.subagents:
        body.append("    subagents=SUBAGENTS,")
    if interrupt_on:
        body.append(f"    interrupt_on={_py(interrupt_on)},".replace("\n", "\n    "))
    body.append(f"    metadata={{\"converted_from\": \"fleet\", \"fleet_agent_name\": {json.dumps(export.name)}}},")
    (out / "agent.py").write_text(
        f'"""{export.name}: converted from a LangSmith Fleet export by utils/fleet_to_mda.py.\n\n'
        "System prompt: instructions.md. Skills: skills/. MDA supplies the runtime,\n"
        "checkpointer, and Context Hub sync on `mda dev` / `mda deploy`.\n"
        '"""\n\n'
        + ("import os\n\nfrom langchain.chat_models import init_chat_model\n" if gateway else "")
        + "from managed_deepagents import define_deep_agent\n\n"
        + ("from subagents import SUBAGENTS\n" if export.subagents else "")
        + ("\n".join(main_imports) + "\n" if main_imports else "")
        + (GATEWAY_MODEL_BLOCK % json.dumps(model) if gateway else "")
        + "\nagent = define_deep_agent(\n" + "\n".join(body) + "\n)\n"
    )

    # Runtime declarations.
    (out / "identity.py").write_text(textwrap.dedent('''\
        """Who may call this deployment: LangSmith API keys (x-api-key)."""

        from managed_deepagents import auth, define_identity

        identity = define_identity(auth=auth.langsmith_api_key())
        '''))
    if export.backend_type not in ("", "default") or export.sandbox_snapshot_id:
        (out / "sandbox").mkdir()
        snapshot = f"\n    snapshot_id={json.dumps(export.sandbox_snapshot_id)}," if export.sandbox_snapshot_id else ""
        (out / "sandbox" / "__init__.py").write_text(
            '"""Managed sandbox (the Fleet agent used a sandbox backend)."""\n\n'
            "from managed_deepagents import define_sandbox\n\n"
            f"sandbox = define_sandbox(\n    idle_ttl_seconds=600,\n    default_timeout=600,{snapshot}\n)\n"
        )
        report.append("Fleet used a sandbox backend; declared a managed sandbox in sandbox/.")

    # Packaging.
    deps = [f"managed-deepagents=={_mda_version()}"]
    provider = model.removeprefix("langsmith:").split(":")[0].split("/")[0]
    if model.startswith("langsmith:"):  # LLM Gateway models are served through langchain-openai
        deps.append("langchain-openai>=1.0.1")
    else:
        deps.append({"openai": "langchain-openai>=1.0.1", "anthropic": "langchain-anthropic>=1.0.0",
                     "google_genai": "langchain-google-genai>=4.0.0"}.get(provider, f"langchain-{provider}"))
    deps += sorted({d for a in adapters.values() for d in a.dependencies})
    slug = slugify(name)
    (out / "pyproject.toml").write_text(
        "[project]\n"
        f'name = "{slug.replace("-", "_")}"\n'
        'version = "0.1.0"\n'
        f'description = {json.dumps(f"{export.name}, converted from LangSmith Fleet to a Managed Deep Agent.")}\n'
        'requires-python = ">=3.11"\n'
        "dependencies = [\n" + "".join(f'    "{d}",\n' for d in deps) + "]\n\n"
        f'[tool.mda]\npython-version = "{DEFAULT_PYTHON}"\n'
    )
    env_lines = [
        "# Deploy auth (mda reads this file before the shell).",
        'LANGSMITH_API_KEY="<langsmith-api-key>"',
        'LANGSMITH_WORKSPACE_ID="<workspace-id>"',
        '# LANGSMITH_ENDPOINT="https://api.smith.langchain.com"',
    ]
    if model.startswith("langsmith:"):
        env_lines += [
            "# LLM Gateway: a key with gateway:invoke (falls back to LANGSMITH_API_KEY), and the",
            "# workspace ID again under a name mda forwards to the deployment (org-scoped keys).",
            'LANGSMITH_GATEWAY_API_KEY="<langsmith-api-key>"',
            'GATEWAY_WORKSPACE_ID="<workspace-id>"',
        ]
    else:
        env_lines.append(f'{provider.upper()}_API_KEY="<{provider}-api-key>"')
    for adapter in adapters.values():
        env_lines.append(f"\n# {adapter.symbol}")
        for key, value in adapter.env.items():
            env_lines.append(f"# {key}=  {value}" if value.startswith("#") else f'{key}="{value}"')
    (out / ".env.example").write_text("\n".join(env_lines) + "\n")
    (out / ".gitignore").write_text(".env\n.env.*\n!.env.example\n.mda/\n__pycache__/\n.venv/\n")

    if export.ignored_settings:
        report.append(
            "Fleet-only settings with no MDA equivalent were dropped: "
            + ", ".join(f"`{k}={v}`" for k, v in export.ignored_settings.items()) + "."
        )
    _write_readme(out, export, name=name, model=model, adapters=adapters,
                  mcp_servers=mcp_servers, interrupt_on=interrupt_on, report=report)
    return report


def _write_readme(out, export, *, name, model, adapters, mcp_servers, interrupt_on, report) -> None:
    rows = [
        ("`AGENTS.md`", "`instructions.md`"),
        ("`config.json` model `" + export.model + "`", f"`model=\"{model}\"`"),
    ]
    rows += [(f"`skills/{s.name}/`", f"`skills/{s.name}/`") for s in export.skills]
    rows += [(f"`subagents/{s.name}/`", f"`subagents.py` + `subagent_prompts/{s.name}.md`") for s in export.subagents]
    rows += [(f"tool `{n}`", f"`tools/{a.module}.py` (authored)") for n, a in adapters.items()]
    rows += [(f"tools from `{s['url']}`", "`tools/mcp.py`") for s in mcp_servers.values()]
    rows.append(("tool approvals (`interrupt_config`)",
                 "`interrupt_on` for " + ", ".join(f"`{t}`" for t in interrupt_on) if interrupt_on else "none (all tools ran on Auto)"))
    table = "\n".join(f"| {a} | {b} |" for a, b in rows)
    notes = "\n".join(f"- {line}" for line in report) or "- Nothing needed manual attention."
    (out / "README.md").write_text(f"""# {out.name}

**{export.name}**, converted from a LangSmith Fleet export into a [Managed Deep Agents](https://docs.langchain.com/langsmith/python/managed-deep-agents-overview) project by `utils/fleet_to_mda.py`.

## What moved where

| Fleet export | Managed Deep Agent |
|---|---|
{table}

## Review before deploying

{notes}

## Run it

```bash
cp .env.example .env        # fill in keys
uv sync
uv run mda build            # compile only
uv run mda dev              # local Agent Server + Studio
uv run mda deploy --name {slugify(name)} --agent-environment development
```

Clients call it with `assistant_id="{name}"`. In the workshop UI (`app.py`), pick **Managed Deep Agent** and set the deployment URL and assistant ID under *Connection settings*.
""")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("export", type=Path, help="Fleet export folder or .zip")
    parser.add_argument("--out", type=Path, help="Output project directory (default: <agent_name>_agent)")
    parser.add_argument("--name", help="define_deep_agent name / assistant ID (default: slug of the Fleet name)")
    parser.add_argument("--model", help="Override the model string entirely")
    parser.add_argument("--no-gateway", action="store_true",
                        help="Keep the Fleet provider model string instead of routing through the LLM Gateway")
    parser.add_argument("--force", action="store_true", help="Replace the output directory if it exists")
    args = parser.parse_args(argv)

    export = load_export(args.export)
    name = args.name or slugify(export.name)
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", name):
        raise SystemExit(f"--name {name!r} must match [a-zA-Z][a-zA-Z0-9_-]*")
    model = args.model or (export.model if args.no_gateway else gateway_model(export.model))
    out = args.out or Path(name.replace("-", "_") + "_agent")
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists; pass --force to replace it.")
        shutil.rmtree(out)

    report = convert(export, out, name=name, model=model)
    print(f"Converted Fleet agent {export.name!r} -> {out}/ (assistant ID {name!r}, model {model!r})")
    for line in report:
        print(f"  ! {line}")
    print(f"\nNext: cd {out} && cp .env.example .env && uv sync && uv run mda build")


if __name__ == "__main__":
    main(sys.argv[1:])
