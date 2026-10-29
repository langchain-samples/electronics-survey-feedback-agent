"""Survey-feedback tools for the Managed Deep Agent (and the Google ADK agent).

The functions are plain Python with typed signatures and Google-style
docstrings, so both frameworks use them as-is: Google ADK accepts plain
functions as tools, and the Managed Deep Agent gets LangChain tools via
`LANGCHAIN_TOOLS`. Same tools in both agents keeps the comparison fair: the
same evaluators score the same kind of output.

Survey exports ship inside the project (`data/`), because `mda deploy` copies
only the project directory into the build.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import requests
from langchain_core.tools import tool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SURVEY_EXPORT = DATA_DIR / "survey_export.csv"
# Simulated posts go outside the project so they never ship with `mda deploy`.
OUTBOX = Path(os.environ.get("SURVEY_OUTBOX", Path(tempfile.gettempdir()) / "survey_outbox.jsonl"))


@lru_cache(maxsize=1)
def load_survey_export() -> tuple[dict, ...]:
    """Load the survey export once and return its rows as dicts."""
    with SURVEY_EXPORT.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            row["nps"] = int(row["nps"])
            rows.append(row)
        return tuple(rows)


def _nps(scores: list[int]) -> float:
    if not scores:
        return 0.0
    promoters = sum(s >= 9 for s in scores)
    detractors = sum(s <= 6 for s in scores)
    return round(100 * (promoters - detractors) / len(scores), 1)


def read_survey_export(offset: int = 0, limit: int = 20) -> dict:
    """Read raw rows from the latest customer survey export.

    Use this to see what the export contains (columns, date range, sample
    comments). For counts and NPS use get_survey_stats; to find quotes on a
    topic use search_responses.

    Args:
        offset: Row to start from (0-based).
        limit: Maximum number of rows to return (at most 50).

    Returns:
        The export's file name, total row count, column names, and the requested rows.
    """
    rows = load_survey_export()
    limit = max(1, min(limit, 50))
    return {
        "export": SURVEY_EXPORT.name,
        "total_rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "date_range": [rows[0]["submitted_at"], rows[-1]["submitted_at"]] if rows else [],
        "rows": [dict(r) for r in rows[offset:offset + limit]],
    }


def get_survey_stats(group_by: str = "channel") -> dict:
    """Compute response counts and NPS for the survey export.

    Use this for any number you report (counts, NPS, shares) instead of
    estimating from the raw text.

    Args:
        group_by: Column to break results down by: "channel" or "product_category".

    Returns:
        Overall response count and NPS, plus the same figures for each group.
    """
    if group_by not in {"channel", "product_category"}:
        return {"error": "group_by must be 'channel' or 'product_category'"}
    rows = load_survey_export()
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[row[group_by]].append(row["nps"])
    return {
        "total_responses": len(rows),
        "overall_nps": _nps([r["nps"] for r in rows]),
        "group_by": group_by,
        "groups": {
            name: {"responses": len(scores), "nps": _nps(scores)}
            for name, scores in sorted(groups.items())
        },
    }


def search_responses(keywords: list[str], max_results: int = 10) -> list[dict]:
    """Find survey responses whose comment mentions any of the keywords.

    Use this to collect verbatim quotes that back up a theme. Quote the
    `comment` text exactly and cite its `response_id`.

    Args:
        keywords: Case-insensitive words or phrases to look for, e.g. ["curbside", "pickup"].
        max_results: Maximum number of responses to return.

    Returns:
        Matching responses with response_id, channel, product_category, nps, and comment.
    """
    needles = [k.lower() for k in keywords if k.strip()]
    matches = [
        row for row in load_survey_export()
        if any(n in row["comment"].lower() for n in needles)
    ]
    return [dict(row) for row in matches[:max_results]]


def theme_counts(response_ids: list[str]) -> dict:
    """Summarize channel mix and average NPS score for a set of responses.

    Args:
        response_ids: Response IDs that belong to one theme, e.g. ["R-1002", "R-1009"].

    Returns:
        How many of the IDs exist, their channel mix, and their average NPS score.
    """
    wanted = set(response_ids)
    rows = [r for r in load_survey_export() if r["response_id"] in wanted]
    if not rows:
        return {"matched": 0}
    return {
        "matched": len(rows),
        "unknown_ids": sorted(wanted - {r["response_id"] for r in rows}),
        "channels": dict(Counter(r["channel"] for r in rows)),
        "avg_nps_score": round(sum(r["nps"] for r in rows) / len(rows), 1),
    }


def post_themes_to_channel(channel: str, message: str) -> str:
    """Post the approved survey-theme summary to a Slack or Teams channel.

    This sends a message outside the agent, so it must only run after a human
    approves it.

    Args:
        channel: Destination channel name, e.g. "#survey-insights".
        message: The summary to post.

    Returns:
        Confirmation of where the message went.
    """
    webhook = os.environ.get("SURVEY_WEBHOOK_URL")
    if webhook:
        resp = requests.post(webhook, json={"text": f"*{channel}*\n{message}"}, timeout=15)
        resp.raise_for_status()
        return f"Posted to {channel} via webhook."
    record = {
        "posted_at": datetime.now(timezone.utc).isoformat(),
        "channel": channel,
        "message": message,
    }
    try:
        with OUTBOX.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass
    return (
        f"Posted to {channel} (simulated: no SURVEY_WEBHOOK_URL set, "
        f"message recorded to the outbox)."
    )


PLAIN_TOOLS = [read_survey_export, get_survey_stats, search_responses, theme_counts, post_themes_to_channel]
LANGCHAIN_TOOLS = [tool(fn, parse_docstring=True) for fn in PLAIN_TOOLS]
TOOLS_BY_NAME = {t.name: t for t in LANGCHAIN_TOOLS}
