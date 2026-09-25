"""Shared, conservative policy for requests that need live public evidence."""
from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta


def explicit_freshness_requirement(query: str) -> tuple[str | None, datetime | None]:
    if re.search(r"\b(this month)\b", query, re.I):
        days, window = 31, "month"
    elif re.search(r"\b(this week|recent)\b", query, re.I):
        days, window = 7, "week"
    elif re.search(r"\b(latest|current|today|news|live|now|tomorrow)\b", query, re.I):
        days, window = 1, "day"
    else:
        return None, None
    return window, datetime.now(UTC) - timedelta(days=days)


def freshness_requirement(query: str) -> tuple[str | None, datetime | None]:
    explicit = explicit_freshness_requirement(query)
    if explicit[0]:
        return explicit
    if re.search(r"\bsearch the web\b", query, re.I):
        return "day", datetime.now(UTC) - timedelta(days=1)
    # Changing facts still trigger web lookup, without imposing a date cutoff.
    if re.search(r"\b(what (?:is|are) (?:a|an)|how (?:does|do|to)|explain|definition|history|historical|was|were|in (?:19|20)\d{2})\b", query, re.I):
        return None, None
    changing_fact = re.search(
        r"\b(?:who (?:is|are|leads|runs)|(?:ceo|president|prime minister|governor|mayor) of|"
        r"(?:stock|share|bitcoin|crypto|gold|oil) prices?|exchange rates?|interest rates?|"
        r"weather|forecast|election results?|scores?|standings|opening hours|"
        r"(?:is|are) .{0,60}(?:open|available)|release date|"
        r"(?:price|cost) of|how much (?:does|is)|who won|trading at|market cap|"
        r"(?:software|python|node|react) version|version of|flight status|"
        r"travel restrictions|tax rates|release schedule)\b",
        query, re.I,
    )
    return ("day", datetime.now(UTC) - timedelta(days=1)) if changing_fact else (None, None)
