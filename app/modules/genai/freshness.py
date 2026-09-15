"""Shared, conservative policy for requests that need live public evidence."""
from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta


def freshness_requirement(query: str) -> tuple[str | None, datetime | None]:
    if re.search(r"\b(this month)\b", query, re.I):
        days, window = 31, "month"
    elif re.search(r"\b(this week|recent)\b", query, re.I):
        days, window = 7, "week"
    elif re.search(r"\b(latest|current|today|news|live|now|tomorrow|search the web)\b", query, re.I):
        days, window = 1, "day"
    else:
        # Historical and explanatory questions do not require a live lookup.
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
        if not changing_fact:
            return None, None
        days, window = 1, "day"
    return window, datetime.now(UTC) - timedelta(days=days)
