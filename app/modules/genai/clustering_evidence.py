"""Cluster patterns and citation-bound external context; no fitting or assignment changes."""
from __future__ import annotations
import math
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

CLUSTER_LABELS = {"SUPPORTED", "PARTIALLY_SUPPORTED", "CONTEXTUALLY_RELEVANT", "NOT_FOUND", "CONTRADICTORY"}
CLUSTER_DISCLAIMER = ("Native cluster profiles describe groups discovered in this dataset. External sources may "
                      "support a similar pattern or provide context; they do not prove this cluster is correct or establish causation.")

def canonical_url(value: str) -> str:
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
        return ""
    port = parsed.port
    authority = host + (f":{port}" if port and port not in {80, 443} else "")
    return urlunsplit(("https", authority, parsed.path.rstrip("/") or "/", "", ""))

def source_quality(source: dict[str, Any]) -> tuple[int, str]:
    host = urlsplit(source.get("url") or "").hostname or ""
    metadata = str(source.get("title") or "") + " " + str(source.get("source_type") or "")
    # Excerpts mentioning a study do not make the article itself peer-reviewed.
    if re.search(r"peer.review|systematic review|meta.analysis|academic journal|\bdoi\b", metadata, re.I):
        return 1, "ACADEMIC_INDICATOR"
    if re.search(r"\.gov$|\.(?:gov|ac|edu)\.[a-z]{2}$|\.edu$", host, re.I):
        return 2, "PUBLIC_OR_ACADEMIC_INSTITUTION"
    if re.search(r"research (?:institute|organization|organisation|foundation|centre|center)", metadata, re.I):
        return 3, "RESEARCH_ORGANIZATION_INDICATOR"
    if re.search(r"industry (?:study|survey|report)|research report|survey methodology", metadata, re.I):
        return 4, "INDUSTRY_STUDY_INDICATOR"
    if re.search(r"reporting|journalism|financial news|business news", metadata, re.I):
        return 5, "JOURNALISM_INDICATOR"
    return 6, "GENERAL_CONTEXT"

def deduplicate_sources(sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    accepted, urls, titles, aliases = [], {}, {}, {}
    for source in sources:
        try:
            url = canonical_url(str(source.get("url") or ""))
        except ValueError:
            continue
        if not url: continue
        title_key = (urlsplit(url).hostname, re.sub(r"\W+", " ", str(source.get("title") or "").lower()).strip())
        existing = urls.get(url) or (titles.get(title_key) if title_key[1] else None)
        if existing:
            aliases[str(source["id"])] = str(existing["id"])
            continue
        rank, quality = source_quality(source)
        original = urlsplit(str(source["url"]))
        citation_url = urlunsplit((original.scheme, original.netloc, original.path, original.query, ""))
        row = {**source, "url": citation_url, "canonical_url": url, "quality_rank": rank, "quality_category": quality,
               "quality_note": "Metadata-based preference; independent credibility is not guaranteed."}
        accepted.append(row)
        urls[url] = row
        if title_key[1]: titles[title_key] = row
        aliases[str(source["id"])] = str(row["id"])
    return sorted(accepted, key=lambda row: row["quality_rank"]), aliases

def cluster_patterns(native: dict[str, Any], business_context: str = "", understanding: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    columns = {c["native_name"]: c for c in (understanding or {}).get("columns") or []}
    patterns = []
    for key, profile in (native.get("cluster_profiles") or {}).items():
        traits = profile.get("characteristics") or []
        if str(key) == "-1" or not traits: continue
        differentiators = []
        for statement in traits[:5]:
            statement = str(statement)[:600]
            feature, separator, detail = statement.partition(" is ")
            if not separator: continue
            numbers = [abs(float(v)) for v in re.findall(r"([+-]?\d+(?:\.\d+)?) SD", detail)]
            if not all(math.isfinite(v) for v in numbers): continue
            relative = re.search(r"(\d+(?:\.\d+)?) SD (above|below) other clusters", detail)
            overall = re.search(r"(\d+(?:\.\d+)?) SD (above|below) the overall mean", detail)
            chosen = relative if relative and float(relative[1]) >= .25 else overall
            direction = ("high" if chosen[2] == "above" else "low") if chosen else "typical_category" if detail.startswith("typically ") else "unspecified"
            column = next((c for name, c in columns.items() if re.sub(r"[_-]+", " ", name).lower() == feature.lower()), {})
            meaning = column.get("likely_meaning") if column.get("learner_confirmed") else feature
            differentiators.append({"feature": feature, "direction": direction, "relative_strength": max(numbers) if numbers else None,
                "display_meaning": meaning, "native_statement": statement,
                "strength_method": "maximum reported absolute SD deviation" if numbers else "native categorical profile"})
        if not differentiators: continue
        patterns.append({"cluster_id": str(key), "cluster_name": str(profile.get("segment_label") or f"Cluster {key}"),
            "differentiators": differentiators, "profile_summary": str(profile.get("profile") or "")[:1000],
            "business_context": business_context[:1000], "cluster_size": profile.get("count") or profile.get("size")})
    return patterns[:20]

def cluster_findings(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": "cluster:" + pattern["cluster_id"], "feature": pattern["cluster_name"],
             "native_statement": pattern["profile_summary"], "cluster_pattern": pattern} for pattern in patterns[:5]]

def research_query(record: dict[str, Any], domain: str = "") -> str:
    patterns = record["native_evidence"].get("cluster_patterns") or cluster_patterns(record["native_evidence"], record.get("business_problem") or "")
    concepts = []
    for pattern in patterns[:5]:
        terms = []
        for trait in pattern["differentiators"][:3]:
            terms.append(trait["native_statement"] if trait["direction"] == "typical_category"
                         else trait["direction"] + " " + trait["display_meaning"])
        concepts.append(" + ".join(terms))
    if not concepts: raise ValueError("No measured cluster pattern is available for research.")
    context = str(record.get("business_problem") or "")[:500] + " " + domain[:200]
    return re.sub(r"https?://\S+", "", "Patterns: " + "; ".join(concepts) + " " + context +
                  " recent behavioral pattern research academic studies public institutions industry surveys")[:2200]


def direction_matches(trait: dict[str, Any], quote: str, contradictory: bool) -> bool:
    # Attribute-local direction prevents "high A, low B" from supporting "low A, high B".
    for clause in re.split(r"[;,]|\band\b|\bwhile\b|\bbut\b|\bwhereas\b", quote, flags=re.I):
        if not re.search(re.escape(trait["feature"]), clause, re.I): continue
        if re.search(r"\b(?:not|never|without)\b", clause, re.I): continue
        high = bool(re.search(r"\b(?:high\w*|increas\w*|greater|above)\b", clause, re.I))
        low = bool(re.search(r"\b(?:low\w*|declin\w*|less|below)\b", clause, re.I))
        if trait["direction"] == "typical_category":
            category = re.search(r"is typically (.*?) \(", trait["native_statement"])
            if not contradictory and category and re.search(re.escape(category[1]), clause, re.I) and not re.search(r"\b(?:not|without)\b", clause, re.I):
                return True
        elif high != low:
            expected = trait["direction"]
            if contradictory: expected = "low" if expected == "high" else "high" if expected == "low" else "unspecified"
            if expected == "high" and high or expected == "low" and low: return True
    return False

def normalize_comparisons(proposals: list[dict[str, Any]], findings: list[dict[str, Any]], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {str(s["id"]): s for s in sources}
    by_finding = {str(p.get("finding_id")): p for p in proposals}
    rows = []
    reasons = {
        "SUPPORTED": "Quoted external evidence closely matches the profile attributes; this does not verify the discovered cluster.",
        "PARTIALLY_SUPPORTED": "Quoted evidence aligns with some profile attributes; remaining attributes are not established.",
        "CONTEXTUALLY_RELEVANT": "The source provides related context, without independently validating this cluster pattern.",
        "CONTRADICTORY": "Quoted evidence conflicts with an explicitly comparable profile interpretation; populations and conditions may differ.",
        "NOT_FOUND": "No reliable external evidence was found for this finding.",
    }
    for finding in findings:
        pattern = finding["cluster_pattern"]
        proposal = by_finding.get(finding["id"], {})
        matches, citations, seen_sources = [], [], set()
        traits = {d["feature"]: d for d in pattern["differentiators"]}
        for candidate in proposal.get("citations") or []:
            if not isinstance(candidate, dict): continue
            source = indexed.get(str(candidate.get("source_id")))
            quote = str(candidate.get("quote") or "").strip()
            if not source or str(source["id"]) in seen_sources or not 20 <= len(quote) <= 300 or quote not in source["evidence_statement"]: continue
            matched = [name for name in candidate.get("matched_differentiators") or [] if isinstance(name, str) and name in traits]
            # Strong matches require quoted attribute AND direction evidence, not only an LLM label.
            proposed_label = candidate.get("match_status", proposal.get("agreement", "CONTEXTUALLY_RELEVANT"))
            direct = [name for name in matched if direction_matches(traits[name], quote, proposed_label == "CONTRADICTORY")]
            label = proposed_label
            if label not in CLUSTER_LABELS: label = "CONTEXTUALLY_RELEVANT"
            if label in {"SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTORY"}:
                if source.get("quality_rank", 6) >= 5 or not direct:
                    label = "CONTEXTUALLY_RELEVANT"
                elif label == "SUPPORTED" and (len(direct) < 2 or len(direct) < len(traits)):
                    label = "PARTIALLY_SUPPORTED"
            if label == "NOT_FOUND": continue
            citation = {"source_id": source["id"], "title": source["title"], "url": source["url"],
                        "date": source.get("publication_date"), "quote": quote}
            seen_sources.add(str(source["id"]))
            citations.append(citation)
            matches.append({"cluster_id": pattern["cluster_id"], "source_id": source["id"],
                "matched_differentiators": direct if label != "CONTEXTUALLY_RELEVANT" else matched,
                "match_status": label, "rationale": reasons[label], "citation": citation})
        statuses = {m["match_status"] for m in matches}
        label = ("PARTIALLY_SUPPORTED" if "CONTRADICTORY" in statuses and statuses & {"SUPPORTED", "PARTIALLY_SUPPORTED"}
                 else "CONTRADICTORY" if "CONTRADICTORY" in statuses else "SUPPORTED" if "SUPPORTED" in statuses
                 else "PARTIALLY_SUPPORTED" if "PARTIALLY_SUPPORTED" in statuses else "CONTEXTUALLY_RELEVANT" if matches else "NOT_FOUND")
        rows.append({"finding_id": finding["id"], "feature": pattern["cluster_name"], "cluster_id": pattern["cluster_id"],
            "agreement": label, "evidence_statement": reasons[label], "citations": citations,
            "source_matches": matches, "business_interpretation": "Use the native profile to explore possible decisions; assess feasibility, risks and limitations separately."})
    return rows
