"""Redaction (design, section 1.1, D6): replace each ungrounded number span with a marker.

Only the span the extractor found (number, scale word, unit) is replaced; the words around it
stay. Deterministic, with no new model text, so there is nothing new to verify.
"""

from anp_agent.footer import TEXT
from anp_agent.guardrail.match import Report


def redact(answer: str, report: Report, language: str) -> tuple[str, list[str]]:
    """(redacted answer, removed spans in reading order)."""
    marker = TEXT[language]["marker"]
    spans = sorted({(v.claim.start, v.claim.end) for v in report.ungrounded})
    out, removed, pos = [], [], 0
    for start, end in spans:
        if start < pos:  # overlapping spans: the first one already covers it
            continue
        out += [answer[pos:start], marker]
        removed.append(answer[start:end])
        pos = end
    out.append(answer[pos:])
    return "".join(out), removed
