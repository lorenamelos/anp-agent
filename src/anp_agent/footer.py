"""Deterministic texts added to an answer (design, sections 1.1, 1.5 and 3).

Built from the state, never by the model, in the language of the question: the R4 notice
(budget reached), the R5 notice (recent months may be revised), and the footer (sources, and
values removed by redaction). A small stop-word count picks Portuguese or English,
defaulting to Portuguese.
"""

import re

from anp_agent.guardrail.match import Report

PT, EN = "pt", "en"
WORD_RE = re.compile(r"[a-zà-ÿ]+")
STOP_WORDS = {
    PT: {"de", "da", "do", "das", "dos", "que", "em", "no", "na", "nos", "nas", "para", "com",
         "qual", "quais", "quanto", "quantos", "foi", "é", "o", "os", "as", "um", "uma", "mês",
         "meses", "produção", "campo", "campos", "entre", "por", "maior", "menor", "ano"},
    EN: {"the", "of", "what", "which", "in", "on", "was", "is", "for", "with", "how", "much",
         "many", "did", "does", "a", "an", "to", "month", "months", "production", "field",
         "fields", "between", "by", "largest", "smallest", "year"},
}
MONTHS = {
    PT: ("jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"),
    EN: ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"),
}
TEXT = {
    PT: {
        "marker": "[não confirmado]",
        "sources": "Fontes",
        "removed": "Removido por não ter origem confirmada",
        "budget": "Resposta limitada pelo orçamento por pergunta",
        "tokens": "o limite de tokens foi atingido",
        "query": ("{n} consulta não foi feita", "{n} consultas não foram feitas"),
        "calc": ("{n} cálculo não foi feito", "{n} cálculos não foram feitos"),
        "for": "para",
        "and": "e",
        "values": ("{n} valor", "{n} valores"),
        "recent": "Os dados dos meses mais recentes ({months}) ainda podem ser revisados pela ANP.",
    },
    EN: {
        "marker": "[unconfirmed]",
        "sources": "Sources",
        "removed": "Removed for lack of a confirmed source",
        "budget": "Answer limited by the per-question budget",
        "tokens": "the token limit was reached",
        "query": ("{n} query was not made", "{n} queries were not made"),
        "calc": ("{n} calculation was not made", "{n} calculations were not made"),
        "for": "for",
        "and": "and",
        "values": ("{n} value", "{n} values"),
        "recent": "Data for the most recent months ({months}) may still be revised by ANP.",
    },
}
RECENT_MONTHS = 3  # ANP re-exports the two months before each release (D11)
SUBJECT_KEYS = ("field_group", "text")  # args shown bare in the footer and the R4 notice


def detect_language(question: str) -> str:
    words = WORD_RE.findall(question.lower())
    pt = sum(w in STOP_WORDS[PT] for w in words)
    en = sum(w in STOP_WORDS[EN] for w in words)
    return EN if en > pt else PT


def _plural(forms: tuple[str, str], n: int) -> str:
    return forms[n != 1].format(n=n)


def budget_notice(not_executed: list[dict], token_limited: bool, language: str) -> str | None:
    """R4: what the budget cut. `not_executed` holds {"name", "args"} of refused calls."""
    t = TEXT[language]
    parts = [t["tokens"]] if token_limited else []
    queries = [c for c in not_executed if c["name"] != "calculate"]
    if queries:
        subjects: dict[str, list[str]] = {}
        for c in queries:
            subjects.setdefault(c["name"], [])
            subjects[c["name"]] += [str(c["args"][k]) for k in SUBJECT_KEYS
                                    if c.get("args", {}).get(k)]
        detail = "; ".join(f"{name} {t['for']} {', '.join(dict.fromkeys(s))}" if s else name
                           for name, s in subjects.items())
        parts.append(f"{_plural(t['query'], len(queries))} ({detail})")
    if calcs := len(not_executed) - len(queries):
        parts.append(_plural(t["calc"], calcs))
    if not parts:
        return None
    body = parts[0] if len(parts) == 1 else f"{'; '.join(parts[:-1])} {t['and']} {parts[-1]}"
    return f"{t['budget']}: {body}."


def recent_months(latest_month: str) -> list[str]:
    """The dataset's last RECENT_MONTHS months, oldest first ("2026-07", ..., "2026-09")."""
    y, m = int(latest_month[:4]), int(latest_month[5:7])
    out = []
    for back in range(RECENT_MONTHS - 1, -1, -1):
        yy, mm = divmod(y * 12 + m - 1 - back, 12)
        out.append(f"{yy}-{mm + 1:02d}")
    return out


def uses_recent_months(report: Report, latest_month: str) -> bool:
    """R5: does a grounded number of the answer cover, or a date of it name, a recent month?"""
    first = recent_months(latest_month)[0]
    for v in report.verdicts:
        if not v.grounded:
            continue
        if any(c.until and c.until >= first for c in v.used):
            return True
        if v.claim.month and f"{v.claim.year}-{v.claim.month:02d}" >= first:
            return True
    return False


def recent_notice(latest_month: str, language: str) -> str:
    months = recent_months(latest_month)
    names = MONTHS[language]
    (y0, m0), (y1, m1) = ((int(s[:4]), int(s[5:])) for s in (months[0], months[-1]))
    if language == PT:
        span = (f"{names[m0 - 1]}–{names[m1 - 1]}/{y1}" if y0 == y1
                else f"{names[m0 - 1]}/{y0}–{names[m1 - 1]}/{y1}")
    else:
        span = (f"{names[m0 - 1]}–{names[m1 - 1]} {y1}" if y0 == y1
                else f"{names[m0 - 1]} {y0}–{names[m1 - 1]} {y1}")
    return TEXT[language]["recent"].format(months=span)


def describe_call(call: dict, language: str) -> str:
    """field_group_production(BÚZIOS, 2026-01..2026-09); calculate(mean, 9 valores)."""
    name, args = call["name"], dict(call.get("args") or {})
    if name == "calculate":
        n = len(args.get("values") or [])
        return f"calculate({args.get('operation')}, {_plural(TEXT[language]['values'], n)})"
    parts = [str(args.pop(k)) for k in SUBJECT_KEYS if k in args]
    start, end = args.pop("start", None), args.pop("end", None)
    if start or end:
        parts.append(f"{start or ''}..{end or ''}")
    parts += [f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in args.items()]
    return f"{name}({', '.join(parts)})"


def sources_line(calls: list[dict], report: Report, language: str) -> str | None:
    """Every call (1-based, as in the provenance sources) that grounded a claim of the answer,
    including the calls a calculator result took its inputs from."""
    used = sorted({n for v in report.verdicts if v.grounded for c in v.used
                   for n in (c.call, *c.based_on) if n is not None and 1 <= n <= len(calls)})
    if not used:
        return None
    described = dict.fromkeys(describe_call(calls[i - 1], language) for i in used)
    return f"{TEXT[language]['sources']}: {'; '.join(described)}"


def removed_line(removed: list[str], language: str) -> str | None:
    if not removed:
        return None
    return f"{TEXT[language]['removed']}: " + "; ".join(f'"{r}"' for r in removed)


def finalize(answer: str, *, language: str, calls: list[dict], report: Report,
             removed: list[str] = (), latest_month: str | None = None,
             not_executed: list[dict] = (), token_limited: bool = False) -> str:
    """The answer as the user sees it: text, notices, then the footer.

    `report` is the guardrail report of the final draft; only its grounded claims are used,
    so the same report serves before and after redaction."""
    notices = [budget_notice(list(not_executed), token_limited, language)]
    if latest_month and uses_recent_months(report, latest_month):
        notices.append(recent_notice(latest_month, language))
    footer = [sources_line(calls, report, language), removed_line(list(removed), language)]
    blocks = [answer.rstrip(), *(n for n in notices if n)]
    if lines := [f for f in footer if f]:
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
