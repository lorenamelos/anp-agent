"""The answer when no LLM call fits the budget (design, section 3): deterministic, readable,
and made only of values the tools returned.

    one sentence saying the limit was reached
    what the data obtained shows: each successful call's values, labelled, with units
    the queries made, and those left out

No raw JSON and no interpretation: the values are the tool results' own leaves, formatted
for the question's language, so the guardrail grounds every one of them.
"""

from anp_agent import footer
from anp_agent.guardrail.sources import unit_of
from anp_agent.state import ToolRecord

MAX_ROWS = 12  # rows shown per list: a time series shows its latest, a ranking its top
ROW_LABELS = ("month", "year", "field_group")  # the key that names a row, in order
LABELS = {
    "pt": {"petroleum_bpd": "petróleo", "onshore_bpd": "terra", "offshore_bpd": "mar",
           "active_wells": "poços ativos", "before_bpd": "antes", "recent_bpd": "recente",
           "change_bpd": "variação", "change_pct": "variação", "window_months": "janela (meses)",
           "latest_petroleum_bpd": "produção no último mês", "latest_month": "último mês"},
    "en": {"petroleum_bpd": "petroleum", "onshore_bpd": "onshore", "offshore_bpd": "offshore",
           "active_wells": "active wells", "before_bpd": "before", "recent_bpd": "recent",
           "change_bpd": "change", "change_pct": "change", "window_months": "window (months)",
           "latest_petroleum_bpd": "latest-month production", "latest_month": "latest month"},
}
TEXT = {
    "pt": {
        "intro": ("O limite por pergunta foi atingido antes de concluir a resposta. Seguem os "
                  "valores que as consultas devolveram, sem interpretação."),
        "shows": "O que os dados obtidos mostram:",
        "made": "Consultas feitas:",
        "left": "Ficaram de fora:",
        "none": "nenhuma",
        "earlier": "(+{n} meses anteriores omitidos)",
        "more": "(+{n} linhas seguintes omitidas)",
        "empty": "sem dados",
        "answer": "a conclusão da resposta (limite de tokens)",
    },
    "en": {
        "intro": ("The per-question limit was reached before the answer was finished. These "
                  "are the values the queries returned, without interpretation."),
        "shows": "What the data obtained shows:",
        "made": "Queries made:",
        "left": "Left out:",
        "none": "none",
        "earlier": "(+{n} earlier periods omitted)",
        "more": "(+{n} more rows omitted)",
        "empty": "no data",
        "answer": "the conclusion of the answer (token limit)",
    },
}


def render(records: list[ToolRecord], not_executed: list[dict], token_limited: bool,
           language: str) -> str:
    t = TEXT[language]
    lines = [t["intro"], "", t["shows"]]
    for r in records:
        if r["status"] == "ok" and isinstance(r["structured"], dict):
            head, rows = _values(r["name"], r["structured"], language)
            lines.append(f"- {footer.describe_call(r, language)}: {head or t['empty']}"
                         if head or not rows else f"- {footer.describe_call(r, language)}:")
            lines += [f"  - {row}" for row in rows]
    made = [footer.describe_call(r, language) for r in records]
    left = [footer.describe_call(c, language) for c in not_executed]
    if token_limited:
        left.append(t["answer"])
    lines += ["", f"{t['made']} {'; '.join(made) or t['none']}",
              f"{t['left']} {'; '.join(left) or t['none']}"]
    return "\n".join(lines)


def _values(name: str, result: dict, language: str) -> tuple[str, list[str]]:
    """(the call's own values on one line, one line per row of its lists)."""
    if name == "calculate":
        return _number(result.get("result"), result.get("unit"), language), []
    head, rows = [], []
    for value in result.values():
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            rows += _rows(value, language)
        elif isinstance(value, dict):
            head.append(_fields(value, language))
    scalars = _fields({k: v for k, v in result.items() if not isinstance(v, list | dict)},
                      language, numbers_only=True)
    return "; ".join(p for p in (scalars, *head) if p), rows


def _rows(rows: list[dict], language: str) -> list[str]:
    t = TEXT[language]
    series = any(k in rows[0] for k in ("month", "year"))
    shown = rows[-MAX_ROWS:] if series else rows[:MAX_ROWS]
    out = []
    for row in shown:
        label = next((str(row[k]) for k in ROW_LABELS if isinstance(row.get(k), str | int)), "")
        fields = _fields({k: v for k, v in row.items() if k not in ROW_LABELS}, language,
                         numbers_only=True)
        out.append(f"{label}: {fields}" if label else fields)
    hidden = len(rows) - len(shown)
    if hidden and series:
        out.insert(0, t["earlier"].format(n=hidden))
    elif hidden:
        out.append(t["more"].format(n=hidden))
    return out


def _fields(fields: dict, language: str, numbers_only: bool = False) -> str:
    """Labelled values: numbers with their unit, and month strings; other text is left out."""
    out = []
    for key, value in fields.items():
        label = LABELS[language].get(key, key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            out.append(f"{label} {_number(value, unit_of(key), language)}")
        elif isinstance(value, str) and (key.endswith("month")
                                         or (not numbers_only and len(value) <= 10)):
            out.append(f"{label} {value}")
    return ", ".join(out)


def _number(value, unit: str | None, language: str) -> str:
    if not isinstance(value, int | float):
        return str(value)
    text = f"{value:,.0f}" if float(value).is_integer() else f"{value:,.3f}".rstrip("0")
    if language == "pt":
        text = text.replace(",", "\0").replace(".", ",").replace("\0", ".")
    if unit == "%":
        return f"{text}%"
    if unit == "wells" or unit is None:
        return text
    return f"{text} {unit.replace('m3', 'm³')}"
