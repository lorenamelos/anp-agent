"""Ground truth for instrument agent-v1 (design, section 4.1).

Extends anp-mcp's evals/ground_truth.py without editing it: every key of its compute() plus
the keys below, each a query on the same dataset files the server reads, never typed by hand.
National figures follow the national_production tool: yearly values are day-weighted means.

    uv run python -m agent_evals.ground_truth_agent
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "anp-mcp"))
from evals.ground_truth import compute as compute_v3
from evals.ground_truth import connect, one

DAYS = "day(last_day(period))"


def national_year(con, column: str, year: int) -> float:
    """Day-weighted yearly mean, as national_production(frequency='year') computes it."""
    return one(con, f"""SELECT sum({column} * {DAYS}) / sum({DAYS}) FROM n
                        WHERE year(CAST(period AS DATE)) = {year}""")


def compute() -> dict:
    t = compute_v3()
    con = connect()
    latest = t["latest_month"]
    # share of the national total
    t["national_2026_09_bpd"] = one(con, """SELECT national_total FROM n
                                           WHERE CAST(period AS DATE) = '2026-09-01'""")
    t["buzios_share_2026_09_pct"] = 100 * t["buzios_2026_09_bpd"] / t["national_2026_09_bpd"]
    # ranking, then a lookup on each of the 5
    top5 = t["top5_up_6m"]
    names = ", ".join(f"'{g}'" for g in top5)
    t["top5_up_6m_most_wells"], t["top5_up_6m_most_wells_count"] = con.execute(f"""
        SELECT field_group, active_wells FROM g
        WHERE period = (SELECT max(period) FROM g) AND field_group IN ({names})
        ORDER BY active_wells DESC LIMIT 1""").fetchone()
    # old name (Lula is in the TUPI group) and a comparison of 2025 means
    t["mero_2025_mean_bpd"] = one(con, """SELECT avg(petroleum_bpd) FROM g
                                         WHERE field_group = 'MERO' AND year(period) = 2025""")
    t["tupi_vs_mero_2025"] = "TUPI" if t["tupi_2025_mean_bpd"] > t["mero_2025_mean_bpd"] else "MERO"
    # national, year over year
    t["national_2024_bpd"] = national_year(con, "national_total", 2024)
    t["national_2025_bpd"] = national_year(con, "national_total", 2025)
    t["national_2025_vs_2024_bpd"] = t["national_2025_bpd"] - t["national_2024_bpd"]
    t["national_2025_vs_2024_pct"] = 100 * (t["national_2025_bpd"] / t["national_2024_bpd"] - 1)
    t["national_offshore_2025_bpd"] = national_year(con, "national_mar", 2025)
    # the 10 largest field groups of 2025 (budget stress: more than the 6-call cap)
    t["top10_2025"] = [r[0] for r in con.execute("""
        SELECT field_group FROM g WHERE year(period) = 2025
        GROUP BY 1 ORDER BY avg(petroleum_bpd) DESC LIMIT 10""").fetchall()]
    # a derivation the guardrail must see grounded: Búzios in barrels per year
    t["buzios_latest_x365_bbl"] = t["buzios_latest_bpd"] * 365
    t["buzios_2025_total_bbl"] = one(con, f"""SELECT sum(petroleum_bpd * {DAYS}) FROM g
                                             WHERE field_group = 'BÚZIOS'
                                               AND year(period) = 2025""")
    t["latest_month"] = latest
    return t


if __name__ == "__main__":
    for key, value in compute().items():
        print(f"{key:32} {value}")
