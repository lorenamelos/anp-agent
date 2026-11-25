"""The eval instrument is reused from vendor/anp-mcp: it must be the pinned tag, unmodified,
and the same commit as the anp-mcp package the agent runs."""

import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "anp-mcp"
TAG = "v0.1.0"
TAG_COMMIT = "a3132497e765657a31a1e7d2d32bbe01ee1b42d0"  # v0.1.0; CI checkouts carry no tags


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(VENDOR), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_vendor_is_checked_out():
    assert (VENDOR / "evals" / "run_evals.py").is_file(), "run: git submodule update --init"


def test_vendor_is_at_the_tag():
    assert git("rev-parse", "HEAD") == TAG_COMMIT


def test_vendor_has_no_local_changes():
    assert git("status", "--porcelain") == ""


def test_lock_uses_the_same_commit():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    pkg = next(p for p in lock["package"] if p["name"] == "anp-mcp")
    source = pkg["source"]["git"]
    assert f"tag={TAG}" in source
    locked = re.search(r"#([0-9a-f]{40})$", source).group(1)
    assert locked == git("rev-parse", "HEAD")
