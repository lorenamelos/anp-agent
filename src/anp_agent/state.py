"""Per-question state records (design, section 1.2)."""

from typing import Literal, TypedDict

ToolStatus = Literal["ok", "tool_error", "timeout", "transport_error", "not_executed"]


class ToolRecord(TypedDict):
    call_id: str
    name: str  # anp-mcp tool name or "calculate"
    args: dict
    result_text: str  # what the model saw
    structured: dict | None  # parsed JSON (source of numbers for R1)
    status: ToolStatus
    seconds: float
