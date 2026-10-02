"""Core result→argument bleed detection over agent tool-call JSONL traces."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class ToolCall:
    """A tool invocation with stringified arguments."""

    index: int  # line/record index (0-based)
    name: str
    arguments: str
    call_id: str | None = None


@dataclass
class ToolResult:
    """A tool result payload."""

    index: int
    name: str
    content: str
    call_id: str | None = None


@dataclass
class BleedHit:
    """A result substring that later appears in tool-call arguments."""

    result_index: int
    result_tool: str
    call_index: int
    call_tool: str
    snippet: str
    candidate_len: int
    source: str = ""  # file path label


@dataclass
class Report:
    hits: list[BleedHit] = field(default_factory=list)
    records: int = 0
    tool_calls: int = 0
    tool_results: int = 0
    candidates: int = 0
    source: str = ""

    @property
    def hit_count(self) -> int:
        return len(self.hits)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _snippet(text: str, limit: int = 96) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _stringify_args(args: Any) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        # Try parse JSON string args (OpenAI style) then re-dump for stability
        stripped = args.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.dumps(json.loads(stripped), ensure_ascii=False, sort_keys=True)
            except (json.JSONDecodeError, TypeError):
                return args
        return args
    return _as_text(args)


_WS_ONLY = re.compile(r"^\s*$")
_TOKEN_RE = re.compile(r"\S+")


def extract_candidates(text: str, min_len: int = 32) -> list[str]:
    """Extract candidate substrings from a tool result.

    Prefers whole lines and long non-whitespace tokens. Skips whitespace-only
    and ultra-short junk. Candidates are unique, longest-first friendly.
    """
    if not text or min_len < 1:
        return []

    found: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        s = s.strip()
        if len(s) < min_len:
            return
        if _WS_ONLY.match(s):
            return
        if s in seen:
            return
        # Skip if only punctuation/whitespace after strip of alnum? keep pragmatic
        if not any(c.isalnum() for c in s):
            return
        seen.add(s)
        found.append(s)

    # Whole lines
    for line in text.splitlines():
        add(line)

    # Long contiguous non-whitespace tokens
    for m in _TOKEN_RE.finditer(text):
        add(m.group(0))

    # Long quoted strings (secrets, paths often appear quoted)
    for m in re.finditer(r'''["']([^"']{%d,})["']''' % min_len, text):
        add(m.group(1))

    # If the whole result is one long blob without newlines, include a
    # contiguous run of the full stripped text.
    stripped = text.strip()
    if "\n" not in stripped:
        add(stripped)

    # Prefer longer candidates first (better evidence, fewer trivial hits)
    found.sort(key=len, reverse=True)
    return found


def _iter_jsonl_records(path: Path | None, text: str | None = None) -> Iterator[tuple[int, dict[str, Any]]]:
    if path is not None:
        raw = path.read_text(encoding="utf-8", errors="replace")
    else:
        raw = text or ""
    raw = raw.strip()
    if not raw:
        return
    # JSON array support
    if raw.startswith("["):
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON array: {exc}") from exc
        if not isinstance(arr, list):
            raise ValueError("expected JSON array or JSONL object lines")
        for i, item in enumerate(arr):
            if isinstance(item, dict):
                yield i, item
        return
    for i, line in enumerate(raw.splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {i + 1}: invalid JSON: {exc}") from exc
        if isinstance(obj, dict):
            yield i, obj


def _name_from_call(obj: dict[str, Any]) -> str:
    for key in ("name", "tool", "tool_name", "function_name"):
        if key in obj and obj[key]:
            return str(obj[key])
    fn = obj.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return "unknown"


def _args_from_call(obj: dict[str, Any]) -> Any:
    if "arguments" in obj:
        return obj["arguments"]
    if "args" in obj:
        return obj["args"]
    if "input" in obj:
        return obj["input"]
    if "parameters" in obj:
        return obj["parameters"]
    fn = obj.get("function")
    if isinstance(fn, dict) and "arguments" in fn:
        return fn["arguments"]
    return None


def _call_id_of(obj: dict[str, Any]) -> str | None:
    for key in ("tool_call_id", "call_id", "id", "toolCallId"):
        val = obj.get(key)
        if val is not None and val != "":
            return str(val)
    return None


def _result_content(obj: dict[str, Any]) -> Any:
    for key in ("content", "result", "output", "response", "text", "body"):
        if key in obj:
            return obj[key]
    return None


def parse_events(records: Iterator[tuple[int, dict[str, Any]]]) -> tuple[list[ToolCall], list[ToolResult]]:
    """Parse flexible JSONL into ordered tool calls and results."""
    calls: list[ToolCall] = []
    results: list[ToolResult] = []
    # Map call_id -> tool name for correlating results
    id_to_name: dict[str, str] = {}
    pending_by_order: list[str] = []  # tool names for results without id

    for index, obj in records:
        role = str(obj.get("role", "")).lower()
        etype = str(obj.get("type", "")).lower()

        # OpenAI-style assistant with tool_calls
        if role == "assistant" or etype in {"assistant", "message"}:
            tcalls = obj.get("tool_calls") or obj.get("toolCalls")
            if isinstance(tcalls, list):
                for tc in tcalls:
                    if not isinstance(tc, dict):
                        continue
                    name = _name_from_call(tc)
                    args = _stringify_args(_args_from_call(tc))
                    cid = _call_id_of(tc)
                    calls.append(ToolCall(index=index, name=name, arguments=args, call_id=cid))
                    if cid:
                        id_to_name[cid] = name
                    pending_by_order.append(name)
            # Some traces put a single function_call on the assistant
            if "function_call" in obj and isinstance(obj["function_call"], dict):
                fc = obj["function_call"]
                name = _name_from_call(fc)
                args = _stringify_args(_args_from_call(fc))
                cid = _call_id_of(fc)
                calls.append(ToolCall(index=index, name=name, arguments=args, call_id=cid))
                if cid:
                    id_to_name[cid] = name
                pending_by_order.append(name)

        # Flat tool_call event
        if etype in {"tool_call", "toolcall", "function_call", "functioncall"}:
            name = _name_from_call(obj)
            args = _stringify_args(_args_from_call(obj))
            cid = _call_id_of(obj)
            calls.append(ToolCall(index=index, name=name, arguments=args, call_id=cid))
            if cid:
                id_to_name[cid] = name
            pending_by_order.append(name)

        # Tool result: role=tool or type=tool_result
        is_result = role == "tool" or etype in {
            "tool_result",
            "toolresult",
            "function_result",
            "functionresult",
            "tool_response",
            "toolresponse",
        }
        if is_result:
            content = _as_text(_result_content(obj))
            cid = None
            for key in ("tool_call_id", "call_id", "toolCallId"):
                if obj.get(key):
                    cid = str(obj[key])
                    break
            name = "unknown"
            if cid and cid in id_to_name:
                name = id_to_name[cid]
            elif obj.get("name") or obj.get("tool") or obj.get("tool_name"):
                name = _name_from_call(obj)
            elif pending_by_order:
                name = pending_by_order.pop(0)
            results.append(ToolResult(index=index, name=name, content=content, call_id=cid))

    return calls, results


def find_bleeds(
    calls: list[ToolCall],
    results: list[ToolResult],
    *,
    min_len: int = 32,
    allow_same_tool: bool = False,
) -> tuple[list[BleedHit], int]:
    """Find result→argument bleeds. Returns (hits, candidate_count)."""
    # Build candidate pool from each result, tagged with result metadata
    pool: list[tuple[ToolResult, str]] = []
    for res in results:
        for cand in extract_candidates(res.content, min_len=min_len):
            pool.append((res, cand))

    hits: list[BleedHit] = []
    # Avoid duplicate (result_index, call_index, snippet) reports
    seen_hits: set[tuple[int, int, str]] = set()

    for call in calls:
        for res, cand in pool:
            # Only later tool calls (by record index); same-turn result→call impossible
            if call.index <= res.index:
                continue
            if allow_same_tool and call.name == res.name:
                continue
            if cand not in call.arguments:
                continue
            key = (res.index, call.index, cand)
            if key in seen_hits:
                continue
            # Prefer reporting the longest match; if a shorter cand is substring
            # of an already-reported longer match for same pair, skip it.
            subsumed = False
            for prev_ri, prev_ci, prev_snip in seen_hits:
                if prev_ri == res.index and prev_ci == call.index and cand in prev_snip and cand != prev_snip:
                    subsumed = True
                    break
            if subsumed:
                continue
            seen_hits.add(key)
            hits.append(
                BleedHit(
                    result_index=res.index,
                    result_tool=res.name,
                    call_index=call.index,
                    call_tool=call.name,
                    snippet=_snippet(cand),
                    candidate_len=len(cand),
                )
            )

    # Sort by result then call order
    hits.sort(key=lambda h: (h.result_index, h.call_index, -h.candidate_len))
    return hits, len(pool)


def analyze_text(
    text: str,
    *,
    min_len: int = 32,
    allow_same_tool: bool = False,
    source: str = "<stdin>",
) -> Report:
    records = list(_iter_jsonl_records(None, text))
    calls, results = parse_events(iter(records))
    hits, n_cand = find_bleeds(calls, results, min_len=min_len, allow_same_tool=allow_same_tool)
    for h in hits:
        h.source = source
    return Report(
        hits=hits,
        records=len(records),
        tool_calls=len(calls),
        tool_results=len(results),
        candidates=n_cand,
        source=source,
    )


def analyze_path(
    path: Path,
    *,
    min_len: int = 32,
    allow_same_tool: bool = False,
) -> Report:
    return analyze_text(
        path.read_text(encoding="utf-8", errors="replace"),
        min_len=min_len,
        allow_same_tool=allow_same_tool,
        source=str(path),
    )


def format_report(report: Report) -> str:
    lines: list[str] = []
    src = report.source or "?"
    lines.append(f"resultbleed: {src}")
    lines.append(
        f"  records={report.records} tool_calls={report.tool_calls} "
        f"tool_results={report.tool_results} candidates={report.candidates} "
        f"bleeds={report.hit_count}"
    )
    if not report.hits:
        lines.append("  clean: no result→argument bleed found")
        return "\n".join(lines)
    for i, h in enumerate(report.hits, 1):
        lines.append(
            f"  [{i}] BLEED file={h.source or src} "
            f"result_line={h.result_index + 1} ({h.result_tool}) → "
            f"call_line={h.call_index + 1} ({h.call_tool}) "
            f"len={h.candidate_len}"
        )
        lines.append(f"      snippet: {h.snippet!r}")
    return "\n".join(lines)
