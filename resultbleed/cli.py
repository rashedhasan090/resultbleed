"""CLI entrypoint for resultbleed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from resultbleed import __version__
from resultbleed.detect import analyze_path, analyze_text, format_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="resultbleed",
        description=(
            "Scan agent tool-call / conversation JSONL for result→argument "
            "bleed: substrings from a prior tool result that reappear in a "
            "later tool call's arguments (secret/PII forwarding and "
            "prompt-injection payload chaining)."
        ),
    )
    p.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="JSONL (or JSON array) transcript file(s); omit to read stdin",
    )
    p.add_argument(
        "--min-len",
        type=int,
        default=32,
        metavar="N",
        help="minimum candidate substring length (default: 32)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when any bleed is found (CI gate)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text",
    )
    p.add_argument(
        "--allow-same-tool",
        action="store_true",
        help="Skip bleeds where the later call has the same tool name as the result",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _hit_dict(h) -> dict:
    return {
        "source": h.source,
        "result_line": h.result_index + 1,
        "result_tool": h.result_tool,
        "call_line": h.call_index + 1,
        "call_tool": h.call_tool,
        "snippet": h.snippet,
        "candidate_len": h.candidate_len,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_len < 1:
        print("resultbleed: --min-len must be >= 1", file=sys.stderr)
        return 2

    reports = []
    try:
        if not args.paths:
            text = sys.stdin.read()
            reports.append(
                analyze_text(
                    text,
                    min_len=args.min_len,
                    allow_same_tool=args.allow_same_tool,
                    source="<stdin>",
                )
            )
        else:
            for path in args.paths:
                if not path.is_file():
                    print(f"resultbleed: not a file: {path}", file=sys.stderr)
                    return 2
                reports.append(
                    analyze_path(
                        path,
                        min_len=args.min_len,
                        allow_same_tool=args.allow_same_tool,
                    )
                )
    except OSError as exc:
        print(f"resultbleed: I/O error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"resultbleed: {exc}", file=sys.stderr)
        return 2

    total_hits = sum(r.hit_count for r in reports)

    if args.json:
        payload = {
            "files": [
                {
                    "source": r.source,
                    "records": r.records,
                    "tool_calls": r.tool_calls,
                    "tool_results": r.tool_results,
                    "candidates": r.candidates,
                    "bleeds": [_hit_dict(h) for h in r.hits],
                }
                for r in reports
            ],
            "bleed_count": total_hits,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for r in reports:
            print(format_report(r))
            if len(reports) > 1:
                print()
        if len(reports) > 1:
            print(f"resultbleed: total bleeds across {len(reports)} file(s): {total_hits}")

    if args.strict and total_hits:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
