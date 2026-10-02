# resultbleed

Offline, stdlib-only Python CLI that scans agent tool-call / conversation JSONL
transcripts for **result→argument bleed**: substrings from a prior tool
*result* that reappear in a later tool *call's* arguments.

This flags accidental secret/PII forwarding and prompt-injection payload
chaining across tools (different tool name, or same tool on a later turn).

## Why this is novel

Many agent-trace checkers look at failures, retries, orphaned calls, n-gram
leakage into the *assistant* text, or citation freshness. **resultbleed**
targets a different failure mode: a tool result's sensitive contiguous text
quietly showing up inside the *arguments* of a subsequent tool call.

Distinct from sandclock, tokpack, hushdiff, runseal, toolflow, hedgescope,
diffintent, promptfence, aegispath, rippleguard, shardroom, claimcite,
stubtruth, callstorm, toolorphan, seedguard, ngramleak, refage, failthrough,
and citationcheckertool.

## Install

```bash
pip install -e .
# or run without install:
python -m resultbleed examples/sample.jsonl
```

Requires Python 3.10+. No third-party dependencies.

## Usage

```bash
resultbleed TRACE.jsonl [MORE.jsonl...]
resultbleed --strict --json TRACE.jsonl   # CI gate + machine output
resultbleed --min-len 40 TRACE.jsonl
resultbleed --allow-same-tool TRACE.jsonl # skip same-name later calls
cat TRACE.jsonl | resultbleed             # stdin
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Clean, or bleeds found but `--strict` not set |
| 1 | `--strict` and one or more bleeds found |
| 2 | Usage / I/O / parse error |

### Options

- `--min-len N` — minimum candidate substring length (default: **32**)
- `--strict` — exit 1 when any bleed is found
- `--json` — machine-readable JSON report
- `--allow-same-tool` — do not flag when the later call has the same tool name

## What it accepts

Pragmatic JSONL shapes:

- OpenAI-ish messages with `role` / `tool_calls` / `role=tool` results
- Flat events with `type` / `name` / `arguments` / `result` / `content`

Tool calls and results are correlated by `tool_call_id` when present, else by
order. Candidates prefer whole lines, long tokens, and quoted strings of length
`>= --min-len`; whitespace-only and non-alnum junk are skipped.

## Demo

```bash
python -m resultbleed examples/sample.jsonl
```

`examples/sample.jsonl` includes a **bleed** (API key from `read_file` forwarded
into `http_request` body) and later **clean** turns. `examples/clean.jsonl` is
fully clean.

Example human report:

```
resultbleed: examples/sample.jsonl
  records=10 tool_calls=4 tool_results=4 candidates=… bleeds=1
  [1] BLEED file=examples/sample.jsonl result_line=3 (read_file) → call_line=4 (http_request) len=…
      snippet: 'production_api_key=sk-live-NEVER-FORWARD-THIS-SECRET-TOKEN-9f3a2b'
```

## License

MIT © 2026 Md Rashedul Hasan
