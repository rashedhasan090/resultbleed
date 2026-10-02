"""Tests for resultbleed detection (stdlib unittest)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from resultbleed.cli import main
from resultbleed.detect import (
    analyze_text,
    extract_candidates,
    find_bleeds,
    parse_events,
    _iter_jsonl_records,
)


SAMPLE_BLEED = """\
{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\": \\"/vault/key.txt\\"}"}}]}
{"role":"tool","tool_call_id":"c1","content":"api_token=sk-live-NEVER-FORWARD-THIS-SECRET-TOKEN-ABCDEF"}
{"role":"assistant","tool_calls":[{"id":"c2","type":"function","function":{"name":"http_post","arguments":"{\\"body\\": \\"api_token=sk-live-NEVER-FORWARD-THIS-SECRET-TOKEN-ABCDEF\\"}"}}]}
{"role":"tool","tool_call_id":"c2","content":"ok"}
"""

SAMPLE_CLEAN = """\
{"type":"tool_call","id":"a1","name":"get_weather","arguments":{"city":"Lincoln"}}
{"type":"tool_result","tool_call_id":"a1","result":"Temperature is 72F with clear skies over downtown Lincoln Nebraska today."}
{"type":"tool_call","id":"a2","name":"write_note","arguments":{"text":"Weather looks fine for an outdoor walk."}}
{"type":"tool_result","tool_call_id":"a2","result":"saved"}
"""

SAMPLE_SAME_TOOL = """\
{"type":"tool_call","id":"s1","name":"fetch_url","arguments":{"url":"https://example.com/a"}}
{"type":"tool_result","tool_call_id":"s1","result":"UNIQUE-PAYLOAD-STRING-FOR-SAME-TOOL-TEST-12345"}
{"type":"tool_call","id":"s2","name":"fetch_url","arguments":{"url":"https://example.com/b","hint":"UNIQUE-PAYLOAD-STRING-FOR-SAME-TOOL-TEST-12345"}}
{"type":"tool_result","tool_call_id":"s2","result":"ok"}
"""


class ExtractCandidatesTests(unittest.TestCase):
    def test_min_len_and_lines(self):
        text = "short\n" + ("X" * 40) + "\n" + ("Y" * 10)
        cands = extract_candidates(text, min_len=32)
        self.assertTrue(any(len(c) >= 32 for c in cands))
        self.assertTrue(any("X" * 40 in c or c == "X" * 40 for c in cands))
        self.assertFalse(any(c == "short" for c in cands))

    def test_skips_whitespace_and_non_alnum(self):
        text = "   \n" + ("!" * 40) + "\n" + ("ok-token-" + "A" * 40)
        cands = extract_candidates(text, min_len=32)
        self.assertFalse(any(set(c) <= set("! \t") for c in cands))
        self.assertTrue(any("ok-token-" in c for c in cands))


class ParseAndDetectTests(unittest.TestCase):
    def test_openai_bleed(self):
        report = analyze_text(SAMPLE_BLEED, min_len=32)
        self.assertEqual(report.tool_calls, 2)
        self.assertEqual(report.tool_results, 2)
        self.assertGreaterEqual(report.hit_count, 1)
        hit = report.hits[0]
        self.assertEqual(hit.result_tool, "read_file")
        self.assertEqual(hit.call_tool, "http_post")
        self.assertIn("sk-live-NEVER-FORWARD", hit.snippet)

    def test_flat_clean(self):
        report = analyze_text(SAMPLE_CLEAN, min_len=32)
        self.assertEqual(report.hit_count, 0)

    def test_allow_same_tool(self):
        flagged = analyze_text(SAMPLE_SAME_TOOL, min_len=32, allow_same_tool=False)
        self.assertGreaterEqual(flagged.hit_count, 1)
        skipped = analyze_text(SAMPLE_SAME_TOOL, min_len=32, allow_same_tool=True)
        self.assertEqual(skipped.hit_count, 0)

    def test_order_correlation_without_ids(self):
        text = "\n".join(
            [
                json.dumps({"type": "tool_call", "name": "alpha", "arguments": {"q": 1}}),
                json.dumps(
                    {
                        "type": "tool_result",
                        "result": "ORDER-CORRELATED-SECRET-VALUE-ZZZZZZZZZZ",
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "name": "beta",
                        "arguments": {
                            "x": "ORDER-CORRELATED-SECRET-VALUE-ZZZZZZZZZZ"
                        },
                    }
                ),
            ]
        )
        report = analyze_text(text, min_len=32)
        self.assertGreaterEqual(report.hit_count, 1)
        self.assertEqual(report.hits[0].result_tool, "alpha")
        self.assertEqual(report.hits[0].call_tool, "beta")

    def test_json_array_input(self):
        arr = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "t1",
                        "function": {
                            "name": "secrets_get",
                            "arguments": '{"k":"db"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "t1",
                "content": "password=SuperSecretDatabasePasswordValue99",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "t2",
                        "function": {
                            "name": "run_sql",
                            "arguments": '{"auth":"password=SuperSecretDatabasePasswordValue99"}',
                        },
                    }
                ],
            },
        ]
        report = analyze_text(json.dumps(arr), min_len=32)
        self.assertGreaterEqual(report.hit_count, 1)


class CliTests(unittest.TestCase):
    def test_strict_exit_and_clean(self):
        with tempfile.TemporaryDirectory() as td:
            bleed = Path(td) / "bleed.jsonl"
            clean = Path(td) / "clean.jsonl"
            bleed.write_text(SAMPLE_BLEED, encoding="utf-8")
            clean.write_text(SAMPLE_CLEAN, encoding="utf-8")
            self.assertEqual(main([str(clean)]), 0)
            self.assertEqual(main(["--strict", str(clean)]), 0)
            self.assertEqual(main([str(bleed)]), 0)  # no strict
            self.assertEqual(main(["--strict", str(bleed)]), 1)

    def test_missing_file_exit_2(self):
        self.assertEqual(main(["/nonexistent/path/resultbleed-missing.jsonl"]), 2)

    def test_json_output(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "b.jsonl"
            path.write_text(SAMPLE_BLEED, encoding="utf-8")
            # Capture via analyze path indirectly: cli prints JSON
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(["--json", str(path)])
            self.assertEqual(code, 0)
            data = json.loads(buf.getvalue())
            self.assertGreaterEqual(data["bleed_count"], 1)


class ExamplesOnDiskTests(unittest.TestCase):
    def test_repo_examples(self):
        root = Path(__file__).resolve().parents[1]
        sample = root / "examples" / "sample.jsonl"
        clean = root / "examples" / "clean.jsonl"
        if sample.is_file():
            self.assertGreaterEqual(main([str(sample)]), 0)
            self.assertEqual(main(["--strict", str(sample)]), 1)
        if clean.is_file():
            self.assertEqual(main(["--strict", str(clean)]), 0)


if __name__ == "__main__":
    unittest.main()
