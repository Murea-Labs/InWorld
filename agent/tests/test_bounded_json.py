"""Bound runaway JSON whitespace while preserving formatting, text and schemas."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from murea_agent import config
from murea_agent.guards import density_audit
from murea_agent.director import agent_schema
from murea_agent.providers import llm
from murea_agent.guards import motion


class RequestTest(unittest.TestCase):
    def test_opt_in_only_changes_the_output_grammar(self):
        schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        args = ("system", [], "input", "local", schema, 0.9, 0, [("image/jpeg", b"frame")])
        with patch.object(config, "OPENAI_JSON_MAX_WHITESPACE", 0):
            original = llm._openai_body(*args, stream=True)
        with (
            patch.object(config, "OPENAI_JSON_MAX_WHITESPACE", 32),
            patch.object(llm, "_bounded_json_grammar", return_value="bounded") as compiler,
        ):
            bounded = llm._openai_body(*args, stream=True)
        original.pop("response_format")
        self.assertEqual("bounded", bounded.pop("ebnf"))
        self.assertEqual(original, bounded)
        self.assertEqual((json.dumps(schema, ensure_ascii=False), 32), compiler.call_args.args)
        self.assertTrue(llm._log_body({**bounded, "ebnf": "bounded"})["schema"])


@unittest.skipUnless(importlib.util.find_spec("xgrammar"), "optional SGLang compiler absent")
class GrammarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import xgrammar as xgr

        cls.xgr = xgr
        cls.compiler = xgr.GrammarCompiler(
            xgr.TokenizerInfo([bytes([i]) for i in range(256)], stop_token_ids=[])
        )

    def accepts(self, schema, text):
        grammar = llm._bounded_json_grammar(json.dumps(schema, ensure_ascii=False), 32)
        matcher = self.xgr.GrammarMatcher(
            self.compiler.compile_grammar(grammar), terminate_without_stop_token=True
        )
        return matcher.accept_string(text) and matcher.is_terminated()

    def test_normal_indentation_and_arbitrary_string_whitespace_survive(self):
        schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        value = {"text": '中文台词 "quote"\n' + " " * 256 + "\\path"}
        self.assertTrue(self.accepts(schema, json.dumps(value, ensure_ascii=False, indent=4)))
        self.assertTrue(self.accepts(schema, json.dumps(value, ensure_ascii=False, separators=(",", ":"))))
        self.assertFalse(self.accepts(schema, '{"text": "ok"' + " " * 33 + "}"))
        self.assertFalse(self.accepts(schema, "{}"))

    def test_nested_stall_from_real_failure_is_rejected(self):
        schema = {
            "type": "object",
            "properties": {
                "update": {
                    "type": "object",
                    "properties": {"progress": {"type": "string"}, "place": {"type": "string"}},
                    "required": ["progress", "place"],
                }
            },
            "required": ["update"],
        }
        self.assertTrue(self.accepts(schema, '{"update":{"progress":"完成", "place":"通风管后"}}'))
        self.assertFalse(
            self.accepts(schema, '{"update":{"progress":"完成"' + " " * 4000 + ',"place":"通风管后"}}')
        )

    def test_initial_physical_contract_rejects_the_reported_invalid_plan(self):
        schema = density_audit._motion_replan_schema(
            {
                "type": "object",
                "properties": {"motion_plan": agent_schema.MOTION_PLAN_SCHEMA},
                "required": ["motion_plan"],
            }
        )
        inactive = copy.deepcopy(motion._EMPTY_MOTION_PLAN)
        self.assertTrue(self.accepts(schema, json.dumps({"motion_plan": inactive})))
        inactive["phase"] = "settle"
        self.assertFalse(self.accepts(schema, json.dumps({"motion_plan": inactive})))
        incomplete = copy.deepcopy(inactive)
        incomplete.update(interaction_type="locomotion", phase="approach", relation_change="distance")
        self.assertFalse(self.accepts(schema, json.dumps({"motion_plan": incomplete})))


if __name__ == "__main__":
    unittest.main(verbosity=2)
