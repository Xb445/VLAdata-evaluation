"""Regression checks for masking, provenance and paired comparisons."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from vla_eval.audit import _raw_timing, assignments
from vla_eval.evaluate import _attempts, action_errors, parse_simulation
from vla_eval.io import digest, write_jsonl
from vla_eval.report import _paired_interval, _policy_results, wilson


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases = {
            "environment": "simulation",
            "split": "test",
            "cases": [
                {
                    "case_id": f"c{i}",
                    "task_id": "place",
                    "cluster_id": f"cluster-{i}",
                    "scenario_id": f"pose-{i}",
                    "seed": i,
                    "suite": "pusht",
                    "task_index": 0,
                }
                for i in (10, 11)
            ],
        }
        self.plan = {"cases": self.cases, "recipe": {"seeds": [1, 2]}}
        self.record = {"case_id": "c10", "scenario_id": "pose-10", "case_seed": 10, "attempt": 1}

    def test_padding_is_excluded(self):
        dims = [{"min": -1, "max": 1, "max_step": 0.5}] * 2
        metrics = action_errors([[1, 1], [999, 999]], [[0, 0], [0, 0]], [False, True], [1, 2], dims)
        self.assertEqual(metrics["valid_steps"], 1)
        self.assertEqual(metrics["scaled_mae"], 0.75)
        self.assertEqual(metrics["valid_transitions"], 0)
        self.assertEqual(metrics["outside_bounds_steps"]["prediction"], 0)
        with self.assertRaisesRegex(ValueError, "horizon"):
            action_errors([[1]], [[0], [0]], [False, True], [1], dims)

    def test_chunk_jump_is_compared_with_the_label(self):
        dims = [{"min": -1, "max": 1, "max_step": 0.5}]
        metrics = action_errors([[0], [2], [0]], [[0], [0.2], [0.4]], [False] * 3, [1], dims)
        self.assertEqual(metrics["valid_transitions"], 2)
        self.assertEqual(metrics["chunk_jump_steps"], {"prediction": 2, "label": 0})
        self.assertEqual(metrics["outside_bounds_steps"], {"prediction": 1, "label": 0})

    def test_source_group_cannot_cross_train_and_dev(self):
        rows = [
            {
                "episode_index": i,
                "source_group": "same-capture",
                "split": split,
                "task_id": "place",
                "coverage_cell": "cell",
                "action_verified": True,
                "language_verified": True,
            }
            for i, split in enumerate(("train", "dev"))
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                assignments(
                    path,
                    SimpleNamespace(episodes={0: {}, 1: {}}),
                    {"coverage": {"cell": {"task_id": "place"}}},
                )

    def test_unknown_clock_and_noncausal_clock_are_distinct(self):
        rules = {
            "require_raw_timing": False,
            "timing": {"age_ms": [1, 2], "camera_skew_ms": [1, 2], "command_delay_ms": [1, 2]},
        }
        problems = []

        def issue(*args):
            problems.append(args[:2])

        q, _ = _raw_timing({}, [{}], 0, ["camera"], rules, issue)
        self.assertIsNone(q)
        self.assertEqual(problems[-1], ("RAW_TIMING_MISSING", "review"))
        row = {
            "decision_ns": 100,
            "state_sample_ns": 95,
            "state_receive_ns": 97,
            "command_issue_ns": 101,
            "command_apply_ns": 102,
            "image_capture_ns": {"camera": 101},
            "image_receive_ns": {"camera": 103},
        }
        q, _ = _raw_timing({(0, 0): row}, [{}], 0, ["camera"], rules, issue)
        self.assertIsNone(q)
        self.assertEqual(problems[-1], ("RAW_TIMING_INVALID", "block"))

    def test_infrastructure_attempt_requires_completed_replacement(self):
        rows = [{**self.record, "outcome": "infra_invalid", "reason": "camera disconnected"}]
        bundle = _attempts({}, self.plan, rows, {})
        p, details, _ = _policy_results("baseline", self.plan, {("baseline", 1): bundle}, {"place": 1})
        self.assertIsNone(p)
        self.assertEqual(details["1"]["tasks"]["place"]["n"], 0)
        self.assertEqual(details["1"]["pending"], ["c10", "c11"])
        rows.append({**self.record, "outcome": "success", "attempt": 2})
        rows.append({**self.record, "outcome": "failure", "attempt": 3})
        with self.assertRaisesRegex(ValueError, "cannot be replaced"):
            _attempts({}, self.plan, rows, {})

    def test_native_eval_uses_case_order_and_task_identity(self):
        info = {
            "per_task": [
                {
                    "task_group": "pusht",
                    "task_id": 0,
                    "metrics": {"successes": [True, False], "sum_rewards": [0.8, 0.1]},
                }
            ]
        }
        bundle = parse_simulation(info, self.plan, {})
        self.assertEqual(bundle["cases_sha"], digest(self.cases))
        self.assertEqual(
            [(r["case_seed"], r["outcome"]) for r in bundle["attempts"]], [(10, "success"), (11, "failure")]
        )
        info["per_task"][0]["metrics"]["successes"].pop()
        with self.assertRaisesRegex(ValueError, "case count"):
            parse_simulation(info, self.plan, {})

    def test_effect_interval_uses_matched_seeds_and_cases(self):
        before = {s: {"c10": False, "c11": False} for s in (1, 2)}
        after = {s: {"c10": True, "c11": True} for s in (1, 2)}
        self.assertEqual(_paired_interval(before, after, self.cases, {"place": 1}), [100, 100])
        low, high = wilson(16, 20)
        self.assertAlmostEqual(low, 58.4, places=1)
        self.assertAlmostEqual(high, 91.9, places=1)


if __name__ == "__main__":
    unittest.main()
