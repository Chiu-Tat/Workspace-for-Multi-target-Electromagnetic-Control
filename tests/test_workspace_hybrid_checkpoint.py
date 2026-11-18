import json
import os
import tempfile
import unittest
from collections import Counter

from emns.numerics.interval import interval
from emns.workspace.checks import Create3DBox

from cases.case_6_interaction_obstacle_avoidance.workspace_hybrid import (
    EPS,
    _load_checkpoint,
    _save_checkpoint,
)


class WorkspaceHybridCheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_search_state(self):
        pending = [
            Create3DBox(
                interval[-0.05, 0.0],
                interval[0.0, 0.05],
                interval[0.0, 0.1],
            )
        ]
        inside = [
            Create3DBox(
                interval[0.0, 0.005],
                interval[0.01, 0.015],
                interval[1.0, 1.1],
            )
        ]
        reasons = Counter({"rohn": 1})

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.json")
            _save_checkpoint(
                path,
                pending=pending,
                inside=inside,
                outside=[],
                undetermined=[],
                iteration=32,
                reasons=reasons,
                elapsed_s=12.5,
                eps=EPS,
            )
            restored = _load_checkpoint(path, EPS)

            self.assertEqual(restored["iteration"], 32)
            self.assertEqual(restored["elapsed_s"], 12.5)
            self.assertEqual(restored["reasons"], reasons)
            self.assertEqual(len(restored["pending"]), 1)
            self.assertEqual(len(restored["inside"]), 1)
            self.assertEqual(
                restored["pending"][0][0][0], interval[-0.05, 0.0]
            )
            self.assertFalse(os.path.exists(path + ".tmp"))

            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["version"], 1)

    def test_checkpoint_rejects_incompatible_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.json")
            _save_checkpoint(
                path,
                pending=[],
                inside=[],
                outside=[],
                undetermined=[],
                iteration=0,
                reasons=Counter(),
                elapsed_s=0.0,
                eps=EPS,
            )

            with self.assertRaisesRegex(ValueError, "precision"):
                _load_checkpoint(path, [0.01, 0.01, 0.2])

    def test_checkpoint_rejects_other_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.json")
            _save_checkpoint(
                path,
                pending=[],
                inside=[],
                outside=[],
                undetermined=[],
                iteration=0,
                reasons=Counter(),
                elapsed_s=0.0,
                eps=EPS,
                layout="gate",
            )

            self.assertEqual(_load_checkpoint(path, EPS, "gate")["iteration"], 0)
            with self.assertRaisesRegex(ValueError, "protocol"):
                _load_checkpoint(path, EPS, "triangle")


if __name__ == "__main__":
    unittest.main()
