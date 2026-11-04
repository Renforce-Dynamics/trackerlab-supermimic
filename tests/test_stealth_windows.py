from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from prepare_lafan_stealth_windows import WINDOWS, validate_windows
from prepare_style_references import STYLE_POLICIES


class StealthWindowTests(unittest.TestCase):
    def test_windows_are_ordered_disjoint_and_training_side(self):
        self.assertEqual(validate_windows(), WINDOWS)
        by_source = {}
        for name, start, end, direction in WINDOWS:
            self.assertTrue(name.startswith("aiming2_"))
            self.assertIn(direction, ("forward", "backward", "left", "right"))
            prior_end = by_source.get(name, 0)
            self.assertGreaterEqual(start, prior_end)
            self.assertGreaterEqual(end - start, 100)
            by_source[name] = end
        self.assertEqual(len(WINDOWS), 4)
        self.assertEqual(set(by_source), {"aiming2_subject3.bvh", "aiming2_subject5.bvh"})
        self.assertEqual(STYLE_POLICIES["stealth"]["maximum_clearance_m"], .70)

    def test_test_take_and_overlapping_source_windows_are_rejected(self):
        for candidate in (
            (("aiming1_subject1.bvh", 100, 220),),
            (("aiming2_subject3.bvh", 100, 220),
             ("aiming2_subject3.bvh", 200, 320)),
            (("aiming2_subject3.bvh", 100, 199),),
        ):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_windows(candidate)


if __name__ == "__main__":
    unittest.main()
