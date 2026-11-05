from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))

from prepare_lafan_stealth_windows import WINDOWS
from prepare_single_stealth import (
    TEST_ORIGINAL,
    TRAIN_ORIGINAL,
    TRAIN_WINDOW,
    source_range_receipt,
    validate_single_pair,
    validate_train_window,
)
from prepare_style_references import ORIGINAL_AIMING_SHA256


def clip(split, group, original, source_hash):
    return SimpleNamespace(
        split=split,
        group=group,
        record={
            "source_sha256": source_hash,
            "source_window_frames": [340, 460] if split == "train" else [3420, 3720],
            "origin": {
                "source_file": original,
                "source_sha256": ORIGINAL_AIMING_SHA256[original],
            },
        },
        values=[0] * 100,
        begin=0,
        end=100,
    )


class SingleStealthTests(unittest.TestCase):
    def test_training_window_is_the_pinned_forward_only_range(self):
        self.assertEqual(TRAIN_WINDOW, ("aiming2_subject3.bvh", 340, 460, "forward"))
        self.assertIn(TRAIN_WINDOW, WINDOWS)
        self.assertEqual(validate_train_window(), TRAIN_WINDOW)

    def test_altered_or_nonforward_training_window_is_rejected(self):
        for window in (
            ("aiming2_subject3.bvh", 340, 460, "backward"),
            ("aiming2_subject3.bvh", 341, 460, "forward"),
            ("aiming1_subject1.bvh", 340, 460, "forward"),
        ):
            with self.subTest(window=window), self.assertRaises(ValueError):
                validate_train_window(window)

    def test_pair_requires_exactly_one_disjoint_original_take_per_split(self):
        train = clip("train", "aiming2", TRAIN_ORIGINAL, "a" * 64)
        test = clip("test", "aiming1", TEST_ORIGINAL, "b" * 64)
        self.assertEqual(validate_single_pair((train, test)), (train, test))
        for candidate in (
            (train,),
            (train, test, test),
            (clip("train", "aiming1", TRAIN_ORIGINAL, "a" * 64), test),
            (train, clip("test", "aiming2", TEST_ORIGINAL, "b" * 64)),
            (train, clip("test", "aiming1", TEST_ORIGINAL, "a" * 64)),
        ):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_single_pair(candidate)

    def test_cropped_candidate_is_rejected(self):
        train = clip("train", "aiming2", TRAIN_ORIGINAL, "a" * 64)
        test = clip("test", "aiming1", TEST_ORIGINAL, "b" * 64)
        train.begin = 1
        with self.assertRaisesRegex(ValueError, "complete prepared intervals"):
            validate_single_pair((train, test))

    def test_prepared_frames_map_to_fractional_original_bvh_positions(self):
        mapped = source_range_receipt(
            {
                "source_file": TRAIN_ORIGINAL,
                "source_sha256": ORIGINAL_AIMING_SHA256[TRAIN_ORIGINAL],
                "source_frame_range": [340, 460],
                "output_frames": 199,
                "source_fps": 30.0,
                "output_fps": 50.0,
            },
            {"source_window_frames": [28, 134]},
        )
        self.assertEqual(mapped["retarget_source_frame_range"], [340, 460])
        self.assertEqual(mapped["retargeted_csv_frame_range"], [28, 134])
        self.assertEqual(mapped["original_sample_frame_positions"], [356.8, 419.8])


if __name__ == "__main__":
    unittest.main()
