from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from prepare_command_style import (
    ANOMALOUS_SOURCE_SHA256, is_anomalous_parkour_source,
    is_anomalous_walk_source, moving_segments,
)


class CommandStylePreparationTests(unittest.TestCase):
    def test_excludes_prefixed_t1_and_bumi_source_names_only(self):
        row = {
            "file": "gait_002__clip_002__walk1__00091-00491.csv",
            "source_file": "/data/t1/clip_002__walk1__00091-00491.csv",
        }
        self.assertTrue(is_anomalous_walk_source("t1", row))
        self.assertTrue(is_anomalous_walk_source("bumi", row))
        self.assertFalse(is_anomalous_walk_source("g1", row))
        self.assertFalse(is_anomalous_walk_source("t1", {"file": "clip_002__walk2.csv"}))

    def test_parkour_filter_matches_original_clip_identity_not_other_walk1_windows(self):
        row = {"source": {"source_file": "walk1_subject1.csv",
                          "source_sha256": ANOMALOUS_SOURCE_SHA256,
                          "prepared_frame_range": [91, 491]}}
        self.assertTrue(is_anomalous_parkour_source("t1", row))
        self.assertTrue(is_anomalous_parkour_source("bumi", row))
        self.assertFalse(is_anomalous_parkour_source("g1", row))
        other = {"source": row["source"] | {"prepared_frame_range": [491, 891]}}
        self.assertFalse(is_anomalous_parkour_source("t1", other))

    def test_continuous_moving_segments_obey_minimum_history(self):
        import numpy as np
        from scipy.spatial.transform import Rotation

        values = np.zeros((24, 7), dtype=np.float64)
        values[:, 3:7] = Rotation.identity().as_quat()
        values[:, 0] = np.arange(24) * .01
        expected = moving_segments(values, minimum_speed=.15, history=10)
        self.assertTrue(expected)
        self.assertTrue(all(end - begin >= 10 for begin, end in expected))


if __name__ == "__main__":
    unittest.main()
