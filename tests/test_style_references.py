"""Contracts for local LAFAN1 style-reference manifests."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

import prepare_style_references as style_refs
from trackerTask.supermimic.robots import get_robot


def fake_g1(frames=360):
    values = np.zeros((frames, 36), dtype=np.float64)
    values[:, 0] = np.arange(frames) / 30 * 0.30
    values[:, 2] = 0.76
    values[:, 6] = 1.0
    return values


def fake_fk(values, robot="g1", fps=30):
    frames = len(values)
    root_height = values[:, 2]
    low_profile = root_height < .70
    hands_x = np.where(low_profile, .20, .05)
    return {
        "speed": np.full(frames, .30),
        "clearance": np.where(low_profile, .60, .73),
        "root_height": root_height,
        "root_vertical_speed": np.zeros(frames),
        "hands_root": np.stack((np.stack((hands_x, np.full(frames, .15), np.zeros(frames)), -1),
                                np.stack((hands_x, np.full(frames, -.15), np.zeros(frames)), -1)), 1),
        "hands_forward_min": hands_x,
        "hands_separation": np.full(frames, .30),
    }


class StyleReferenceTests(unittest.TestCase):
    def write_source(self, directory, name, values=None):
        path = Path(directory) / name
        np.savetxt(path, values if values is not None else fake_g1(), delimiter=",", fmt="%.9g")
        return path

    def origins(self, directory, paths):
        path = Path(directory) / "origins.json"
        rows = [{
            "file": item.name, "sha256": style_refs.digest(item),
            "source_file": item.with_suffix(".bvh").name,
            "source_sha256": style_refs.ORIGINAL_AIMING_SHA256.get(item.with_suffix(".bvh").name, "a" * 64),
            "source_group": style_refs.source_group(item.name),
            "retargeter": "test fixture",
        } for item in paths]
        path.write_text(json.dumps({
            "dataset": "ubisoft/ubisoft-laforge-animation-dataset",
            "revision": style_refs.OFFICIAL_REVISION, "license": "CC BY-NC-ND 4.0",
            "archive_sha256": style_refs.OFFICIAL_ARCHIVE_SHA256,
            "files": rows,
        }), encoding="utf-8")
        return path

    def test_take_group_keeps_subjects_together(self):
        self.assertEqual(style_refs.source_group("walk3_subject1.csv"), "walk3")
        self.assertEqual(style_refs.source_group("walk3_subject5.csv"), "walk3")

    def test_stealth_selection_requires_speed_clearance_and_hands(self):
        values = fake_g1(360)
        values[30:300, 2] = .60
        with patch.object(style_refs, "inspect_kinematics", side_effect=fake_fk):
            intervals, evidence = style_refs.select_intervals(values, 30, "stealth")
        self.assertEqual(intervals, [(30, 300)])
        self.assertEqual(evidence["qualifying_frames"], 270)
        self.assertTrue(evidence["selection_semantics"].startswith("Kinematic"))

    def test_explicit_patterns_generate_local_hashes_and_preserve_upper_body(self):
        with tempfile.TemporaryDirectory(dir=REPO / "data") as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "derived"
            source.mkdir()
            values = fake_g1()
            values[20:340, 2] = .60
            robot = get_robot("g1")
            arms = [index for index, name in enumerate(robot.joint_names)
                    if any(part in name for part in ("shoulder", "elbow", "wrist"))]
            values[:, np.asarray(arms) + 7] = np.arange(len(arms))[None] * .01 + .45
            original = self.write_source(source, "aiming1_subject1.csv", values)
            training = self.write_source(source, "aiming2_subject2.csv", values + 0)
            # Give the second recording distinct bytes, as in an actual take.
            values[:, 0] += .01
            np.savetxt(training, values, delimiter=",", fmt="%.9g")
            origin_manifest = self.origins(root, (original, training))
            raw_bytes = original.read_bytes()
            with patch.object(style_refs, "inspect_kinematics", side_effect=fake_fk):
                manifest = style_refs.prepare_style_references(
                    source, output, ("aiming*.csv",), "stealth",
                    holdout_groups=("aiming1",), minimum_frames=100,
                    origin_manifest=origin_manifest,
                )
            self.assertEqual(original.read_bytes(), raw_bytes)
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(len(manifest["clips"]), 2)
            clip = manifest["clips"][0]
            self.assertEqual(clip["split"], "test")
            self.assertEqual(clip["fps"], 50)
            self.assertEqual(clip["source_fps"], 30)
            self.assertEqual(clip["source_group"], "aiming1")
            self.assertEqual(hashlib.sha256((output / clip["file"]).read_bytes()).hexdigest(),
                             clip["sha256"])
            target = np.loadtxt(output / clip["file"], delimiter=",", ndmin=2)
            self.assertTrue(clip["upper_body_handling"]["source_posture_retained"])
            for joint in arms:
                self.assertTrue(np.allclose(target[:, 7 + joint], values[0, 7 + joint]))
            self.assertEqual(json.loads((output / "test.json").read_text())["files"], [clip["file"]])
            self.assertEqual(manifest["evaluation_files"], [clip["file"]])
            self.assertIn("CC BY-NC-ND", manifest["license"])

    def test_only_explicit_files_are_considered_and_take_split_does_not_leak(self):
        with tempfile.TemporaryDirectory(dir=REPO / "data") as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "derived"
            source.mkdir()
            first = fake_g1()
            first[10:350, 2] = .60
            second = fake_g1()
            second[10:350, 2] = .60
            second[:, 0] += .01
            third = fake_g1()
            third[10:350, 2] = .60
            third[:, 0] += .02
            first_path = self.write_source(source, "aiming1_subject1.csv", first)
            second_path = self.write_source(source, "aiming1_subject4.csv", second)
            third_path = self.write_source(source, "aiming2_subject2.csv", third)
            self.write_source(source, "walk3_subject5.csv", third)
            origins = self.origins(root, (first_path, second_path, third_path))
            with patch.object(style_refs, "inspect_kinematics", side_effect=fake_fk):
                manifest = style_refs.prepare_style_references(
                    source, output,
                    ("aiming1_subject*.csv", "aiming2_subject2.csv"),
                    "stealth", holdout_groups=("aiming1",), minimum_frames=100,
                    origin_manifest=origins,
                )
            split_by_group = {}
            for clip in manifest["clips"]:
                previous = split_by_group.setdefault(clip["source_group"], clip["split"])
                self.assertEqual(previous, clip["split"])
            self.assertEqual(split_by_group["aiming1"], "test")
            self.assertEqual(split_by_group["aiming2"], "train")
            self.assertEqual(len(manifest["clips"]), 3)

    def test_walk_clips_cannot_substitute_for_original_aiming(self):
        with tempfile.TemporaryDirectory(dir=REPO / "data") as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "derived"
            source.mkdir()
            values = fake_g1()
            values[:, 2] = .60
            self.write_source(source, "walk3_subject5.csv", values)
            with patch.object(style_refs, "inspect_kinematics") as inspect:
                manifest = style_refs.prepare_style_references(
                    source, output, ("walk*.csv",), "stealth")
            inspect.assert_not_called()
            self.assertEqual(manifest["status"], "waiting_for_correct_data")
            self.assertFalse((output / "train.json").exists())
            self.assertFalse((output / "test.json").exists())
            self.assertEqual(len(manifest["required_original_aiming_files"]), 5)

    def test_source_config_preserves_fps_and_needs_disjoint_style_recordings(self):
        with tempfile.TemporaryDirectory(dir=REPO / "data") as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "derived"
            source.mkdir()
            self.write_source(source, "walk1_subject1.csv")
            config = style_refs.motion_config("g1", ["walk1_subject1.csv"])
            config["source_fps"] = 50
            config_path = source / "source.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(style_refs, "inspect_kinematics", side_effect=fake_fk) as inspect:
                manifest = style_refs.prepare_style_references(
                    source, output, (), "walk", source_config=config_path)
            self.assertEqual(inspect.call_args.args[2], 50)
            self.assertEqual(manifest["status"], "waiting_for_disjoint_style_split")
            self.assertFalse((output / "train.json").exists())
            self.assertEqual(manifest["clips"][0]["frames"], 360)

    def test_origin_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory(dir=REPO / "data") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            path = self.write_source(source, "aiming1_subject1.csv")
            origins = self.origins(root, (path,))
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Origin CSV hash"):
                style_refs.prepare_style_references(
                    source, root / "out", (path.name,), "stealth",
                    origin_manifest=origins)

    def test_unreviewed_original_bvh_hash_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO / "data") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            path = self.write_source(source, "aiming1_subject1.csv")
            origins = self.origins(root, (path,))
            document = json.loads(origins.read_text())
            document["files"][0]["source_sha256"] = "f" * 64
            origins.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Original Aiming BVH hash"):
                style_refs.prepare_style_references(
                    source, root / "out", (path.name,), "stealth",
                    origin_manifest=origins)

    def test_same_source_bytes_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory(dir=REPO / "data") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            self.write_source(source, "walk1_subject1.csv")
            self.write_source(source, "walk2_subject1.csv")
            with patch.object(style_refs, "inspect_kinematics", side_effect=fake_fk):
                with self.assertRaisesRegex(ValueError, "Identical source bytes"):
                    style_refs.prepare_style_references(
                        source, root / "out", ("walk*.csv",), "walk",
                        holdout_groups=("walk2",))

    def test_audited_gait_bytes_are_copied_only_into_output(self):
        with tempfile.TemporaryDirectory(dir=REPO / "data") as tmp:
            root = Path(tmp)
            gait, output = root / "gait", root / "output"
            gait.mkdir()
            output.mkdir()
            a = self.write_source(gait, "walk_train.csv")
            b_values = fake_g1()
            b_values[:, 0] += .01
            b = self.write_source(gait, "walk_test.csv", b_values)
            (gait / "gait_audit.json").write_text(json.dumps({
                "robot": "g1", "status": "passed",
                "training_files": [a.name], "evaluation_files": [b.name],
                "records": {item.name: {"sha256": style_refs.digest(item)}
                            for item in (a, b)},
            }), encoding="utf-8")
            train, test, records = style_refs._load_gait_refs(gait, "g1", output)
            for row in records:
                self.assertEqual((output / row["file"]).read_bytes(),
                                 Path(row["source_file"]).read_bytes())
                self.assertFalse((root / row["file"]).exists())
            self.assertEqual(len(train), 1)
            self.assertEqual(len(test), 1)

    def test_parkour_keeps_flight_frames_and_uses_no_gait_audit(self):
        values = fake_g1()
        with patch.object(style_refs, "inspect_kinematics", side_effect=fake_fk) as inspect:
            intervals, evidence = style_refs.select_intervals(values, 30, "parkour")
        self.assertEqual(intervals, [(0, len(values))])
        self.assertIsNone(evidence["policy"]["minimum_clearance_m"])
        inspect.assert_called_once()

    def test_constant_floor_shift_preserves_airborne_vertical_profile(self):
        values = fake_g1(6)
        values[:, 2] = np.array([.2, .35, .8, 1.0, .72, .3])
        distances = np.array([.01, .16, .61, .81, .53, .11])
        aligned, receipt = style_refs.align_constant_floor(values, distances, .003)
        self.assertAlmostEqual(receipt["root_z_shift_m"], -.007)
        np.testing.assert_allclose(aligned[:, 2] - values[:, 2], -.007)
        self.assertTrue(receipt["airborne_trajectory_preserved"])
        self.assertFalse(receipt["per_frame_clamp"])
        self.assertAlmostEqual(np.ptp(aligned[:, 2]), np.ptp(values[:, 2]))

    def test_parkour_pipeline_writes_the_floor_aligned_result(self):
        with tempfile.TemporaryDirectory(dir=REPO / "data") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            a_values = fake_g1()
            b_values = fake_g1()
            b_values[:, 0] += .01
            self.write_source(source, "walk1_subject1.csv", a_values)
            self.write_source(source, "walk2_subject1.csv", b_values)
            def floor(values, robot):
                return np.linspace(-.02, .20, len(values))
            with patch.object(style_refs, "inspect_kinematics", side_effect=fake_fk):
                with patch.object(style_refs, "physical_floor_distances", side_effect=floor):
                    manifest = style_refs.prepare_style_references(
                        source, root / "out", ("walk*.csv",), "parkour",
                        holdout_groups=("walk2",))
            self.assertEqual(manifest["status"], "ready")
            for row in manifest["clips"]:
                target = np.loadtxt(root / "out" / row["file"], delimiter=",", ndmin=2)
                np.testing.assert_allclose(target[:, 2], .783)
                self.assertAlmostEqual(row["floor_alignment"]["root_z_shift_m"], .023)
                self.assertFalse(row["floor_alignment"]["per_frame_clamp"])

    def test_unmatched_explicit_pattern_fails_instead_of_silently_omitting_data(self):
        with tempfile.TemporaryDirectory(dir=REPO / "data") as tmp:
            root = Path(tmp)
            self.write_source(root, "walk1_subject1.csv")
            with self.assertRaisesRegex(FileNotFoundError, "matched no files"):
                style_refs._resolve_inputs(root, ("walk1*.csv", "walk2*.csv"))

    def test_go2_is_rejected_and_other_humanoids_need_explicit_opt_in(self):
        with tempfile.TemporaryDirectory(dir=REPO / "data") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            self.write_source(source, "walk3_subject5.csv")
            with self.assertRaisesRegex(ValueError, "quadruped"):
                style_refs.prepare_style_references(
                    source, root / "go2", ("walk3_subject5.csv",), "stealth", robot="go2")
            with self.assertRaisesRegex(ValueError, "explicit"):
                style_refs.prepare_style_references(
                    source, root / "a3", ("walk3_subject5.csv",), "stealth", robot="a3")

    def test_resampling_preserves_duration_and_quaternion_norm(self):
        values = fake_g1(61)
        values[:, 6] = 1
        target = style_refs.resample_g1(values, 30)
        self.assertEqual(len(target), 101)
        self.assertAlmostEqual((len(target) - 1) / 50, 2.)
        np.testing.assert_allclose(np.linalg.norm(target[:, 3:7], axis=1), 1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
