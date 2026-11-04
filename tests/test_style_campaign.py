"""Bounded selection, immutable pools, split provenance and publication tests."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
import prepare_style_campaign as campaign
import prepare_gait_references


class StyleCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        model = self.root / "model.xml"
        model.write_text("<model/>", encoding="utf-8")
        widths = {"g1": 4, "a3": 5, "t1": 3, "bumi": 2, "pi_plus": 6, "go2": 3}
        self.specs = {
            robot: SimpleNamespace(
                name=robot, joint_names=tuple(f"joint_{i}" for i in range(width)),
                num_joints=width, mjcf_path=str(model), default_q=(0.,) * width)
            for robot, width in widths.items()
        }
        for target in ("prepare_style_campaign.get_robot", "prepare_gait_references.get_robot"):
            patcher = patch(target, side_effect=lambda robot: self.specs[robot])
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(campaign, "_require_ignored_output")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.gaits, self.parkour, self.output = (
            self.root / "gaits", self.root / "parkour", self.root / "output")
        self.counter = 0

    def motion(self, robot="g1", frames=700, bump=None):
        self.counter += 1
        values = np.zeros((frames, 7 + self.specs[robot].num_joints))
        values[:, 0] = np.arange(frames) * .01 + self.counter * .11
        values[:, 2] = .7 + .01 * np.sin(np.arange(frames) * .03)
        values[:, 6] = 1.
        values[:, 7:] = .3 + np.arange(frames)[:, None] * .00001
        if bump is not None:
            begin, end, height = bump
            values[begin:end, 2] += height * np.sin(np.linspace(0, np.pi, end - begin))
        return values

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def create_gait(self, robot):
        directory = self.gaits / robot
        directory.mkdir(parents=True, exist_ok=True)
        records = {}
        for split in ("train", "test"):
            name = f"{split}_recording.csv"
            values = self.motion(robot, 30)
            np.savetxt(directory / name, values, delimiter=",")
            records[name] = {
                "sha256": campaign.digest(directory / name), "frames": len(values),
                "physical_foot_floor_distance_scale_m": {"min": .003, "max": .01},
                "source": {"dataset": "synthetic_gait_fixture", "recording": split},
            }
            self.write(directory / f"{split}.json", campaign.motion_config(robot, [name]))
        self.write(directory / "gait_audit.json", {
            "robot": robot, "status": "passed",
            "model_sha256": campaign.digest(self.specs[robot].mjcf_path),
            "training_files": ["train_recording.csv"], "evaluation_files": ["test_recording.csv"],
            "records": records,
        })

    def create_style(self, directory=None, style="parkour", groups=None, repeats=2):
        directory = directory or self.parkour
        directory.mkdir(parents=True, exist_ok=True)
        groups = groups or (["walk1", "walk2", "walk4", "run1", "jumps1", "obstacles2"]
                            if style == "parkour" else ["aiming1", "aiming2"])
        records = []
        for group in groups:
            for index in range(repeats):
                values = self.motion(bump=(480, 660, .5) if group.startswith("jumps") else None)
                name = f"{group}__window_{index}.csv"
                np.savetxt(directory / name, values, delimiter=",")
                split = "test" if group in ("jumps1", "walk4", "aiming1", "obstacles1") else "train"
                subject = 2 if group == "aiming2" else 1
                source = f"{group}_subject{subject}.csv"
                row = {
                    "file": name, "sha256": campaign.digest(directory / name), "frames": len(values),
                    "fps": 50, "source_file": source, "source_sha256": campaign._identity(source),
                    "source_group": group, "source_fps": 30,
                    "source_window_frames": [index * 600, index * 600 + 420],
                    "split": split,
                }
                if style == "stealth":
                    bvh = source.replace(".csv", ".bvh")
                    row.update(origin={
                        "source_file": bvh, "source_sha256": campaign.ORIGINAL_AIMING_SHA256[bvh]},
                        upper_body_handling={"source_posture_retained": True})
                records.append(row)
        for split in ("train", "test"):
            self.write(directory / f"{split}.json", campaign.motion_config(
                "g1", [row["file"] for row in records if row["split"] == split]))
        manifest = {
            "status": "ready", "style": style, "robot": "g1", "output_fps": 50,
            "quaternion_order": "xyzw", "joint_names": list(self.specs["g1"].joint_names),
            "clips": records, "dataset": "synthetic_fixture", "license": "local-only fixture",
        }
        if style == "stealth":
            manifest["reviewed_official_source"] = {
                "revision": campaign.OFFICIAL_REVISION,
                "archive_sha256": campaign.OFFICIAL_ARCHIVE_SHA256,
            }
        self.write(directory / "manifest.json", manifest)
        return directory

    def prepare(self, robots=("g1",), **kwargs):
        return campaign.prepare_campaign(
            self.gaits, self.parkour, self.output, robots=robots, **kwargs)

    def create_go2_expanded(self):
        directory = self.root / "go2-expanded"
        directory.mkdir()
        records = []
        for name, kind, split in (
            ("a.csv", "jump", "train"), ("b.csv", "turn", "train"),
            ("c.csv", "jump", "test"), ("d.csv", "run", "test"),
        ):
            values = self.motion("go2", frames=100, bump=(20, 80, .3) if kind == "jump" else None)
            np.savetxt(directory / name, values, delimiter=",")
            records.append({
                "file": name, "clip": f"source_take_{name[0]}", "split": split, "clip_type": kind,
                "sha256": campaign.digest(directory / name), "frames": len(values), "fps": 50,
                "raw_sha256": campaign._identity(["raw", name]),
                "source_pose_sha256": campaign._identity(["pose", name]),
                "source_frames": 120, "source_fps": 60,
                "raw_fk": {"frames_checked": 120, "foot_link_max_error_m": 1e-6},
            })
        configs = {}
        for split in ("train", "test"):
            self.write(directory / f"{split}.json", campaign.motion_config(
                "go2", [row["file"] for row in records if row["split"] == split]))
            configs[split] = {"file": f"{split}.json", "sha256": campaign.digest(directory / f"{split}.json")}
        self.write(directory / "manifest.json", {
            "schema": "go2-style-expanded-v1", "version": 1, "status": "ready", "robot": "go2",
            "source_fps": 60, "output_fps": 50, "quaternion_order": "xyzw",
            "joint_names": list(self.specs["go2"].joint_names),
            "mjcf_sha256": campaign.digest(self.specs["go2"].mjcf_path),
            "files": records, "training_files": [row["file"] for row in records if row["split"] == "train"],
            "evaluation_files": [row["file"] for row in records if row["split"] == "test"], "configs": configs,
        })
        return directory

    def test_crop_favors_jump_vertical_range_and_keeps_exact_samples(self):
        values = self.motion(frames=1000, bump=(700, 950, 1.))
        begin, end = campaign.crop_window(values, 400, "jumps")
        self.assertEqual(end - begin, 400)
        self.assertGreater(begin, 300)
        self.assertGreater(np.ptp(values[begin:end, 2]), .99)
        np.testing.assert_array_equal(values[begin:end], values[np.arange(begin, end)])

    def test_bounded_diverse_plan_reassigns_old_jumps_holdout(self):
        self.create_style(repeats=3)
        candidates, _ = campaign.load_style(self.parkour, "parkour")
        selected, report = campaign.select_bounded(candidates)
        self.assertLessEqual(sum(len(clip.selected) for clip in selected), 4000)
        self.assertTrue(all(len(clip.selected) <= 400 for clip in selected))
        self.assertTrue(all(count <= 2 for count in campaign.Counter(
            clip.original_id for clip in selected).values()))
        self.assertEqual(set(report["training_families"]), set(campaign.TRAIN_FAMILIES))
        self.assertEqual(set(report["held_out_groups"]), {"walk2", "walk4"})
        jumps = [clip for clip in selected if clip.group == "jumps1"]
        self.assertTrue(jumps)
        self.assertTrue(all(clip.split == "train" and clip.record["split"] == "test" for clip in jumps))
        with self.assertRaisesRegex(ValueError, "jumps1 must"):
            campaign.load_style(self.parkour, "parkour", ("jumps1",))

    def test_g1_configs_provenance_and_gaits_match_real_loader_schema(self):
        self.create_gait("g1")
        self.create_style()
        original = (self.parkour / "manifest.json").read_bytes()
        with patch.object(campaign, "retarget") as retarget:
            mapping = self.prepare()
        retarget.assert_not_called()
        self.assertEqual(original, (self.parkour / "manifest.json").read_bytes())
        self.assertEqual(mapping["version"], 1)
        self.assertIsNone(mapping["robots"]["g1"]["stealth"])
        from supervise_style import resolve_styles
        resolved = resolve_styles(self.output / "style-map.json")
        self.assertEqual(resolved["g1"]["walk"], (self.output / mapping["robots"]["g1"]["walk"]).resolve())
        self.assertNotIn("stealth", resolved["g1"])
        pool = (self.output / mapping["robots"]["g1"]["parkour"]).parent
        manifest = campaign.verify_pool(pool, "g1", "parkour")
        for row in manifest["files"]:
            descriptor = row["source"]
            source = np.loadtxt(self.parkour / descriptor["prepared_file"], delimiter=",")
            begin, end = descriptor["prepared_frame_range"]
            actual = np.loadtxt(pool / row["file"], delimiter=",")
            np.testing.assert_array_equal(actual, source[begin:end])
        walk = (self.output / mapping["robots"]["g1"]["walk"]).parent
        for row in campaign.verify_pool(walk, "g1", "walk")["files"]:
            self.assertEqual((walk / row["file"]).read_bytes(), Path(row["source_file"]).read_bytes())
        self.assertFalse((self.output / ".prepare.lock").exists())

    def test_incremental_go2_preserves_g1_and_never_calls_human_retarget(self):
        self.create_gait("g1")
        self.create_gait("go2")
        self.create_style()
        original = self.prepare()
        with patch.object(campaign, "retarget") as retarget:
            expanded = self.prepare(("go2",))
        retarget.assert_not_called()
        self.assertEqual(original["robots"]["g1"], expanded["robots"]["g1"])
        self.assertIn("gait dynamics only", " ".join(expanded["limitations"]["go2"]))
        self.assertIsNone(expanded["robots"]["go2"]["stealth"])

    def test_failure_during_retarget_keeps_published_map_bytes(self):
        self.create_gait("g1")
        self.create_gait("a3")
        self.create_style()
        self.prepare()
        before = (self.output / "style-map.json").read_bytes()
        with patch.object(campaign, "retarget", side_effect=RuntimeError("IK failed")):
            with self.assertRaisesRegex(RuntimeError, "IK failed"):
                self.prepare(("a3",))
        self.assertEqual(before, (self.output / "style-map.json").read_bytes())
        self.assertFalse((self.output / ".prepare.lock").exists())

    def test_retarget_uses_robot_width_and_constant_floor_without_arm_reset(self):
        self.create_gait("a3")
        self.create_style()
        inputs = []
        def fake_retarget(values, robot):
            inputs.append(values.copy())
            self.assertEqual(robot, "a3")
            output = np.c_[values, np.full(len(values), .77)]
            return output, {"solver": "test Mink", "source": "obsolete donor description"}
        with patch.object(campaign, "retarget", side_effect=fake_retarget):
            with patch.object(campaign, "physical_floor_distances",
                              side_effect=lambda values, robot: np.linspace(-.02, .8, len(values))):
                mapping = self.prepare(("a3",))
        directory = (self.output / mapping["robots"]["a3"]["parkour"]).parent
        manifest = campaign.verify_pool(directory, "a3", "parkour")
        self.assertEqual(len(inputs), len(manifest["files"]))
        for source, record in zip(inputs, manifest["files"]):
            target = np.loadtxt(directory / record["file"], delimiter=",")
            np.testing.assert_allclose(target[:, 2] - source[:, 2], .023)
            np.testing.assert_array_equal(target[:, 7:-1], source[:, 7:])
            self.assertTrue(record["floor_alignment"]["airborne_trajectory_preserved"])
            self.assertFalse(record["floor_alignment"]["per_frame_clamp"])
            self.assertEqual(target.shape[1], 7 + self.specs["a3"].num_joints)
            self.assertNotIn("obsolete", record["retarget"]["source"])

    def test_stealth_retains_aiming_split_and_arms_and_survives_later_partial_run(self):
        self.create_gait("g1")
        self.create_style()
        stealth = self.create_style(self.root / "stealth", style="stealth")
        mapping = self.prepare(g1_stealth=stealth)
        directory = (self.output / mapping["robots"]["g1"]["stealth"]).parent
        manifest = campaign.verify_pool(directory, "g1", "stealth")
        self.assertEqual(manifest["selection"]["missing_training_families"], [])
        for row in manifest["files"]:
            self.assertEqual(row["split"], "test" if row["source_group"] == "aiming1" else "train")
            source = np.loadtxt(stealth / row["source"]["prepared_file"], delimiter=",")
            a, b = row["source"]["prepared_frame_range"]
            actual = np.loadtxt(directory / row["file"], delimiter=",")
            np.testing.assert_array_equal(actual[:, 7:], source[a:b, 7:])
        later = self.prepare()
        self.assertEqual(mapping["robots"]["g1"]["stealth"], later["robots"]["g1"]["stealth"])

    def test_tampered_source_and_audit_refuse_publication(self):
        self.create_gait("g1")
        self.create_style()
        path = self.parkour / "jumps1__window_0.csv"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.prepare()
        self.assertFalse((self.output / "style-map.json").exists())
        self.create_style()
        audit = self.gaits / "g1/gait_audit.json"
        value = json.loads(audit.read_text())
        value["status"] = "failed"
        self.write(audit, value)
        with self.assertRaisesRegex(ValueError, "audit"):
            self.prepare()
        self.assertFalse((self.output / "style-map.json").exists())

    def test_resuming_identical_recipe_verifies_outputs_and_avoids_new_ik(self):
        self.create_gait("g1")
        self.create_style()
        first = self.prepare()
        before = (self.output / "style-map.json").read_bytes()
        self.prepare()
        self.assertEqual(before, (self.output / "style-map.json").read_bytes())
        directory = (self.output / first["robots"]["g1"]["parkour"]).parent
        manifest = json.loads((directory / "manifest.json").read_text())
        path = directory / manifest["files"][0]["file"]
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.prepare()
        self.assertEqual(before, (self.output / "style-map.json").read_bytes())

    def test_optional_go2_config_is_native_and_cannot_reuse_heldout_bytes(self):
        self.create_gait("go2")
        extra = self.root / "extra"
        extra.mkdir()
        values = self.motion("go2", 1000)
        np.savetxt(extra / "native.csv", values, delimiter=",")
        self.write(extra / "native.json", campaign.motion_config("go2", ["native.csv"]))
        with patch.object(campaign, "retarget") as retarget:
            mapping = self.prepare(("go2",), go2_parkour_config=extra / "native.json")
        retarget.assert_not_called()
        directory = (self.output / mapping["robots"]["go2"]["parkour"]).parent
        manifest = campaign.verify_pool(directory, "go2", "parkour")
        self.assertEqual(manifest["additional_native_go2_clips"], 1)
        self.assertFalse(manifest["verified_jump_reference"])
        bad = copy.deepcopy(campaign.motion_config("go2", ["test_recording.csv"]))
        self.write(self.gaits / "go2/extra.json", bad)
        before = (self.output / "style-map.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "crosses train/test"):
            self.prepare(("go2",), go2_parkour_config=self.gaits / "go2/extra.json")
        self.assertEqual(before, (self.output / "style-map.json").read_bytes())

    def test_lock_and_invalid_bounds_fail_before_retargering(self):
        self.output.mkdir()
        lock = self.output / ".prepare.lock"
        lock.write_text("another process")
        with self.assertRaisesRegex(RuntimeError, "Another campaign"):
            self.prepare()
        self.assertEqual(lock.read_text(), "another process")
        for options in ({"clip_frames": 401}, {"max_per_source": 3}, {"frame_budget": 5000}):
            with self.assertRaisesRegex(ValueError, "Bounds"):
                self.prepare(**options)

    def test_too_small_budget_cannot_drop_a_required_family(self):
        self.create_style()
        candidates, _ = campaign.load_style(self.parkour, "parkour")
        with self.assertRaisesRegex(ValueError, "cannot cover"):
            campaign.select_bounded(candidates, frame_budget=400)

    def test_atomic_map_replace_failure_leaves_last_complete_map(self):
        self.create_gait("g1")
        self.create_gait("go2")
        self.create_style()
        self.prepare()
        previous = (self.output / "style-map.json").read_bytes()
        with patch.object(campaign.os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                self.prepare(("go2",))
        self.assertEqual(previous, (self.output / "style-map.json").read_bytes())
        self.assertFalse(list(self.output.glob(".style-map.json-*.tmp")))

    def test_campaign_refuses_unverified_stealth_posture_or_fps(self):
        self.create_gait("g1")
        self.create_style()
        stealth = self.create_style(self.root / "stealth", style="stealth")
        path = stealth / "manifest.json"
        original = json.loads(path.read_text())
        bad = copy.deepcopy(original)
        bad["clips"][0]["upper_body_handling"]["source_posture_retained"] = False
        self.write(path, bad)
        with self.assertRaisesRegex(ValueError, "arm posture"):
            self.prepare(g1_stealth=stealth)
        self.assertFalse((self.output / "style-map.json").exists())
        self.write(path, original)
        bad = copy.deepcopy(original)
        aiming1 = next(row for row in bad["clips"] if row["source_group"] == "aiming1")
        aiming1["split"] = "train"
        self.write(path, bad)
        with self.assertRaisesRegex(ValueError, "membership"):
            self.prepare(g1_stealth=stealth)
        self.write(path, original)
        config = stealth / "train.json"
        bad_config = json.loads(config.read_text())
        bad_config["source_fps"] = 30
        self.write(config, bad_config)
        with self.assertRaisesRegex(ValueError, "frame rate"):
            self.prepare(g1_stealth=stealth)

    def test_prepared_motion_identity_cannot_cross_splits_by_renaming(self):
        self.create_style()
        path = self.parkour / "manifest.json"
        document = json.loads(path.read_text())
        walk = next(row for row in document["clips"] if row["source_group"] == "walk1")
        test = next(row for row in document["clips"] if row["source_group"] == "walk2")
        test["source_sha256"] = walk["source_sha256"]
        self.write(path, document)
        with self.assertRaisesRegex(ValueError, "identity crosses"):
            campaign.load_style(self.parkour, "parkour")

    def test_additional_obstacles_holdouts_apply_to_union_and_keep_bounds(self):
        self.create_gait("g1")
        self.create_style(groups=["walk1", "walk2", "walk4", "run1", "jumps1"])
        obstacles = self.create_style(
            self.root / "obstacles", groups=["obstacles1", "obstacles2", "obstacles3"])
        holdout = ("walk2", "walk4", "obstacles1")
        with self.assertRaisesRegex(ValueError, "holdout groups are absent"):
            campaign.load_style(self.parkour, "parkour", holdout)
        partial, _ = campaign.load_style(
            self.parkour, "parkour", holdout, require_all_holdouts=False)
        self.assertTrue(partial)
        mapping = self.prepare(parkour_additional=(obstacles,), holdout=holdout)
        directory = (self.output / mapping["robots"]["g1"]["parkour"]).parent
        manifest = campaign.verify_pool(directory, "g1", "parkour")
        report = manifest["selection"]
        self.assertEqual(set(report["training_families"]), set(campaign.TRAIN_FAMILIES))
        self.assertEqual(set(report["held_out_groups"]), set(holdout))
        self.assertLessEqual(sum(row["frames"] for row in manifest["files"]), 4000)
        self.assertTrue(all(row["frames"] <= 400 for row in manifest["files"]))
        self.assertEqual(len(report["source_union"]["source_manifests"]), 2)
        for row in manifest["files"]:
            expected = "test" if row["source_group"] in holdout else "train"
            self.assertEqual(row["split"], expected)
        obstacles_rows = [row for row in manifest["files"]
                          if row["source_group"].startswith("obstacles")]
        self.assertTrue(any(row["split"] == "train" for row in obstacles_rows))
        self.assertTrue(any(row["source_group"] == "obstacles1" for row in obstacles_rows))
        self.assertTrue(all(
            row["upstream_manifest"]["source_manifest"] == str(obstacles / "manifest.json")
            for row in obstacles_rows))

    def test_combined_pool_dedup_preserves_first_source_windows(self):
        self.create_style()
        obstacles = self.create_style(
            self.root / "obstacles", groups=["obstacles1", "obstacles2", "obstacles3"])
        clips, receipt = campaign.load_parkour_sources(
            self.parkour, (obstacles, self.parkour), ("walk2", "walk4", "obstacles1"))
        original = [clip for clip in clips if clip.group == "obstacles2"]
        self.assertEqual(len(original), 2)
        self.assertTrue(all(clip.path.parent == self.parkour for clip in original))
        self.assertEqual(len(receipt["source_manifests"]), 2)
        self.assertEqual(receipt["source_manifests"][1]["duplicate_source_clips"], 2)
        selected, report = campaign.select_bounded(clips)
        self.assertTrue(all(count <= 2 for count in campaign.Counter(
            clip.original_id for clip in selected).values()))
        self.assertEqual(report["missing_training_families"], [])

    def test_combined_missing_holdout_or_cross_pool_hash_conflict_keeps_map(self):
        self.create_gait("g1")
        self.create_style()
        self.prepare()
        previous = (self.output / "style-map.json").read_bytes()
        obstacles = self.create_style(
            self.root / "obstacles", groups=["obstacles1", "obstacles2"])
        with self.assertRaisesRegex(ValueError, "combined prepared inputs"):
            self.prepare(parkour_additional=(obstacles,),
                         holdout=("walk2", "walk4", "obstacles6"))
        self.assertEqual(previous, (self.output / "style-map.json").read_bytes())
        primary = json.loads((self.parkour / "manifest.json").read_text())
        manifest_path = obstacles / "manifest.json"
        extra = json.loads(manifest_path.read_text())
        extra["clips"][0]["source_sha256"] = primary["clips"][0]["source_sha256"]
        self.write(manifest_path, extra)
        with self.assertRaisesRegex(ValueError, "conflicting groups"):
            self.prepare(parkour_additional=(obstacles,), holdout=("walk2", "walk4", "obstacles1"))
        self.assertEqual(previous, (self.output / "style-map.json").read_bytes())

    def test_go2_paired_verified_manifest_preserves_types_hashes_and_heldout_jumps(self):
        self.create_gait("go2")
        expanded = self.create_go2_expanded()
        with patch.object(campaign, "retarget") as retarget:
            mapping = self.prepare(
                ("go2",), go2_parkour_config=expanded / "train.json",
                go2_parkour_test_config=expanded / "test.json")
        retarget.assert_not_called()
        directory = (self.output / mapping["robots"]["go2"]["parkour"]).parent
        manifest = campaign.verify_pool(directory, "go2", "parkour")
        self.assertTrue(manifest["verified_jump_reference"])
        extras = [row for row in manifest["files"] if "source_verification" in row]
        self.assertEqual(len(extras), 4)
        self.assertEqual({row["split"] for row in extras if row["clip_type"] == "jump"}, {"train", "test"})
        for row in extras:
            self.assertTrue(row["source_verification"]["every_frame_fk"])
            self.assertEqual(row["raw_sha256"], row["upstream_record"]["raw_sha256"])
            self.assertEqual(row["source_pose_sha256"], row["upstream_record"]["source_pose_sha256"])
            self.assertEqual(row["split"], row["upstream_record"]["split"])
            np.testing.assert_array_equal(
                np.loadtxt(directory / row["file"], delimiter=","),
                np.loadtxt(row["source_file"], delimiter=","))

    def test_go2_train_only_or_generic_filename_never_certifies_both_jump_splits(self):
        self.create_gait("go2")
        expanded = self.create_go2_expanded()
        mapping = self.prepare(("go2",), go2_parkour_config=expanded / "train.json")
        directory = (self.output / mapping["robots"]["go2"]["parkour"]).parent
        self.assertFalse(campaign.verify_pool(directory, "go2", "parkour")["verified_jump_reference"])
        generic = self.root / "generic"
        generic.mkdir()
        for split in ("train", "test"):
            name = f"obvious_jump_{split}.csv"
            np.savetxt(generic / name, self.motion("go2", 80), delimiter=",")
            self.write(generic / f"{split}.json", campaign.motion_config("go2", [name]))
        mapping = self.prepare(
            ("go2",), go2_parkour_config=generic / "train.json",
            go2_parkour_test_config=generic / "test.json")
        directory = (self.output / mapping["robots"]["go2"]["parkour"]).parent
        manifest = campaign.verify_pool(directory, "go2", "parkour")
        self.assertFalse(manifest["verified_jump_reference"])
        self.assertTrue(all(not row["source_verification"]["verified"]
                            for row in manifest["files"] if "source_verification" in row))

    def test_go2_manifest_rejects_model_fps_fk_config_and_csv_corruption(self):
        expanded = self.create_go2_expanded()
        path = expanded / "manifest.json"
        original = json.loads(path.read_text())
        mutations = [
            ("model", lambda m: m.update(mjcf_sha256="f" * 64)),
            ("FPS", lambda m: m.update(output_fps=30)),
            ("every-frame FK", lambda m: m["files"][0]["raw_fk"].update(frames_checked=119)),
            ("every-frame FK", lambda m: m["files"][0]["raw_fk"].update(foot_link_max_error_m=.001)),
            ("raw_sha256", lambda m: m["files"][0].pop("raw_sha256")),
            ("config hash", lambda m: m["configs"]["train"].update(sha256="f" * 64)),
        ]
        for message, mutation in mutations:
            with self.subTest(message=message):
                document = copy.deepcopy(original)
                mutation(document)
                self.write(path, document)
                with self.assertRaisesRegex(ValueError, message):
                    campaign._go2_extra(expanded / "train.json", 400, 4000)
        self.write(path, original)
        csv = expanded / "a.csv"
        csv.write_text(csv.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Motion hash mismatch"):
            campaign._go2_extra(expanded / "train.json", 400, 4000)

    def test_go2_pair_rejects_shared_raw_pose_or_group_and_keeps_map(self):
        self.create_gait("go2")
        expanded = self.create_go2_expanded()
        self.prepare(("go2",))
        previous = (self.output / "style-map.json").read_bytes()
        manifest_path = expanded / "manifest.json"
        original = json.loads(manifest_path.read_text())
        for field in ("raw_sha256", "source_pose_sha256", "clip"):
            with self.subTest(field=field):
                document = copy.deepcopy(original)
                document["files"][2][field] = document["files"][0][field]
                self.write(manifest_path, document)
                with self.assertRaisesRegex(ValueError, "crosses train/test"):
                    self.prepare(
                        ("go2",), go2_parkour_config=expanded / "train.json",
                        go2_parkour_test_config=expanded / "test.json")
                self.assertEqual(previous, (self.output / "style-map.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "paired train"):
            self.prepare(("go2",), go2_parkour_test_config=expanded / "test.json")

    def test_walk_expansion_is_opt_in_walk_only_bounded_and_preserves_pose_samples(self):
        self.create_gait("g1")
        self.create_style(groups=["walk1", "walk2", "walk4", "run1", "jumps1"])
        obstacles = self.create_style(self.root / "obstacles", groups=["obstacles1", "obstacles2"])
        mapping = self.prepare(
            expand_walk_style=True, parkour_additional=(obstacles,),
            holdout=("walk2", "walk4", "obstacles1"))
        directory = (self.output / mapping["robots"]["g1"]["walk"]).parent
        manifest = campaign.verify_pool(directory, "g1", "walk")
        self.assertTrue(manifest["expanded_walk_style"])
        extras = [row for row in manifest["files"] if "source" in row]
        self.assertTrue(extras)
        self.assertLessEqual(sum(row["frames"] for row in extras), 1600)
        self.assertEqual(manifest["walk_selection"]["missing_training_families"], [])
        self.assertEqual({row["source"]["family"] for row in extras}, {"walk"})
        for row in extras:
            source = np.loadtxt(
                Path(row["upstream_manifest"]["source_manifest"]).parent / row["source"]["prepared_file"],
                delimiter=",")
            begin, end = row["source"]["prepared_frame_range"]
            np.testing.assert_array_equal(np.loadtxt(directory / row["file"], delimiter=","), source[begin:end])
            self.assertEqual(row["split"], "test" if row["source_group"] in ("walk2", "walk4") else "train")
        copied = [row for row in manifest["files"] if "gait_audit_sha256" in row]
        self.assertEqual(len(copied), 2)
        self.assertTrue(all((directory / row["file"]).read_bytes() == Path(row["source_file"]).read_bytes()
                            for row in copied))

    def test_walk_expansion_has_separate_ik_pool_and_identical_resume_avoids_ik(self):
        self.create_gait("a3")
        self.create_style()
        def fake_retarget(values, robot):
            return np.c_[values, np.full(len(values), .77)], {"solver": "test Mink"}
        with patch.object(campaign, "retarget", side_effect=fake_retarget) as retarget:
            with patch.object(campaign, "physical_floor_distances",
                              side_effect=lambda values, robot: np.zeros(len(values))):
                mapping = self.prepare(("a3",), expand_walk_style=True)
        total = 0
        for style in ("walk", "parkour"):
            directory = (self.output / mapping["robots"]["a3"][style]).parent
            manifest = campaign.verify_pool(directory, "a3", style)
            crops = [row for row in manifest["files"] if "source" in row]
            total += len(crops)
            if style == "walk":
                self.assertLessEqual(sum(row["frames"] for row in crops), 1600)
                self.assertTrue(all(row["source"]["family"] == "walk" for row in crops))
        self.assertEqual(retarget.call_count, total)
        with patch.object(campaign, "retarget") as retarget:
            self.prepare(("a3",), expand_walk_style=True)
        retarget.assert_not_called()


if __name__ == "__main__":
    unittest.main()
