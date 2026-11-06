"""Assetless immutable-pool checks using the real campaign builder/verifier."""

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
import prepare_stealth_entry as entry
import prepare_style_campaign as campaign
import prepare_gait_references as gait
from trackerTask.supermimic.robots import get_robot


class PrepareStealthEntryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stealth-entry-pool-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        model = self.root / "model.xml"
        model.write_text("<model/>", encoding="utf-8")
        g1 = get_robot("g1")
        self.spec = SimpleNamespace(
            joint_names=g1.joint_names, num_joints=g1.num_joints, mjcf_path=str(model))
        for module in (entry, campaign, gait):
            patcher = patch.object(module, "get_robot", return_value=self.spec)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(entry, "_require_ignored_output")
        self.ignored = patcher.start()
        self.addCleanup(patcher.stop)
        self.arm_names = [name for name in g1.joint_names
                          if any(part in name for part in ("shoulder", "elbow", "wrist"))]
        self.reference = {name: .0125 * (index + 1) for index, name in enumerate(self.arm_names)}
        self.reference_path = self.root / "reference.json"
        self.reference_path.write_text(json.dumps(self.reference), encoding="utf-8")
        self.source_bytes = {}

        def writer(directory):
            records = []
            for split, group, offset, frames in (("train", "aiming2", .1, 12), ("test", "aiming1", .7, 15)):
                name = f"{group}.csv"
                lines = [b"# synthetic numeric fixture\r\n", b"\n"]
                for frame in range(frames):
                    values = [offset + frame * .01, -0., .7, 0., 0., 0., 1.]
                    values += [offset + frame * .001] * g1.num_joints
                    fields = [format(value, ".17g").encode() for value in values]
                    fields[1] = b" -0.000000e+00"
                    fields[2] = b"7.000000000000000e-01 "
                    ending = b"" if frame == frames - 1 else b"\r\n" if frame % 2 else b"\n"
                    lines.append(b",".join(fields) + ending)
                self.source_bytes[name] = b"".join(lines)
                (directory / name).write_bytes(self.source_bytes[name])
                records.append({
                    "file": name, "sha256": entry._sha(self.source_bytes[name]),
                    "frames": frames, "fps": 50, "split": split,
                    "source_group": group, "original_sha256": entry._sha(group.encode()),
                    "upstream_record": {"origin": {"source_file": f"{group}.bvh"}},
                })
            return records, {"selection": {"training_clips": 1, "held_out_clips": 1}}

        self.source, _ = campaign._build_pool(
            self.root / "original", "g1", "stealth", {"fixture": "single-source"}, writer)
        self.before = {path: path.read_bytes() for path in self.source.iterdir()}
        self.provenance = "Synthetic local static-arm correction; content-integrity test only."

    def prepare(self, name="derived", reference=None):
        return entry.prepare(self.source, reference or self.reference_path,
                             self.root / name, self.provenance)

    def test_only_named_training_arm_fields_change_and_holdout_is_byte_identical(self):
        reference_before = self.reference_path.read_bytes()
        pool = self.prepare()
        verified = campaign.verify_pool(pool, "g1", "stealth")
        self.assertEqual((pool / "aiming1.csv").read_bytes(), self.source_bytes["aiming1.csv"])
        original = self.source_bytes["aiming2.csv"].splitlines(keepends=True)
        corrected = (pool / "aiming2.csv").read_bytes().splitlines(keepends=True)
        columns = [7 + list(self.spec.joint_names).index(name) for name in self.arm_names]
        self.assertEqual(len(original), len(corrected))
        for before, after in zip(original, corrected):
            left, right = before.rstrip(b"\r\n"), after.rstrip(b"\r\n")
            self.assertEqual(before[len(left):], after[len(right):])
            if not left.strip() or left.lstrip().startswith(b"#"):
                self.assertEqual(before, after)
                continue
            for index, (a, b) in enumerate(zip(left.split(b","), right.split(b","))):
                if index in columns:
                    self.assertEqual(float(b), self.reference[self.spec.joint_names[index - 7]])
                else:
                    self.assertEqual(a, b)
        values = np.loadtxt(pool / "aiming2.csv", delimiter=",")
        np.testing.assert_array_equal(values[:, columns], np.tile(list(self.reference.values()), (12, 1)))
        train = next(row for row in verified["files"] if row["split"] == "train")
        self.assertFalse(train["local_correction"]["raw_lafan_motion"])
        self.assertFalse(train["local_correction"]["source_upper_body_posture_retained"])
        self.assertEqual(train["local_correction"]["provenance"], self.provenance)
        self.assertIn("not raw LAFAN", verified["source"])
        old_test = next(row for row in campaign.verify_pool(self.source, "g1", "stealth")["files"]
                        if row["split"] == "test")
        self.assertEqual(next(row for row in verified["files"] if row["split"] == "test"), old_test)
        self.assertEqual({path: path.read_bytes() for path in self.source.iterdir()}, self.before)
        self.assertEqual(self.reference_path.read_bytes(), reference_before)
        self.ignored.assert_called_once_with((self.root / "derived").resolve())

    def test_wrapped_reference_preserves_entry_profile_and_evidence_document(self):
        document = {"reference_q": self.reference, "source": {"evidence_sha256": "a" * 64},
                    "entry_times_s": [0., 2.],
                    "entry_reference_q": [{name: 0. for name in self.arm_names}, self.reference]}
        self.reference_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        pool = self.prepare()
        self.assertEqual((pool / "upper-body-reference.json").read_bytes(), self.reference_path.read_bytes())
        manifest = campaign.verify_pool(pool, "g1", "stealth")
        self.assertEqual(manifest["auxiliary_files"]["reference"]["sha256"], entry._sha(self.reference_path.read_bytes()))
        self.assertEqual(manifest["recipe"]["correction"]["reference_q"], self.reference)

    def test_entry_profile_maps_named_waypoints_and_hashes_both_inputs(self):
        names = self.arm_names[::-1]
        profile = {"joint_names": names, "waypoint_times_s": [0., 1.5, 4.82],
                   "waypoint_q_rad": [[0.] * 14, [.1] * 14, [self.reference[name] for name in names]]}
        profile_path = self.root / "entry-parameters.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        pool = entry.prepare(self.source, self.reference_path, self.root / "scheduled",
                             self.provenance, entry_profile=profile_path)
        wrapper = json.loads((pool / "upper-body-reference.json").read_bytes())
        self.assertEqual(wrapper["reference_q"], self.reference)
        self.assertEqual(wrapper["entry_times_s"], [0., 1.5, 4.82])
        self.assertEqual(wrapper["entry_reference_q"][-1], self.reference)
        self.assertEqual(wrapper["entry_reference_q"][1], dict.fromkeys(self.arm_names, .1))
        self.assertEqual(wrapper["source"]["reference_input_sha256"],
                         entry._sha(self.reference_path.read_bytes()))
        self.assertEqual(wrapper["source"]["entry_profile_sha256"], entry._sha(profile_path.read_bytes()))
        self.assertEqual((pool / "entry-profile.json").read_bytes(), profile_path.read_bytes())
        self.assertEqual((pool / "aiming1.csv").read_bytes(), self.source_bytes["aiming1.csv"])
        campaign.verify_pool(pool, "g1", "stealth")

    def test_invalid_entry_profiles_and_conflicting_schedules_fail_before_output_creation(self):
        valid = {"joint_names": self.arm_names, "waypoint_times_s": [0., 4.82],
                 "waypoint_q_rad": [[0.] * 14, list(self.reference.values())]}
        invalid = [[], {}, {**valid, "joint_names": [None] * 14},
                   {**valid, "joint_names": [self.arm_names[0]] * 14},
                   {**valid, "waypoint_times_s": [0., 0.]},
                   {**valid, "waypoint_times_s": [1., 4.82]},
                   {**valid, "waypoint_times_s": [0., float("nan")]},
                   {**valid, "waypoint_q_rad": [[0.] * 14, [0.] * 14]},
                   {**valid, "waypoint_q_rad": [[0.] * 13, list(self.reference.values())]}]
        profile_path = self.root / "entry.json"
        for index, profile in enumerate(invalid):
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            with self.subTest(index=index), self.assertRaises(ValueError):
                entry.prepare(self.source, self.reference_path, self.root / f"bad-entry-{index}",
                              self.provenance, entry_profile=profile_path)
            self.assertFalse((self.root / f"bad-entry-{index}").exists())
        profile_path.write_text(json.dumps(valid), encoding="utf-8")
        self.reference_path.write_text(json.dumps(
            {"reference_q": self.reference, "entry_times_s": [0., 2.]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from the entry profile"):
            entry.prepare(self.source, self.reference_path, self.root / "conflict",
                          self.provenance, entry_profile=profile_path)

    def test_relocated_inputs_and_fresh_outputs_reproduce_pool_identity_and_bytes(self):
        first = self.prepare("first")
        relocated = self.root / "copied-source"
        relocated.mkdir()
        for path, payload in self.before.items():
            (relocated / path.name).write_bytes(payload)
        reference = self.root / "renamed-reference.json"
        reference.write_bytes(self.reference_path.read_bytes())
        second = entry.prepare(relocated, reference, self.root / "second", self.provenance)
        self.assertEqual(first.name, second.name)
        self.assertEqual({path.name: path.read_bytes() for path in first.iterdir()},
                         {path.name: path.read_bytes() for path in second.iterdir()})
        self.assertEqual((self.root / "first/preparation.json").read_bytes(),
                         (self.root / "second/preparation.json").read_bytes())

    def test_changed_pose_or_provenance_creates_distinct_identity(self):
        first = self.prepare("first")
        self.reference["left_wrist_yaw_joint"] += .1
        self.reference_path.write_text(json.dumps(self.reference), encoding="utf-8")
        second = self.prepare("second")
        third = entry.prepare(self.source, self.reference_path, self.root / "third", "Different correction evidence")
        self.assertEqual(len({first.name, second.name, third.name}), 3)
        for pool in (first, second, third):
            self.assertEqual((pool / "aiming1.csv").read_bytes(), self.source_bytes["aiming1.csv"])

    def test_existing_nested_or_unignored_output_is_rejected_without_writes(self):
        existing = self.root / "existing"
        existing.mkdir()
        sentinel = existing / "keep.txt"
        sentinel.write_bytes(b"retain")
        with self.assertRaises(FileExistsError):
            entry.prepare(self.source, self.reference_path, existing, self.provenance)
        self.assertEqual(sentinel.read_bytes(), b"retain")
        with self.assertRaisesRegex(ValueError, "separate trees"):
            entry.prepare(self.source, self.reference_path, self.source / "nested", self.provenance)
        self.ignored.side_effect = campaign._require_ignored_output
        with self.assertRaisesRegex(ValueError, "ignored repository data"):
            self.prepare()
        self.assertFalse((self.root / "derived").exists())

    def test_invalid_named_reference_is_rejected_before_output_creation(self):
        cases = [[], {}, {**self.reference, "left_hip_pitch_joint": 0.},
                 {name: value for name, value in self.reference.items() if name != self.arm_names[0]}]
        for value in (True, "0.1", float("nan"), float("inf"), 1e50):
            cases.append({**self.reference, self.arm_names[0]: value})
        for index, value in enumerate(cases):
            self.reference_path.write_text(json.dumps(value), encoding="utf-8")
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.prepare(f"invalid-{index}")
            self.assertFalse((self.root / f"invalid-{index}").exists())
        with self.assertRaisesRegex(ValueError, "provenance"):
            entry.prepare(self.source, self.reference_path, self.root / "empty", " ")

    def test_tampered_source_and_wrong_split_are_rejected(self):
        path = self.source / "aiming2.csv"
        path.write_bytes(path.read_bytes().replace(b"7.000", b"6.000"))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.prepare()
        path.write_bytes(self.before[path])
        manifest_path = self.source / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["files"][0]["source_group"] = "walk1"
        manifest_path.write_bytes(campaign._json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "Aiming2"):
            self.prepare()
        self.assertFalse((self.root / "derived").exists())

    def test_cli_requires_all_inputs_and_forwards_explicit_paths(self):
        with patch.object(entry, "prepare", return_value=self.source) as prepare, \
                patch("builtins.print"):
            entry.main(["--source-pool", str(self.source), "--reference", str(self.reference_path),
                        "--output", str(self.root / "cli"), "--provenance", self.provenance])
        prepare.assert_called_once_with(self.source, self.reference_path, self.root / "cli", self.provenance)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            entry.main([])


if __name__ == "__main__":
    unittest.main()
