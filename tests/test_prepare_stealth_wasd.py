"""CPU-only immutable-pool integration for explicit locomotion expert roles."""

from io import BytesIO
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
import prepare_stealth_wasd as publisher
import prepare_style_campaign as campaign
import prepare_gait_references as gait


class PrepareStealthWASDTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stealth-wasd-pool-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        model = self.root / "g1.xml"
        model.write_bytes(b"synthetic-g1-model")
        self.names = tuple([f"leg_{i}" for i in range(15)] + [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
            "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ])
        self.robot = SimpleNamespace(name="g1", joint_names=self.names,
                                     num_joints=29, mjcf_path=str(model))
        for module in (publisher, campaign, gait):
            patcher = patch.object(module, "get_robot", return_value=self.robot)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(publisher, "_require_ignored_output")
        self.ignored = patcher.start()
        self.addCleanup(patcher.stop)
        self.source_root = self.root / "raw"
        self.source_root.mkdir()
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.source_hashes = {"aiming2_subject5.bvh": "a"*64,
                              "aiming2_subject3.bvh": "b"*64}
        self.input_files = {}
        for source_name, marker in (("aiming2_subject5.bvh", b"source5"),
                                    ("aiming2_subject3.bvh", b"source3")):
            payload = marker * 8
            (self.source_root / source_name).write_bytes(payload)
            self.source_hashes[source_name] = publisher._sha(payload)
        source_hash_patch = patch.dict(publisher.ORIGINAL_AIMING_SHA256, self.source_hashes)
        source_hash_patch.start()
        self.addCleanup(source_hash_patch.stop)
        self.rows = []
        for index, role in enumerate(("forward", "backward", "idle")):
            source_name = "aiming2_subject5.bvh" if role != "idle" else "aiming2_subject3.bvh"
            begin = index * 100
            csv_name = f"{role}.csv"
            report_name = f"{role}-receipt.json"
            values = np.zeros((101, 36), dtype=np.float64)
            values[:, :3] = np.array([index * 2., 0., .7])
            values[:, 3:7] = [0., 0., 0., 1.]
            values[:, 7:] = .02 * index
            values[:, 7 + 3] = .1
            csv = (",".join(f"{v:.12f}" for v in values[0]) + "\r\n").encode()
            rows = [csv]
            for frame in range(1, 101):
                row = values[frame].copy()
                row[0] += .02 * frame * (1 if role == "forward" else -1 if role == "backward" else 0)
                rows.append((",".join(f"{v:.12f}" for v in row) + ("\r\n" if frame < 100 else "\n")).encode())
            csv_bytes = b"".join(rows)
            (self.prepared / csv_name).write_bytes(csv_bytes)
            self.input_files[csv_name] = csv_bytes
            begin = {"forward": 100, "backward": 400, "idle": 50}[role]
            receipt = {
                "schema": "lafan1-g1-origin-v1",
                "files": [{
                    "source_file": source_name, "source_sha256": self.source_hashes[source_name],
                    "source_frame_range": [begin, begin+61], "output_frames": 101,
                    "output_fps": 50., "source_fps": 30.,
                    "file": csv_name, "sha256": publisher._sha(csv_bytes),
                    "source_group": "aiming2", "split": "train",
                    "joint_names": list(self.names), "target_model_sha256": publisher._sha(
                        (self.root / "g1.xml").read_bytes()),
                    "root_scale": 1., "role": role,
                }],
            }
            report_bytes = json.dumps(receipt).encode()
            (self.prepared / report_name).write_bytes(report_bytes)
            self.input_files[report_name] = report_bytes
            self.rows.append({
                "role": role, "file": csv_name, "sha256": publisher._sha(csv_bytes),
                "frames": 101, "retarget_report": report_name,
                "retarget_report_sha256": publisher._sha(report_bytes),
                "source_file": source_name, "source_sha256": self.source_hashes[source_name],
                "source_frame_range": [begin, begin+61],
            })
        self.prepared_manifest = self.prepared / "prepared-manifest.json"
        self.prepared_manifest.write_text(json.dumps({
            "schema": "stealth-wasd-prepared-v1", "robot": "g1", "fps": 50,
            "quaternion_order": "xyzw", "joint_names": list(self.names),
            "source_scan_sha256": "c"*64, "clips": self.rows,
        }), encoding="utf-8")
        self.heldout = self.root / "heldout"
        self.heldout.mkdir()
        test_bytes = b"\n".join(
            ",".join(map(str, np.r_[i*.01, 0, .7, 0, 0, 0, 1, np.zeros(29)])).encode()
            for i in range(12))
        (self.heldout / "aiming1.csv").write_bytes(test_bytes)
        record = {
            "file": "aiming1.csv", "sha256": publisher._sha(test_bytes), "frames": 12,
            "fps": 50, "duration_seconds": .22, "split": "test", "source_group": "aiming1",
            "original_sha256": "d"*64, "source_file": "aiming1_subject1.csv",
            "upstream_record": {"fixture": "heldout"},
        }
        # _build_pool enforces both splits, so create the real test+train source fixture.
        train_bytes = self.input_files["forward.csv"]
        train_record = {
            **record, "file": "original-forward.csv", "sha256": publisher._sha(train_bytes),
            "frames": 101, "duration_seconds": 2., "split": "train", "source_group": "aiming2",
            "original_sha256": "e"*64, "source_file": "aiming2_subject5.csv",
        }
        def source_writer(directory):
            (directory / "aiming1.csv").write_bytes(test_bytes)
            (directory / "original-forward.csv").write_bytes(train_bytes)
            return [record, train_record], {"fixture": True}
        self.heldout_pool, _ = campaign._build_pool(
            self.root / "heldout-parent", "g1", "stealth", {"fixture": "heldout-v1"}, source_writer)
        self.heldout_bytes = {
            path: path.read_bytes() for path in self.heldout_pool.iterdir() if path.is_file()
        }
        self._patch_source_fk()

    def _patch_source_fk(self):
        metrics = []
        for role in ("forward", "backward", "idle"):
            vx = {"forward": .3, "backward": -.3, "idle": .02}[role]
            metrics.append({
                "method": "source FK fixture",
                "body_vx_m_s_p05_p50_p95": [vx, vx, vx],
                "body_vy_m_s_p05_p50_p95": [0., 0., 0.],
                "planar_speed_m_s_p05_p50_p95": [abs(vx), abs(vx), abs(vx)],
                "yaw_rate_rad_s_p05_p50_p95": [0., 0., 0.],
                "forward_fraction_above_0_08": float(role == "forward"),
                "backward_fraction_below_minus_0_08": float(role == "backward"),
                "low_speed_fraction_below_0_15": float(role == "idle"),
                "low_yaw_fraction_below_0_25": 1.,
                "clearance_proxy_m_p05_p50_p95": [.46,.47,.48],
                "upright_cos_p05_p50_p95": [.95,.98,1.],
                "hands_forward_g1_proxy_m_p05_p50_p95": [.2,.3,.4],
                "foot_xy_speed_proxy_mean_m_s": .003 if role == "idle" else .2,
                "foot_xy_range_proxy_max_m": .02,
                "qualification": "fixture",
            })
        self.metrics = metrics
        self.patcher = patch.object(publisher, "_source_metrics",
                                    side_effect=lambda payload, begin, end, scale:
                                    self.metrics[next(i for i,r in enumerate(self.rows)
                                                     if r["source_frame_range"][0] == begin)])
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def prepare(self, output="output", **kwargs):
        return publisher.prepare(
            self.prepared_manifest, self.source_root, self.heldout_pool,
            self.root / output, "Synthetic source-provenance fixture.", **kwargs)

    def test_equal_role_pool_preserves_source_order_and_heldout_bytes(self):
        pool = self.prepare()
        manifest = campaign.verify_pool(pool, "g1", "stealth")
        train = [row for row in manifest["files"] if row["split"] == "train"]
        self.assertEqual([row["clip_role"] for row in train], ["forward", "backward", "idle"])
        self.assertEqual({row["frames"] for row in train}, {101})
        self.assertEqual(manifest["sampling"]["role_probabilities"],
                         {"forward": 1/3, "backward": 1/3, "idle": 1/3})
        self.assertEqual(manifest["sampling"]["valid_windows_per_role_at_10"], 92)
        test = next(row for row in manifest["files"] if row["split"] == "test")
        self.assertEqual((pool / test["file"]).read_bytes(),
                         (self.heldout_pool / test["file"]).read_bytes())
        for row in train:
            self.assertEqual((pool / row["file"]).read_bytes(), self.input_files[row["file"]])
            self.assertEqual(row["source_frame_range"],
                             next(item for item in self.rows if item["role"] == row["clip_role"])["source_frame_range"])
        receipt = json.loads((pool.parent.parent.parent / "preparation.json").read_text())
        self.assertTrue(receipt["heldout_bytes_identical"])
        self.assertTrue(receipt["source_inputs_unchanged"])

    def test_optional_static_arm_reference_changes_only_named_arm_fields(self):
        arm_names = [name for name in self.names
                     if any(part in name.split("_") for part in ("shoulder", "elbow", "wrist"))]
        reference = {name: .01 * index for index, name in enumerate(arm_names)}
        path = self.root / "arms.json"
        path.write_text(json.dumps({"reference_q": reference, "source": "fixture"}), encoding="utf-8")
        pool = self.prepare("arms", arm_reference=path)
        train = next(row for row in campaign.verify_pool(pool,"g1","stealth")["files"]
                     if row["split"] == "train")
        old = np.loadtxt(BytesIO(self.input_files[train["file"]]), delimiter=",")
        new = np.loadtxt(pool / train["file"], delimiter=",")
        columns = {7 + self.names.index(name) for name in reference}
        for index in range(36):
            if index in columns:
                continue
            np.testing.assert_array_equal(old[:,index],new[:,index])
        ref = json.loads((pool / "upper-body-reference.json").read_text())
        self.assertEqual(ref["reference_q"], reference)
        self.assertEqual(train["local_static_arms"]["source_upper_body_posture_retained"], False)
        self.assertTrue((pool.parent.parent.parent / "preparation.json").exists())

    def test_rejects_wrong_source_direction_unverified_hash_and_unequal_lengths(self):
        with patch.object(publisher, "_source_metrics",
                          side_effect=[self.metrics[0], self.metrics[0], self.metrics[2]]):
            with self.assertRaisesRegex(ValueError, "backward"):
                self.prepare("wrong-role")
        bad = json.loads(self.prepared_manifest.read_text())
        bad["clips"][0]["source_sha256"] = "0"*64
        self.prepared_manifest.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "hash"):
            self.prepare("bad-hash")
        bad["clips"][0]["source_sha256"] = self.rows[0]["source_sha256"]
        bad["clips"][1]["frames"] -= 1
        self.prepared_manifest.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "frame count"):
            self.prepare("bad-length")

    def test_fresh_output_required_and_parent_pool_is_immutable(self):
        before = {path: path.read_bytes() for path in self.heldout_pool.iterdir() if path.is_file()}
        self.prepare("first")
        self.assertEqual(before, {path: path.read_bytes() for path in self.heldout_pool.iterdir() if path.is_file()})
        with self.assertRaises(FileExistsError):
            self.prepare("first")
        self.prepare("second")

    def test_idle_rejects_high_drift_or_non_crouched_source_metrics(self):
        row = dict(self.metrics[2])
        row["low_speed_fraction_below_0_15"] = .5
        with self.assertRaisesRegex(ValueError, "idle"):
            publisher._validate_role("idle", row)
        row = dict(self.metrics[2])
        row["clearance_proxy_m_p05_p50_p95"] = [.75,.8,.82]
        with self.assertRaisesRegex(ValueError, "idle"):
            publisher._validate_role("idle", row)
        row = dict(self.metrics[2])
        row["upright_cos_p05_p50_p95"] = [.1,.2,.3]
        with self.assertRaisesRegex(ValueError, "idle"):
            publisher._validate_role("idle", row)
        row = dict(self.metrics[2])
        row["hands_forward_g1_proxy_m_p05_p50_p95"] = [-.1, 0., .1]
        with self.assertRaisesRegex(ValueError, "idle"):
            publisher._validate_role("idle", row)


if __name__ == "__main__":
    unittest.main()
