"""CPU integrity, Go2 FK, resampling and held-out split checks."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
import prepare_go2_style as module
from trackerTask.supermimic.robots import get_robot


GO2_MODEL_AVAILABLE = Path(get_robot("go2").mjcf_path).is_file()
requires_go2_model = unittest.skipUnless(
    GO2_MODEL_AVAILABLE, "Local Go2 MJCF assets unavailable")


def numeric_source(spec, x=0., jump=False):
    path = Path(spec.mjcf_path).resolve()
    model_path = str(path.relative_to(Path.cwd())) if path.is_relative_to(Path.cwd()) else str(path)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    qids = model.jnt_qposadr[[model.joint(name).id for name in spec.joint_names]]
    raw = np.zeros((13, 61), dtype=np.float32)
    for frame in range(len(raw)):
        height = .3 + (.15 * np.sin(np.pi * frame/12)**2 if jump else 0)
        data.qpos[:7] = [x + frame*.004, 0, height, 1, 0, 0, 0]
        data.qpos[qids] = spec.default_q
        mujoco.mj_forward(model, data)
        raw[frame, 48:51], raw[frame, 51] = data.qpos[:3], 1
        raw[frame, 6:18] = [spec.default_q[spec.joint_names.index(name)]
                            for name in module.LOCK["raw_joint_order"]]
        raw[frame, 36:48] = np.concatenate([
            data.xpos[model.body(leg + "_calf").id]
            + data.xmat[model.body(leg + "_calf").id].reshape(3, 3) @ [0, 0, -.213]
            for leg in module.LOCK["raw_foot_order"]
        ])
    return raw


def fixture_source(directory):
    spec, records, originals = module.go2_spec(), [], {}
    for name in module.DOCUMENTS:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("local numeric test fixture", encoding="utf-8")
    for index, (clip, kind, _) in enumerate(module.PLAN):
        raw = numeric_source(spec, x=float(index), jump=kind == "jump")
        alias = {"ai4_dog_canter": "ai4_dog_run_00",
                 "ai4_dog_right_turn": "ai4_dog_left_turn"}.get(clip)
        if alias:
            raw = originals[alias].copy()
        originals[clip] = raw
        folder = directory / "data" / clip
        folder.mkdir(parents=True)
        np.save(folder / "motion.npy", raw, allow_pickle=False)
        meta = {"name": clip, "fps": 60, "n_frames": 13,
                "source": "numeric fixture", "source_url": "fixture",
                "attribution": "test", "license": "BSD-3-Clause" if clip.startswith("solo8") else "CC BY-NC 4.0"}
        config = {"env_cfg": {"num_actions": 12, "num_dofs": 12, "control_freq": 60,
                              "urdf_path": "urdf/go2/urdf/go2.urdf",
                              "dof_names": list(spec.joint_names)}}
        module.write_json(folder / "clip.json", meta)
        module.write_json(folder / "config.json", config)
    for name in sorted(module.expected_files()):
        content = (directory / name).read_bytes()
        value = hashlib.sha256(content).hexdigest()
        records.append({"file": name, "bytes": len(content), "sha256": value,
                        "source_hash_algorithm": "sha256", "source_hash": value})
    module.write_json(directory / "source_manifest.json", {
        "dataset": module.DATASET, "revision": module.REVISION,
        "clips": [row[0] for row in module.PLAN], "files": records,
    })
    return originals


def fixture_manifest(directory):
    records = []
    for name in sorted(module.expected_files()):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"fixture:{name}".encode()
        path.write_bytes(content)
        value = hashlib.sha256(content).hexdigest()
        records.append({"file": name, "bytes": len(content), "sha256": value,
                        "source_hash_algorithm": "sha256", "source_hash": value})
    module.write_json(directory / "source_manifest.json", {
        "dataset": module.DATASET, "revision": module.REVISION,
        "clips": [row[0] for row in module.PLAN], "files": records,
    })


class Go2StyleTests(unittest.TestCase):
    def test_pinned_plan_uses_run02_and_jumpb_as_held_out_sources(self):
        self.assertEqual(module.REVISION, "7f03a3fb7f6b2e684899530862ba103e81570363")
        self.assertEqual([clip for clip, _, split in module.PLAN if split == "test"],
                         ["ai4_dog_run_02", "solo8_jump_forward_b"])
        self.assertEqual(len(module.PLAN), 11)
        self.assertFalse(any(name.endswith((".pkl", ".pt")) for name in module.expected_files()))

    def test_resample_has_fifty_hz_duration_and_unit_quaternions(self):
        spec = get_robot("go2")
        raw = np.zeros((13, 61), dtype=np.float32)
        raw[:, 48] = np.arange(len(raw)) * .004
        raw[:, 50] = .3 + .15 * np.sin(np.pi * np.arange(len(raw)) / 12) ** 2
        raw[:, 51] = 1
        values = module.convert_array(raw, spec.joint_names)
        output = module.resample_50hz(values)
        self.assertEqual(output.shape, (11, 19))
        np.testing.assert_allclose(output[[0, -1]], values[[0, -1]], atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(output[:, 3:7], axis=-1), 1)
        self.assertGreater(np.ptp(output[:, 2]), .1)
        for bad in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                module.resample_50hz(values, bad)

    @requires_go2_model
    def test_everyframe_fk_guard_rejects_corrupt_feet_and_wrong_joint_order(self):
        spec = module.go2_spec()
        raw = numeric_source(spec)
        values = module.convert_array(raw, spec.joint_names)
        self.assertEqual(module.validate_fk(raw, values, spec)["frames_checked"], len(raw))
        broken = raw.copy()
        broken[-1, 36] += .01
        with self.assertRaisesRegex(ValueError, "full-frame FK"):
            module.validate_fk(broken, values, spec)
        wrong = values.copy()
        wrong[:, 7:10] = wrong[:, 7:10][:, ::-1]
        with self.assertRaises(ValueError):
            module.validate_fk(raw, wrong, spec)

    def test_grouping_deduplicates_same_split_and_rejects_cross_split_bytes(self):
        first = dict(clip="a", clip_type="run", split="train", raw_sha256="x", source_pose_sha256="y")
        alias = dict(first, clip="b", clip_type="canter")
        selected, aliases = module.group_sources([first, alias])
        self.assertEqual(selected, [first])
        self.assertEqual(aliases[0]["alias_of"], "a")
        for crossed in (dict(alias, split="test"), dict(alias, raw_sha256="z", split="test")):
            with self.assertRaisesRegex(ValueError, "crosses"):
                module.group_sources([first, crossed])

    def test_clip_identity_requires_go2_config_and_authoritative_license(self):
        spec = get_robot("go2")
        meta = dict(name="solo8_jump_forward_a", fps=60, license="BSD-3-Clause", attribution="source")
        config = {"env_cfg": {"urdf_path": "urdf/go2/go2.urdf", "num_actions": 12,
                              "num_dofs": 12, "control_freq": 60, "dof_names": list(spec.joint_names)}}
        module.validate_metadata(meta["name"], meta, config, spec)
        with self.assertRaises(ValueError):
            module.validate_metadata(meta["name"], dict(meta, license="CC BY-NC 4.0"), config, spec)
        config["env_cfg"]["urdf_path"] = "urdf/solo8/solo8.urdf"
        with self.assertRaisesRegex(ValueError, "Go2"):
            module.validate_metadata(meta["name"], meta, config, spec)

    def test_numeric_loader_rejects_object_arrays_without_pickle(self):
        buffer = io.BytesIO()
        np.save(buffer, np.array([{"unsafe": "object"}], dtype=object), allow_pickle=True)
        buffer.seek(0)
        with self.assertRaisesRegex(ValueError, "allow_pickle"):
            np.load(buffer, allow_pickle=False)

    @requires_go2_model
    def test_local_pool_end_to_end_fk_floor_split_config_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "raw", Path(directory) / "prepared"
            source.mkdir()
            fixture_source(source)
            with patch.object(module, "require_ignored_data", side_effect=lambda p: p):
                result = module.prepare(source, output)
            self.assertEqual((len(result["training_files"]), len(result["evaluation_files"])), (7, 2))
            self.assertEqual(len(result["aliases"]), 2)
            self.assertFalse(result["pickle_loaded"])
            for row in result["files"]:
                self.assertEqual(row["raw_fk"]["frames_checked"], 13)
                self.assertGreaterEqual(row["physical_floor"]["minimum_m"], 0)
                if row["clip_type"] == "jump":
                    self.assertTrue(row["floor_alignment"]["airborne_trajectory_preserved"])
                    self.assertFalse(row["floor_alignment"]["per_frame_clamp"])
                    self.assertGreater(np.ptp(np.loadtxt(output / row["file"], delimiter=",")[:, 2]), .1)
            self.assertEqual(module.verify_output(output)["status"], "ready")
            (output / result["training_files"][0]).write_text("corrupt", encoding="utf-8")
            with self.assertRaises(ValueError):
                module.verify_output(output)

    def test_tampered_or_unexpected_source_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            fixture_manifest(source)
            self.assertEqual(module.verify_source(source)["revision"], module.REVISION)
            (source / "LICENSE").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                module.verify_source(source)

    def test_derivatives_are_required_to_stay_in_ignored_data(self):
        with self.assertRaisesRegex(ValueError, "data/"):
            module.require_ignored_data(REPO / "outputs/go2")


if __name__ == "__main__":
    unittest.main()
