"""Unit, joint-order, provenance and registry regression checks."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from prepare_humanoids import (
    convert_a3, convert_bumi, convert_pi_plus, prepare_model, project_joint_limits, validate_motion,
)
from trackerTask.supermimic.registry import TASK_REGISTRY, get_task
from trackerTask.supermimic.robots import get_robot


class HumanoidResourceTests(unittest.TestCase):
    def test_repeated_mjcf_scene_sections_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "output"
            source.mkdir()
            (source / "robot.xml").write_text(
                '<mujoco><compiler/><asset><material name="robot"/></asset>'
                '<worldbody><body name="base"><freejoint/></body></worldbody>'
                '<asset><texture name="sky"/><material name="groundplane"/></asset>'
                '<worldbody><light/><geom type="plane" name="floor"/></worldbody></mujoco>')
            prepare_model(source, target, {
                "model": "robot.xml", "fixed_joints": [],
                "repository": "fixture", "revision": "fixture",
            }, "fixture")
            root = ET.parse(target / "mjcf/fixture.xml").getroot()
            self.assertEqual(len(root.findall("worldbody")), 1)
            self.assertEqual([node.tag for node in root.find("worldbody")], ["body"])
            self.assertEqual(len(root.findall("asset")), 1)
            self.assertEqual([node.get("name") for node in root.find("asset")], ["robot"])
            self.assertIsNotNone(root.find("worldbody/body/freejoint"))

    def test_named_units_and_head_rejection(self):
        joints = get_robot("a3").joint_names
        fields = ["Frame", *(f"root_translate{x}" for x in "XYZ"),
                  *(f"root_rotate{x}" for x in "XYZ"), *reversed(joints),
                  "head_yaw_joint", "head_pitch_joint"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.csv"

            def write(head=0):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for frame in range(3):
                        row = dict.fromkeys(fields, 0)
                        row.update(Frame=frame, root_translateX=200, root_translateZ=110,
                                   root_rotateZ=90, head_pitch_joint=head)
                        row[joints[0]] = 45
                        writer.writerow(row)
            write()
            result = convert_a3(path, joints)
            np.testing.assert_allclose(result[0, :3], [2, 0, 1.1])
            np.testing.assert_allclose(result[0, 3:7], [0, 0, 2**-.5, 2**-.5], atol=1e-7)
            self.assertAlmostEqual(result[0, 7], np.pi / 4)
            self.assertEqual(result.shape, (3, 36))
            validate_motion(result, joints)
            write(head=3)
            with self.assertRaisesRegex(ValueError, "head angles"):
                convert_a3(path, joints)

    def test_invalid_motion_quaternion(self):
        joints = get_robot("t1").joint_names
        values = np.zeros((3, 7 + len(joints)))
        values[:, 2], values[:, 6] = .7, 1
        validate_motion(values, joints)
        values[0, 6] = 0
        with self.assertRaisesRegex(ValueError, "unit norm"):
            validate_motion(values, joints)

    def test_reference_projection_preserves_root_and_records_adjustment(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "robot.xml"
            model.write_text('<mujoco><worldbody><body><joint name="elbow" range="-2.26 0"/>'
                             '</body></worldbody></mujoco>')
            values = np.tile([1, 2, .6, 0, 0, 0, 1, -2.59], (3, 1))
            projected, record = project_joint_limits(values, model, ["elbow"])
            np.testing.assert_array_equal(projected[:, :7], values[:, :7])
            np.testing.assert_allclose(projected[:, 7], -2.26)
            self.assertAlmostEqual(record["max_correction_rad"], .33)
            self.assertEqual(record["modified_joint_samples"], 3)
            values[0, 7] = -3
            with self.assertRaisesRegex(ValueError, "0.5 radians"):
                project_joint_limits(values, model, ["elbow"])

    def test_pi_plus_named_columns_include_both_wrists(self):
        joints = get_robot("pi_plus").joint_names
        fields = ["root pos x", "root pos y", "root pos z",
                  "root rot x", "root rot y", "root rot z", "root rot w",
                  *(name.removesuffix("_joint") for name in reversed(joints))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                row = dict.fromkeys(fields, 0.)
                row.update({"root pos z": .35, "root rot w": 1., "l_wrist": .2, "r_wrist": -.4})
                writer.writerows([row] * 3)
            result = convert_pi_plus(path, joints)
            self.assertEqual(result.shape, (3, 29))
            self.assertAlmostEqual(result[0, 7 + joints.index("l_wrist_joint")], .2)
            self.assertAlmostEqual(result[0, 7 + joints.index("r_wrist_joint")], -.4)
            validate_motion(result, joints)

    def test_bumi_vendor_wxyz_joint_order_and_frame_rate(self):
        joints = get_robot("bumi").joint_names
        source_order = list(reversed(joints))
        payload = {
            "fps": np.array([50]), "body_pos_w": np.tile([0, 0, .6], (3, 22, 1)),
            "body_quat_w": np.tile([2**-.5, 0, 0, 2**-.5], (3, 22, 1)),
            "joint_pos": np.tile(np.arange(21), (3, 1)),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.npz"
            np.savez(path, **payload)
            result = convert_bumi(path, joints, 50, source_order)
            np.testing.assert_array_equal(result[0, 3:7], [0, 0, 2**-.5, 2**-.5])
            np.testing.assert_array_equal(result[0, 7:], np.arange(20, -1, -1))
            validate_motion(result, joints)
            with self.assertRaisesRegex(ValueError, "frame rate"):
                convert_bumi(path, joints, 60, source_order)
            with self.assertRaisesRegex(ValueError, "joint names"):
                convert_bumi(path, joints, 50, [source_order[0]] * 21)

    def test_registry_presets_and_robot_contracts(self):
        for entry in TASK_REGISTRY.values():
            self.assertTrue((REPO / entry.preset).is_file(), entry.preset)
            options = json.loads((REPO / entry.preset).read_text())
            self.assertEqual(options.get("task", "g1"), entry.runtime_task)
            self.assertEqual(options.get("task_profile", "standard"), entry.profile)
            self.assertEqual(get_robot(entry.id).name, entry.robot)
        for robot, joints in (("g1", 29), ("go2", 12), ("a3", 29), ("t1", 23),
                              ("bumi", 21), ("pi_plus", 22)):
            spec = get_robot(robot)
            self.assertEqual(spec.num_joints, joints)
            self.assertEqual(len(set(spec.joint_names)), joints)
            self.assertEqual(len(spec.default_q), joints)
            self.assertEqual(len(spec.action_scale), joints)
            self.assertTrue(all(scale > 0 for scale in spec.action_scale))
            self.assertEqual(get_task(robot).robot, robot)

    def test_registry_discovery_without_simulator(self):
        output = subprocess.check_output(
            [sys.executable, str(REPO / "scripts/rls/supermimic/tasks.py"), "list"], text=True)
        self.assertIn("tracking/a3", output)
        self.assertIn("tracking/t1", output)
        self.assertIn("tracking/bumi", output)
        self.assertIn("tracking/pi_plus", output)


if __name__ == "__main__":
    unittest.main()
