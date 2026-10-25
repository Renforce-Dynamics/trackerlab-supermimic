"""Unit, joint-order, provenance and registry regression checks."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from prepare_humanoids import convert_a3, validate_motion
from trackerTask.supermimic.registry import TASK_REGISTRY, get_task
from trackerTask.supermimic.robots import get_robot


class HumanoidResourceTests(unittest.TestCase):
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

    def test_registry_presets_and_robot_contracts(self):
        for entry in TASK_REGISTRY.values():
            self.assertTrue((REPO / entry.preset).is_file(), entry.preset)
            options = json.loads((REPO / entry.preset).read_text())
            self.assertEqual(options.get("task", "g1"), entry.runtime_task)
            self.assertEqual(options.get("task_profile", "standard"), entry.profile)
            self.assertEqual(get_robot(entry.id).name, entry.robot)
        for robot, joints in (("g1", 29), ("go2", 12), ("a3", 29), ("t1", 23)):
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


if __name__ == "__main__":
    unittest.main()
