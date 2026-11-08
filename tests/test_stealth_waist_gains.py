"""CPU model/force checks for opt-in, named G1 waist PD gains."""

from contextlib import ExitStack
from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerTask.supermimic import mujoco_terrain as builder
from trackerTask.supermimic.terrain import generate, get_task, robot_parameters
from trackerLab.tracker_env.supermimic import terrain as terrain_module
from trackerLab.tracker_env.supermimic.command import CommandSim


WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
INVALID_SCALES = (None, True, False, "4", 0., -1., float("nan"),
                  float("inf"), -float("inf"), 1j, 10**1000)


class WaistMotorTests(unittest.TestCase):
    def test_pure_helper_preserves_inputs_and_selects_exact_names_in_any_order(self):
        spec, _, motors = robot_parameters("g1")
        order = list(range(28, -1, -1))
        names = [spec.joint_names[i] for i in order] + ["waist_roll_joint_extra"]
        source = [list(motors[i]) for i in order] + [[11., 12., 13., 14., .015]]
        before = [row[:] for row in source]
        with patch.dict(sys.modules, {"mujoco": None, "mujoco_warp": None, "warp": None}):
            result = builder.scale_waist_motors("g1", names, source, 4.)
        self.assertIsInstance(result, list)
        self.assertIsNot(result, source)
        self.assertEqual(source, before)
        for name, original, effective in zip(names, source, result):
            expected = original[:]
            if name in WAIST:
                expected[2] *= 4.
                expected[3] *= 2.
            self.assertEqual(tuple(expected), effective, name)
        source[0][2] += 123.
        self.assertNotEqual(result[0][2], source[0][2])

    def test_default_and_non_g1_motor_rows_are_unchanged(self):
        for robot in ("g1", "go2", "a3", "t1", "bumi", "pi_plus"):
            with self.subTest(robot=robot):
                spec, _, motors = robot_parameters(robot)
                result = builder.scale_waist_motors(robot, spec.joint_names, motors)
                self.assertEqual(result, list(motors))
                self.assertEqual(result, builder.scale_waist_motors(
                    robot, spec.joint_names, motors, 1.))
                if robot != "g1":
                    with self.assertRaisesRegex(ValueError, "requires G1"):
                        builder.scale_waist_motors(robot, spec.joint_names, motors, 2.)

    def test_validation_precedes_asset_and_backend_access(self):
        with patch.object(builder, "robot_parameters") as parameters:
            for scale in INVALID_SCALES:
                with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "waist_pd_scale"):
                    builder.build_model("g1", None, waist_pd_scale=scale)
            with self.assertRaisesRegex(ValueError, "requires G1"):
                builder.build_model("go2", None, waist_pd_scale=4.)
            parameters.assert_not_called()
        with patch.dict(sys.modules, {"mujoco_warp": None, "warp": None}):
            for constructor in (terrain_module.TerrainSim, CommandSim):
                for scale in INVALID_SCALES:
                    with self.subTest(constructor=constructor, scale=scale):
                        with self.assertRaisesRegex(ValueError, "waist_pd_scale"):
                            constructor(SimpleNamespace(robot="g1"), None, waist_pd_scale=scale)
                with self.assertRaisesRegex(ValueError, "requires G1"):
                    constructor(SimpleNamespace(robot="go2"), None, waist_pd_scale=4.)

    def test_helper_rejects_incomplete_ambiguous_and_overflowing_maps(self):
        motors = [(50., 37., 28.5, 1.8, .007)] * 3
        for names, rows in ((WAIST[:2], motors), (WAIST, motors[:2]),
                            (WAIST, [row[:4] for row in motors]),
                            ((WAIST[0], WAIST[0], WAIST[2]), motors),
                            ((WAIST[0], "waist_roll", WAIST[2]), motors)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                builder.scale_waist_motors("g1", names, rows, 4.)
        with self.assertRaisesRegex(ValueError, "nonfinite motor gains"):
            builder.scale_waist_motors("g1", WAIST, motors, 1e308)
        self.assertEqual(builder.validate_waist_pd_scale("g1", np.float64(.5)), .5)


class WaistModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec, height, motors = robot_parameters("g1")
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        path = Path(cls.directory.name) / "named-joints.xml"
        root = ET.Element("mujoco")
        ET.SubElement(root, "compiler", angle="radian")
        world = ET.SubElement(root, "worldbody")
        pelvis = ET.SubElement(world, "body", name="pelvis", pos="0 0 .76")
        ET.SubElement(pelvis, "freejoint")
        ET.SubElement(pelvis, "geom", type="sphere", size=".04", mass="1")
        # Physical joint order deliberately differs from the policy actuator order.
        for i in range(28, -1, -1):
            name = spec.joint_names[i]
            body = ET.SubElement(pelvis, "body", name="torso_link" if name == WAIST[2] else f"link_{i}",
                                 pos=f"{i * .1} 0 .1")
            ET.SubElement(body, "joint", name=name, axis="0 1 0", range="-2 2",
                          actuatorfrcrange=f"{-motors[i][0]} {motors[i][0]}")
            ET.SubElement(body, "geom", type="sphere", size=".01", mass=".1")
        path.write_text(ET.tostring(root, encoding="unicode"), encoding="ascii")
        cls.parameters = replace(spec, mjcf_path=str(path)), height, motors
        cls.terrain = generate("stealth", height, 42)

    def build(self, **kwargs):
        with patch.object(builder, "robot_parameters", return_value=self.parameters):
            return builder.build_model("g1", self.terrain, **kwargs)

    def test_assembled_named_gains_caps_and_old_default(self):
        original, spec, _, qpos, qvel, _, _, actuators, contract = self.build()
        explicit, *_, explicit_contract = self.build(waist_pd_scale=1.)
        self.assertEqual(contract, explicit_contract)
        self.assertEqual(contract["waist_pd_scale"], 1.)
        nominal_motors = self.parameters[2]
        self.assertEqual(contract["motors_effort_velocity_kp_kd_armature"], list(nominal_motors))
        unchanged = ("actuator_forcerange", "actuator_forcelimited", "jnt_actfrcrange",
                     "jnt_actfrclimited", "actuator_gear", "dof_armature", "dof_damping",
                     "dof_frictionloss", "body_mass", "body_inertia", "jnt_range", "qpos0")
        for field in (*unchanged, "actuator_gainprm", "actuator_biasprm"):
            np.testing.assert_array_equal(getattr(original, field), getattr(explicit, field))
        self.assertFalse(np.array_equal(qpos, np.arange(7, 36)))
        for factor in (2., 3., 4.):
            with self.subTest(factor=factor):
                scaled, *_, effective = self.build(waist_pd_scale=factor)
                self.assertEqual(effective["waist_pd_scale"], factor)
                self.assertEqual(effective["action_scale"], contract["action_scale"])
                self.assertEqual(effective["default_q"], contract["default_q"])
                for field in unchanged:
                    np.testing.assert_array_equal(getattr(original, field), getattr(scaled, field))
                for i, name in enumerate(spec.joint_names):
                    joint = scaled.joint(name).id
                    actuator = actuators[i]
                    self.assertEqual(scaled.actuator_trnid[actuator, 0], joint)
                    self.assertEqual(scaled.jnt_qposadr[joint], qpos[i])
                    self.assertEqual(scaled.jnt_dofadr[joint], qvel[i])
                    effort, velocity, kp, kd, armature = nominal_motors[i]
                    if name in WAIST:
                        kp *= factor
                        kd *= math.sqrt(factor)
                    self.assertAlmostEqual(scaled.actuator_gainprm[actuator, 0], kp)
                    np.testing.assert_array_equal(scaled.actuator_biasprm[actuator, :3], [0., -kp, -kd])
                    self.assertEqual(effective["motors_effort_velocity_kp_kd_armature"][i],
                                     (effort, velocity, kp, kd, armature))

    def test_forward_force_obeys_scaled_pd_and_original_caps(self):
        model, spec, _, qpos, qvel, _, _, actuators, contract = self.build(waist_pd_scale=4.)
        data = mujoco.MjData(model)
        data.qpos[qpos] = np.linspace(-.2, .2, 29)
        data.qvel[qvel] = np.linspace(-.3, .3, 29)
        data.ctrl[actuators] = data.qpos[qpos] + .01
        mujoco.mj_forward(model, data)
        for i, (_, _, kp, kd, _) in enumerate(contract["motors_effort_velocity_kp_kd_armature"]):
            expected = kp * .01 - kd * data.qvel[qvel[i]]
            self.assertAlmostEqual(data.actuator_force[actuators[i]], expected)
            self.assertAlmostEqual(data.qfrc_actuator[qvel[i]], expected)
        for sign in (-1., 1.):
            data.ctrl[:] = sign * 100.
            mujoco.mj_forward(model, data)
            for i, (effort, *_) in enumerate(self.parameters[2]):
                self.assertAlmostEqual(data.actuator_force[actuators[i]], sign * effort)
                self.assertAlmostEqual(data.qfrc_actuator[qvel[i]], sign * effort)

    def test_terrain_constructor_forwards_gains_to_actual_builder_before_warp(self):
        class BeforeWarp(RuntimeError):
            pass

        def cpu_factory(name):
            def allocate(*args, **kwargs):
                kwargs["device"] = "cpu"
                return getattr(torch, name)(*args, **kwargs)
            return allocate

        cpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), device=torch.device,
                              int32=torch.int32, float32=torch.float32,
                              **{name: cpu_factory(name) for name in ("zeros", "tensor", "randint")})
        wp, mw = ModuleType("warp"), ModuleType("mujoco_warp")
        wp.init = Mock(side_effect=BeforeWarp)
        for options, expected in (({}, 1.), ({"waist_pd_scale": 4.}, 4.)):
            sim = terrain_module.TerrainSim.__new__(terrain_module.TerrainSim)
            with self.subTest(options=options), ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, {"warp": wp, "mujoco_warp": mw}))
                stack.enter_context(patch.object(terrain_module, "torch", cpu))
                stack.enter_context(patch.object(builder, "robot_parameters", return_value=self.parameters))
                build = stack.enter_context(patch.object(terrain_module, "build_model", wraps=builder.build_model))
                with self.assertRaises(BeforeWarp):
                    sim.__init__(get_task("locomotion/g1/stealth"), None, num_envs=2, **options)
                self.assertEqual(build.call_args.kwargs["waist_pd_scale"], expected)
                self.assertEqual(sim.waist_pd_scale, expected)
                self.assertEqual(sim.contract["waist_pd_scale"], expected)
                actuator = sim.cpu_model.actuator("supermimic_waist_roll_joint").id
                self.assertAlmostEqual(sim.cpu_model.actuator_gainprm[actuator, 0],
                                       self.parameters[2][13][2] * expected)

    def test_command_constructor_forwards_scale_and_retains_parent_contract(self):
        mdp = SimpleNamespace(profile="stealth", target_clearance_m=.6992, entry_duration_s=2.42,
                              terrain_navigation=False, obs_dim=283, goal_dim=12, control_mode="wasd",
                              control_mode_version=1, wasd_reward_version=2)
        received = []

        def parent(sim, task, mdp, *args, **kwargs):
            received.append(kwargs)
            sim.waist_pd_scale = kwargs["waist_pd_scale"]
            sim.contract = {"waist_pd_scale": sim.waist_pd_scale, "parent_marker": True}

        for options, expected in (({}, 1.), ({"waist_pd_scale": 4.}, 4.)):
            with self.subTest(options=options), patch.object(terrain_module.TerrainSim, "__init__", new=parent):
                sim = CommandSim(get_task("locomotion/g1/stealth"), mdp, num_envs=2, device="cpu",
                                 command_mode_set="wasd", physical_domain_randomization=True, **options)
            self.assertEqual(received[-1]["waist_pd_scale"], expected)
            self.assertTrue(received[-1]["physical_domain_randomization"])
            self.assertEqual(sim.contract["waist_pd_scale"], expected)
            self.assertTrue(sim.contract["parent_marker"])


@unittest.skipUnless(Path(robot_parameters("g1")[0].mjcf_path).is_file(), "local G1 assets unavailable")
class LoadedWaistTests(unittest.TestCase):
    def test_fixed_pose_roll_force_balance_nominal_and_offnominal(self):
        # Saved pilot1500 run_config.json final arm and waist targets, policy order.
        upper_q = (-.03092535026371479, -.020353026688098907, .5199999809265137,
                   -1.586735725402832, .3756830394268036, -1.0000126361846924, .6509316563606262,
                   -.7050104737281799, -.6297845840454102, .438135027885437,
                   -1.874166488647461, -.09915832430124283, .8154126405715942, .23525474965572357,
                   -.02954377792775631, .4701469838619232, -.5789214372634888)
        expected = {1.: (27.907, None), 2.: (4.271, 15.571), 3.: (2.259, 6.907), 4.: (1.535, 4.403)}
        for factor, bounds in expected.items():
            model, spec, _, qpos, qvel, _, com, actuators, _ = builder.build_model(
                "g1", generate("stealth", .76, 42), waist_pd_scale=factor, self_collisions=True)
            data = mujoco.MjData(model)
            pose = model.qpos0.copy()
            pose[:3] = (0., 0., .76)
            pose[qpos] = spec.default_q
            pose[qpos[12:]] = upper_q
            for offnominal in (False, True):
                if offnominal and bounds[1] is None:
                    continue
                with self.subTest(factor=factor, offnominal=offnominal):
                    target = upper_q[1]
                    kp = model.actuator_gainprm[actuators[13], 0]
                    if offnominal:
                        model.body_mass[:] *= 1.1
                        model.body_inertia[:] *= 1.1
                        model.body_ipos[com] += (-.025, .025, .025)
                        pose[qpos[12]] += .01
                        pose[qpos[14]] -= .01
                        target -= .01
                        kp *= .85
                        mujoco.mj_setConst(model, data)
                    data.qvel[:] = 0.
                    def net_torque(angle):
                        data.qpos[:] = pose
                        data.qpos[qpos[13]] = angle
                        mujoco.mj_forward(model, data)
                        return np.clip(kp * (target - angle), -45., 45.) - data.qfrc_bias[qvel[13]]
                    lo, hi = -.52, .52
                    self.assertGreater(net_torque(lo), 0.)
                    self.assertLess(net_torque(hi), 0.)
                    for _ in range(40):
                        mid = (lo + hi) / 2
                        if net_torque(mid) > 0.:
                            lo = mid
                        else:
                            hi = mid
                    equilibrium = (lo + hi) / 2
                    self.assertAlmostEqual(net_torque(equilibrium), 0., delta=1e-7)
                    error = abs(math.degrees(equilibrium - target))
                    self.assertAlmostEqual(error, bounds[int(offnominal)], delta=.02)
                    self.assertLess(abs(kp * (target - equilibrium)), 45.)


if __name__ == "__main__":
    unittest.main()
