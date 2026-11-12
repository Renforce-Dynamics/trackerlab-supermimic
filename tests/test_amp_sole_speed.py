"""CPU sole-speed contracts: asset-free G1-layout fixture plus optional source G1."""

import copy
from contextlib import chdir
from dataclasses import replace
from io import BytesIO
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.amp_style import AMPStyle
from trackerLab.algorithms.supermimic.locomotion_quality import (
    LocomotionQuality, _g1_sole_geometry, _material_sole_state,
)
from trackerLab.kinematics.supermimic.robot import RobotKinematics
from trackerLab.kinematics.supermimic.rotations import so3_exp
from trackerTask.supermimic.gait import FOOT_BODIES
from trackerTask.supermimic.robots import get_robot


MODE = "postural_end_effectors_vx_sole_speed"
BASE_MODE = "postural_end_effectors_vx"
SOLE_LABEL = "sole_horizontal_material_speed_rms/robot_height"
GEOMETRY_LOADER = "trackerLab.algorithms.supermimic.amp_style._g1_sole_geometry"
STATE_NAMES = ("root_p", "root_R", "root_v", "root_w", "q", "qd")


def synthetic_g1_xml(spec):
    """G1 names and sole spheres on an illustrative 29-hinge linkage.

    Link lengths and inertias are test geometry. Each joint has a nonzero
    anchor to exercise full FK origin velocities; no meshes or includes are
    required. The four centers and .005 m radii match the G1 sole colliders.
    """
    root = ET.Element("mujoco", model="synthetic_g1_amp_sole")
    ET.SubElement(root, "compiler", angle="radian")
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "joint", type="hinge", limited="true", range="-3.14 3.14")

    def body(parent, name, position):
        link = ET.SubElement(parent, "body", name=name, pos=" ".join(map(str, position)))
        ET.SubElement(link, "inertial", pos="0 0 0", mass=".1", diaginertia=".001 .001 .001")
        return link

    def chain(parent, names, positions, axes):
        for name, position, axis in zip(names, positions, axes, strict=True):
            link_name = spec.com_body if name == "waist_pitch_joint" else name.removesuffix("_joint") + "_link"
            parent = body(parent, link_name, position)
            ET.SubElement(parent, "joint", name=name, axis=axis, pos=".003 .002 .001")
        return parent

    pelvis = body(ET.SubElement(root, "worldbody"), spec.root_body, (0., 0., .76))
    ET.SubElement(pelvis, "freejoint", name="floating_base")
    for side, start in ((1, 0), (-1, 6)):
        foot = chain(
            pelvis, spec.joint_names[start:start + 6],
            ((0., side * .1, -.05), (0., 0., -.03), (0., 0., -.03),
             (.015, 0., -.28), (0., 0., -.27), (0., 0., -.035)),
            ("0 1 0", "1 0 0", "0 0 1", "0 1 0", "0 1 0", "1 0 0"),
        )
        for center in ((-.05, .025, -.03), (-.05, -.025, -.03),
                       (.12, .03, -.03), (.12, -.03, -.03)):
            ET.SubElement(foot, "geom", type="sphere", size=".005",
                          pos=" ".join(map(str, center)), contype="1", conaffinity="1")
    torso = chain(pelvis, spec.joint_names[12:15],
                  ((0., 0., .06), (0., 0., .04), (0., 0., .04)),
                  ("0 0 1", "1 0 0", "0 1 0"))
    for side, start in ((1, 15), (-1, 22)):
        chain(
            torso, spec.joint_names[start:start + 7],
            ((0., side * .18, .15), (0., side * .04, 0.), (0., 0., -.03),
             (.02, 0., -.18), (0., 0., -.16), (.015, 0., -.02), (.035, 0., 0.)),
            ("0 1 0", "1 0 0", "0 0 1", "0 1 0", "1 0 0", "0 1 0", "0 0 1"),
        )
    return ET.ElementTree(root)


def motion_fixture(kin, model_kind):
    t = torch.arange(25, dtype=torch.float32) * .02
    phase = .9 * t[:, None] + .2 * torch.arange(kin.num_joints)[None]
    return SimpleNamespace(
        root_p=torch.stack((.17 * t, -.04 * t, .76 + .01 * t), -1),
        root_R=so3_exp(torch.stack((t * 0, t * 0, .3 * t), -1))
        @ so3_exp(torch.tensor([.12, -.18, .07])),
        root_v=torch.tensor([.17, -.04, .01]).expand(25, 3).clone(),
        root_w=torch.tensor([.08, -.12, .3]).expand(25, 3).clone(),
        q=torch.tensor(kin.robot.default_q) + .08 * phase.sin(),
        qd=.072 * phase.cos(),
        clip_start=torch.tensor([0, 12]), clip_end=torch.tensor([11, 24]),
        clip_len=torch.tensor([12, 13]), clip_id=torch.tensor([0] * 12 + [1] * 13),
        names=["g1-fixture-left", "g1-fixture-right"],
        provenance={"source": "analytic test states; no simulation", "model_kind": model_kind},
    )


def recorded(motion, indices):
    return tuple(getattr(motion, name)[indices].detach().clone() for name in STATE_NAMES)


class _AMPSoleSpeedContracts:
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, cls.threads)
        torch.set_num_threads(2)
        cls.enterClassContext(chdir(REPO))
        spec = get_robot("g1")
        path = cls.model_path(spec)
        # MuJoCo's Windows file loader accepts this ASCII relative asset path.
        try:
            path = os.path.relpath(path)
        except ValueError:
            # Windows temporary files can live on a different drive.
            path = str(path)
        cls.spec = replace(spec, mjcf_path=path)
        cls.kin = RobotKinematics("cpu", cls.spec)

    def setUp(self):
        torch.manual_seed(73)
        self.motion = motion_fixture(self.kin, self.model_kind)

    def learner(self, mode=MODE, kin=None, motion=None, **kwargs):
        return AMPStyle(
            kin or self.kin, motion or self.motion, kwargs.pop("robot_height", .76),
            feature_mode=mode, hidden=kwargs.pop("hidden", (16, 8)), replay_capacity=16,
            **kwargs,
        )

    def held(self, rate=0.):
        state = list(recorded(self.motion, torch.zeros(10, dtype=torch.long)))
        angles = torch.arange(10) * .02 * rate
        turn = so3_exp(torch.stack((angles * 0, angles * 0, angles), -1))
        state[1] = turn @ state[1]
        state[2].zero_()
        state[3] = torch.tensor([0., 0., rate]).expand(10, 3).clone()
        state[5].zero_()
        return tuple(state)

    def assert_finite(self, value, nonzero=False):
        self.assertIsNotNone(value)
        self.assertTrue(bool(torch.isfinite(value).all()))
        if nonzero:
            self.assertGreater(value.abs().sum().item(), 0.)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for name in left:
                self.assert_tree_equal(left[name], right[name])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_exact_old_prefix_and_opt_in_layout(self):
        for hands in (None, (), ("right_wrist_yaw_link", "left_wrist_yaw_link")):
            with self.subTest(hands=hands):
                base = self.learner(BASE_MODE, history_steps=10, hand_body_names=hands)
                learner = self.learner(history_steps=10, hand_body_names=hands)
                layout = copy.deepcopy(base.feature_layout)
                layout.update(version=2, feature_mode=MODE, sole_speed=learner.feature_layout["sole_speed"])
                layout["frame"].append([SOLE_LABEL, 2])
                self.assertEqual(learner.feature_layout, layout)
                self.assertEqual(learner.frame_dim, base.frame_dim + 2)
                self.assertEqual(learner.feature_dim, 10 * learner.frame_dim)
                if hands is None:
                    self.assertEqual((learner.frame_dim, learner.feature_dim), (119, 1190))
                self.assertEqual(learner._expert_frames[..., :-2].contiguous().numpy().tobytes(),
                                 base._expert_frames.numpy().tobytes())
                for state in (self.held(), self.held(.55), recorded(self.motion, [0, 4, 13])):
                    actual = learner.frame_features(state)
                    self.assertEqual(actual[..., :-2].contiguous().numpy().tobytes(),
                                     base.frame_features(state).numpy().tobytes())
                    torch.testing.assert_close(actual, learner.frame_features(state, reference=True),
                                               atol=0, rtol=0)
                geometry = layout["sole_speed"]
                self.assertEqual(geometry["body_names"], list(FOOT_BODIES["g1"]))
                self.assertEqual(geometry["tracked_indices"], [3, 6])
                self.assertEqual(geometry["body_indices"], [6, 12])
                self.assertEqual(geometry["point_convention"], "instantaneous_world_down_sphere_pole")
                centers, radii = _g1_sole_geometry(self.spec.mjcf_path, "cpu")
                self.assertEqual(geometry["sphere_centers_body_m"], centers.tolist())
                self.assertEqual(geometry["sphere_radii_m"], radii.tolist())
                self.assertNotEqual(base.expert_hash, learner.expert_hash)

    def test_held_idle_and_both_rigid_yaw_rates_are_separated(self):
        base, learner = self.learner(BASE_MODE, history_steps=10), self.learner(history_steps=10)
        idle = learner.frame_features(self.held())
        torch.testing.assert_close(idle[..., -2:], torch.zeros(10, 2), atol=0, rtol=0)
        for rate in (.55, -.55):
            with self.subTest(rate=rate):
                state = self.held(rate)
                frames = learner.frame_features(state)
                self.assertEqual(base.frame_features(state).numpy().tobytes(),
                                 base.frame_features(self.held()).numpy().tobytes())
                self.assertTrue(bool((frames[..., -2:] > .02).all()))
                self.assertFalse(torch.equal(learner.window_features(frames), learner.window_features(idle)))
                p, R = self.kin.forward(*state)[:2]
                feet = learner._sole_tracked_indices
                points = p[..., feet, None, :] + (
                    R[..., feet, None, :, :] @ learner._sole_centers[..., None]).squeeze(-1)
                points = points - learner._sole_radii[..., None] * points.new_tensor([0., 0., 1.])
                radius2 = (points - state[0][..., None, None, :])[..., :2].square().sum(-1)
                expected = abs(rate) * radius2.mean(-1).sqrt() / .76
                torch.testing.assert_close(frames[..., -2:], expected, atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(learner.frame_features(self.held(.55))[..., -2:],
                                   learner.frame_features(self.held(-.55))[..., -2:],
                                   atol=2e-7, rtol=2e-6)

    def test_global_constant_heading_invariance_with_full_twist_and_joint_rates(self):
        learner = self.learner(history_steps=10)
        state = recorded(self.motion, torch.arange(10))
        baseline = learner.frame_features(state)
        for angle in (-1.7, .8, 2.4):
            with self.subTest(angle=angle):
                Q = so3_exp(torch.tensor([0., 0., angle]))
                p, R, v, w, q, qd = state
                transformed = (p @ Q.T, Q @ R, v @ Q.T, w @ Q.T, q, qd)
                actual = learner.frame_features(transformed)
                torch.testing.assert_close(actual, baseline, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(learner.window_features(actual),
                                           learner.window_features(baseline), atol=2e-6, rtol=2e-6)

    def test_full_fk_inputs_and_material_velocities_match_mujoco_finite_differences(self):
        learner = self.learner()
        state = recorded(self.motion, [7])
        with patch.object(self.kin, "forward", wraps=self.kin.forward) as forward:
            actual = learner.frame_features(state)[0, -2:]
        self.assertEqual(forward.call_count, 2)
        for value, supplied in zip(forward.call_args.args, state):
            torch.testing.assert_close(value, supplied, atol=0, rtol=0)

        model = mujoco.MjModel.from_xml_path(self.spec.mjcf_path)
        data = mujoco.MjData(model)
        arrays = [value[0].double().numpy() for value in state]
        root_joint = model.body_jntadr[model.body(self.spec.root_body).id]
        root_address = model.jnt_qposadr[root_joint]
        joint_addresses = [model.jnt_qposadr[model.joint(n).id] for n in self.kin.joint_names]
        feet = [model.body(n).id for n in FOOT_BODIES["g1"]]
        geom_ids = [[g for g in range(model.ngeom) if model.geom_bodyid[g] == b
                     and (model.geom_contype[g] or model.geom_conaffinity[g])] for b in feet]
        self.assertEqual([len(ids) for ids in geom_ids], [4, 4])

        def points_at(dt, material=None):
            p, R, v, w, q, qd = arrays
            turn = so3_exp(torch.tensor(w * dt, dtype=torch.float64)).numpy()
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, (turn @ R).ravel())
            data.qpos[:] = model.qpos0
            data.qpos[root_address:root_address + 3] = p + dt * v
            data.qpos[root_address + 3:root_address + 7] = quat
            data.qpos[joint_addresses] = q + dt * qd
            mujoco.mj_kinematics(model, data)
            if material is None:
                material = [
                    model.geom_pos[ids] - model.geom_size[ids, :1]
                    * (data.xmat[b].reshape(3, 3).T @ np.array([0., 0., 1.]))
                    for b, ids in zip(feet, geom_ids)
                ]
            points = np.stack([data.xpos[b] + local @ data.xmat[b].reshape(3, 3).T
                               for b, local in zip(feet, material)])
            return points, material

        expected_points, material = points_at(0.)
        dt = 1e-5
        finite_velocity = (points_at(dt, material)[0] - points_at(-dt, material)[0]) / (2 * dt)
        expected = np.sqrt(np.square(finite_velocity[..., :2]).sum(-1).mean(-1)) / .76
        np.testing.assert_allclose(actual.detach().numpy(), expected, atol=3e-7, rtol=2e-6)
        self.assertEqual(data.time, 0.)

        P, R, V, W = self.kin.forward(*state)
        indices = learner._sole_tracked_indices
        foot_state = (P[..., indices, :], R[..., indices, :, :],
                      V[..., indices, :], W[..., indices, :])
        points, velocity = _material_sole_state(
            *foot_state, learner._sole_centers, learner._sole_radii)
        np.testing.assert_allclose(points[0].numpy(), expected_points, atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(velocity[0].numpy(), finite_velocity, atol=3e-7, rtol=2e-6)
        _, center_velocity = _material_sole_state(
            *foot_state, learner._sole_centers, torch.zeros_like(learner._sole_radii))
        self.assertGreater((velocity[..., :2] - center_velocity[..., :2]).abs().max().item(), 1e-4)
        mean_speed = torch.linalg.vector_norm(velocity[..., :2], dim=-1).mean(-1) / .76
        self.assertGreater((actual - mean_speed[0]).abs().max().item(), 1e-5)
        quality = SimpleNamespace(sole_centers=learner._sole_centers, sole_radii=learner._sole_radii)
        self.assert_tree_equal(LocomotionQuality._sole_state(quality, *foot_state), (points, velocity))

    def test_joint_rates_and_root_translation_each_affect_sole_speed(self):
        learner = self.learner()
        state = list(self.held())
        state[5][:, self.kin.joint_names.index("left_ankle_roll_joint")] = .7
        speed = learner.frame_features(state)[..., -2:]
        self.assertTrue(bool((speed[:, 0] > 0).all()))
        torch.testing.assert_close(speed[:, 1], torch.zeros(10), atol=0, rtol=0)
        state[5].zero_()
        state[2][:] = torch.tensor([.3, .4, 9.])
        torch.testing.assert_close(learner.frame_features(state)[..., -2:],
                                   torch.full((10, 2), .5 / .76))
        half_scale = self.learner(robot_height=1.52)
        torch.testing.assert_close(half_scale.frame_features(state)[..., -2:],
                                   torch.full((10, 2), .5 / 1.52))

    def test_zero_speed_has_zero_values_and_finite_zero_state_gradients(self):
        learner = self.learner()
        state = tuple(value[:1].clone().requires_grad_() for value in self.held())
        speed = learner.frame_features(state)[..., -2:]
        torch.testing.assert_close(speed, torch.zeros_like(speed), atol=0, rtol=0)
        for gradient in torch.autograd.grad(speed.sum(), state):
            self.assert_finite(gradient)
            torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)

    def test_rms_handles_tiny_and_large_finite_speeds_without_squared_overflow(self):
        learner = self.learner()
        for magnitude in (1e-25, 1., 1e20, 1e30):
            with self.subTest(magnitude=magnitude):
                state = list(value[:1] for value in self.held())
                state[2] = torch.tensor([[.3 * magnitude, .4 * magnitude, 0.]], requires_grad=True)
                speed = learner.frame_features(state)[..., -2:]
                expected = torch.full_like(speed, .5 * magnitude / .76)
                self.assert_finite(speed, nonzero=True)
                torch.testing.assert_close(speed, expected, atol=0, rtol=3e-6)
                gradient, = torch.autograd.grad(speed.sum(), state[2])
                torch.testing.assert_close(gradient, torch.tensor([[1.2 / .76, 1.6 / .76, 0.]]),
                                           atol=1e-6, rtol=3e-6)

    def test_state_and_action_gradients_reach_full_fk_without_learner_updates(self):
        learner = self.learner(hidden=())
        state = tuple(value.requires_grad_() for value in recorded(self.motion, [2, 5]))
        objective = learner.frame_features(state)[..., -2:].sum()
        gradients = torch.autograd.grad(objective, state)
        for index, gradient in enumerate(gradients):
            self.assert_finite(gradient, nonzero=index > 0)
        epsilon = .001
        index = self.kin.joint_names.index("left_ankle_roll_joint")
        plus, minus = [value.detach().clone() for value in state], [value.detach().clone() for value in state]
        plus[5][:, index] += epsilon
        minus[5][:, index] -= epsilon
        numerical = (learner.frame_features(plus)[..., -2:].sum()
                     - learner.frame_features(minus)[..., -2:].sum()) / (2 * epsilon)
        torch.testing.assert_close(gradients[5][:, index].sum(), numerical, atol=1e-4, rtol=2e-3)

        S0, S1 = recorded(self.motion, [1, 4]), recorded(self.motion, [2, 5])
        with torch.no_grad():
            learner.discriminator[-1].weight.zero_()
            learner.discriminator[-1].weight[0, -2:] = .02
            learner.discriminator[-1].bias.zero_()
        norm = copy.deepcopy(learner.norm.state_dict())
        action = torch.zeros_like(S1[4], requires_grad=True)
        imagined = (*S1[:4], S1[4] + .03 * action, S1[5] + .4 * action)
        with learner.freeze():
            learner.reward(S0, imagined).sum().backward()
        self.assert_finite(action.grad[:, :12], nonzero=True)
        self.assertTrue(all(parameter.grad is None for parameter in learner.parameters()))
        self.assert_tree_equal(learner.norm.state_dict(), norm)
        self.assertEqual((learner.replay_size, learner._updates), (0, 0))

    def test_constructor_serialization_restore_and_motion_free_scoring_roundtrip(self):
        for history in (2, 10):
            with self.subTest(history=history):
                learner = self.learner(history_steps=history)
                features, starts, ends = learner.sample_expert(4, return_indices=True)
                ids = starts[:, None] + torch.arange(history)
                frames = learner.frame_features(recorded(self.motion, ids), reference=True)
                torch.testing.assert_close(features, frames.flatten(-2), atol=0, rtol=0)
                torch.testing.assert_close(self.motion.clip_id[starts], self.motion.clip_id[ends])
                self.assertEqual(learner.observe_window(frames, torch.ones(4, dtype=torch.bool)), 4)
                stream = BytesIO()
                torch.save(learner.checkpoint(), stream)
                stream.seek(0)
                checkpoint = torch.load(stream, weights_only=True, map_location="cpu")
                self.assertEqual(checkpoint["feature_dim"], 119 * history)
                self.assertEqual(checkpoint["feature_mode"], MODE)
                self.assertEqual(checkpoint["expert_provenance"]["feature_layout"], learner.feature_layout)
                self.assertEqual(checkpoint["expert_provenance"]["expert_hash"], learner.expert_hash)
                restored = self.learner(history_steps=history)
                self.assertEqual(restored.restore(checkpoint),
                                 {"replay_restored": False, "replay_refill_required": True})
                self.assertEqual(restored.replay_size, 0)
                for name in ("discriminator", "norm", "optimizer", "updates", "observed", "expert_hash"):
                    self.assert_tree_equal(restored.checkpoint()[name], checkpoint[name])
                for name in ("optimizer", "updates", "observed"):
                    checkpoint.pop(name)
                with patch.object(AMPStyle, "__init__", side_effect=AssertionError("training allocation")), \
                        patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer allocation")):
                    scorer = AMPStyle.for_scoring(self.kin, checkpoint)
                self.assertEqual(scorer.feature_layout, learner.feature_layout)
                self.assertFalse(scorer.training)
                self.assertTrue(all(not p.requires_grad for p in scorer.parameters()))
                for name in ("optimizer", "_expert_frames", "_expert_starts", "_replay"):
                    self.assertFalse(hasattr(scorer, name))
                torch.testing.assert_close(scorer.frame_features(recorded(self.motion, ids)), frames,
                                           atol=0, rtol=0)
                for other in (restored, scorer):
                    torch.testing.assert_close(other.score(features), learner.score(features), atol=0, rtol=0)
                    torch.testing.assert_close(other.reward_from_features(features),
                                               learner.reward_from_features(features), atol=0, rtol=0)
                self.assertEqual(learner._updates, 0)

    def test_expert_identity_observes_yaw_and_geometry_even_at_rest(self):
        idle = copy.deepcopy(self.motion)
        idle.root_v.zero_()
        idle.root_w.zero_()
        idle.qd.zero_()
        yaw = copy.deepcopy(idle)
        yaw.root_w[:, 2] = .55
        self.assertEqual(self.learner(BASE_MODE, motion=idle).expert_hash,
                         self.learner(BASE_MODE, motion=yaw).expert_hash)
        base = self.learner(motion=idle)
        self.assertNotEqual(base.expert_hash, self.learner(motion=yaw).expert_hash)
        centers, radii = _g1_sole_geometry(self.spec.mjcf_path, "cpu")
        for changed in ((centers + .001, radii), (centers, radii * 1.1)):
            with self.subTest(geometry=changed):
                with patch(GEOMETRY_LOADER, return_value=changed):
                    altered = self.learner(motion=idle)
                    with self.assertRaisesRegex(ValueError, "layout"):
                        AMPStyle.for_scoring(self.kin, base.checkpoint())
                torch.testing.assert_close(altered._expert_frames, base._expert_frames, atol=0, rtol=0)
                self.assertNotEqual(altered.expert_hash, base.expert_hash)
                with self.assertRaisesRegex(ValueError, "expert identity"):
                    altered.restore(base.checkpoint())

    def test_geometry_and_mode_metadata_mismatches_are_rejected(self):
        learner = self.learner(history_steps=10)
        saved = learner.checkpoint()
        geometry = saved["feature_layout"]["sole_speed"]
        for name in geometry:
            with self.subTest(missing=name):
                altered = copy.deepcopy(saved)
                altered["feature_layout"]["sole_speed"].pop(name)
                with self.assertRaisesRegex(ValueError, "layout"):
                    learner.restore(altered)
                with self.assertRaisesRegex(ValueError, "layout"):
                    AMPStyle.for_scoring(self.kin, altered)
        for name, value in (("feature_dim", 1170), ("feature_mode", BASE_MODE)):
            with self.subTest(name=name):
                altered = copy.deepcopy(saved)
                altered[name] = value
                with self.assertRaisesRegex(ValueError, "layout"):
                    learner.restore(altered)
                with self.assertRaisesRegex(ValueError, "layout"):
                    AMPStyle.for_scoring(self.kin, altered)
        base = self.learner(BASE_MODE, history_steps=10)
        for target, source in ((base, learner), (learner, base)):
            with self.assertRaisesRegex(ValueError, "expert identity"):
                target.restore(source.checkpoint())
        altered = copy.deepcopy(saved)
        altered["discriminator"] = base.checkpoint()["discriminator"]
        with self.assertRaisesRegex(ValueError, "discriminator"):
            AMPStyle.for_scoring(self.kin, altered)

    def test_incompatible_robots_layouts_geometry_and_fk_are_rejected(self):
        checkpoint = self.learner().checkpoint()
        for kind in ("robot", "joints", "feet"):
            with self.subTest(kind=kind):
                kin = copy.copy(self.kin)
                if kind == "robot":
                    kin.robot = replace(self.spec, name="t1")
                elif kind == "joints":
                    kin.joint_names = tuple(reversed(kin.joint_names))
                else:
                    kin.tracked_idx = [i for i in kin.tracked_idx if kin.body_names[i] != FOOT_BODIES["g1"][0]]
                with self.assertRaisesRegex(ValueError, "G1"):
                    self.learner(kin=kin)
                with self.assertRaises(ValueError):
                    AMPStyle.for_scoring(kin, checkpoint)
        centers, radii = _g1_sole_geometry(self.spec.mjcf_path, "cpu")
        malformed = ((centers[:, :3], radii[:, :3]), (centers, radii * 0),
                     (centers * float("nan"), radii), (centers, radii * float("inf")))
        for geometry in malformed:
            with self.subTest(geometry=geometry), patch(GEOMETRY_LOADER, return_value=geometry):
                with self.assertRaisesRegex(ValueError, "four finite collision spheres"):
                    self.learner()
                with self.assertRaisesRegex(ValueError, "four finite collision spheres"):
                    AMPStyle.for_scoring(self.kin, checkpoint)
        for kind in ("shape", "missing", "nonfinite"):
            with self.subTest(kind=kind):
                original = self.kin.forward

                def bad_fk(*state):
                    p, R, v, w = original(*state)
                    if kind == "missing":
                        return p, R
                    return p, R, v[..., :2] if kind == "shape" else v * float("nan"), w

                with patch.object(self.kin, "forward", side_effect=bad_fk):
                    with self.assertRaisesRegex(ValueError, "FK"):
                        self.learner()
                    with self.assertRaisesRegex(ValueError, "FK"):
                        AMPStyle.for_scoring(self.kin, checkpoint)

    def test_old_modes_keep_dimensions_and_roundtrip_without_sole_geometry(self):
        with patch(GEOMETRY_LOADER, side_effect=AssertionError("old mode read sole geometry")):
            for mode, width in (("full", 113), ("proprioceptive", 67), ("postural", 62),
                                ("postural_end_effectors", 116), (BASE_MODE, 117)):
                with self.subTest(mode=mode):
                    learner = self.learner(mode)
                    self.assertEqual(learner.frame_dim, width)
                    self.assertNotIn("sole_speed", learner.feature_layout)
                    self.assertEqual(learner.feature_layout["version"], 2 if mode == "proprioceptive" else 1)
                    scorer = AMPStyle.for_scoring(self.kin, learner.checkpoint())
                    restored = self.learner(mode)
                    restored.restore(learner.checkpoint())
                    state = recorded(self.motion, [0, 1])
                    for other in (scorer, restored):
                        torch.testing.assert_close(other.features(state, state), learner.features(state, state),
                                                   atol=0, rtol=0)
            default = AMPStyle(self.kin, self.motion, .76, hidden=(16, 8), replay_capacity=16)
            self.assertEqual(default.feature_mode, "full")

    def test_compiled_features_support_scalar_sequence_and_full_state_gradients(self):
        # Isolate model and batch-rank variants from other compilation tests.
        self.addCleanup(torch._dynamo.reset)
        torch._dynamo.reset()
        learner = self.learner().requires_grad_(False)
        compiled = torch.compile(learner.features, backend="eager", fullgraph=True)
        left, right = recorded(self.motion, [0, 3]), recorded(self.motion, [1, 4])
        for S0, S1 in ((left, right),
                       (tuple(v[0] for v in left), tuple(v[0] for v in right)),
                       (tuple(v[None].expand(2, *v.shape) for v in left),
                        tuple(v[None].expand(2, *v.shape) for v in right))):
            actual = compiled(S0, S1)
            torch.testing.assert_close(actual, learner.features(S0, S1), atol=0, rtol=0)
        w = right[3].clone().requires_grad_()
        state = (*right[:3], w, *right[4:])
        actual = torch.autograd.grad(compiled(left, state)[..., -2:].sum(), w)[0]
        expected = torch.autograd.grad(learner.features(left, state)[..., -2:].sum(), w)[0]
        self.assert_finite(actual, nonzero=True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class AMPSoleSpeedTests(_AMPSoleSpeedContracts, unittest.TestCase):
    """All core contracts always run against the self-contained fixture."""

    model_kind = "synthetic_g1_layout"

    @classmethod
    def model_path(cls, spec):
        directory = cls.enterClassContext(tempfile.TemporaryDirectory(prefix="amp-sole-g1-"))
        path = Path(directory) / "synthetic_g1.xml"
        synthetic_g1_xml(spec).write(path, encoding="ascii", xml_declaration=True)
        return path


class AMPRealG1SoleSpeedTests(_AMPSoleSpeedContracts, unittest.TestCase):
    """Additional source-model coverage when the local G1 assets are available."""

    model_kind = "source_g1_mjcf"

    @classmethod
    def model_path(cls, spec):
        path = Path(spec.mjcf_path)
        if not path.is_file():
            raise unittest.SkipTest("Source G1 MJCF unavailable; synthetic G1 contracts run separately")
        return path


if __name__ == "__main__":
    unittest.main()
