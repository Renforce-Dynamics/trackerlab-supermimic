"""Velocity/dodge timing, geometry, outcomes and differentiable respawns."""

import ast
from dataclasses import replace
import math
from pathlib import Path
import sys
from types import SimpleNamespace, MethodType
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.go2_velocity_dodge_cfg import JumpDodgeCfg, VelocityDodgeCfg
from trackerLab.algorithms.supermimic.velocity_buffer import VelocityReplayBuffer
from trackerLab.tracker_env.mdp.supermimic import flatten_state
import trackerLab.tracker_env.mdp.supermimic_velocity_dodge as signals
from trackerLab.kinematics.supermimic.rotations import quat_to_mat


def state(batch=1, dtype=torch.float32):
    p = torch.zeros(batch, 3, dtype=dtype)
    p[:, 2] = .29
    return (p, torch.eye(3, dtype=dtype).repeat(batch, 1, 1), torch.zeros(batch, 3, dtype=dtype),
            torch.zeros(batch, 3, dtype=dtype), torch.zeros(batch, 12, dtype=dtype),
            torch.zeros(batch, 12, dtype=dtype))


class Kinematics:
    robot = get_robot("go2")
    q_lo, q_hi = torch.full((12,), -3.), torch.full((12,), 3.)

    def forward(self, p, R, v, w, q, qd):
        n = len(self.robot.tracked_bodies)
        return (p.unsqueeze(-2).expand(*p.shape[:-1], n, 3),
                R.unsqueeze(-3).expand(*R.shape[:-2], n, 3, 3),
                v.unsqueeze(-2).expand(*v.shape[:-1], n, 3),
                w.unsqueeze(-2).expand(*w.shape[:-1], n, 3))


class Motion:
    def __init__(self, frames=8, dtype=torch.float32):
        self.q = torch.zeros(frames, 12, dtype=dtype)
        self.root_R = torch.eye(3, dtype=dtype).repeat(frames, 1, 1)
        self.clip_start = torch.tensor([0])
        self.clip_end = torch.tensor([frames - 1])
        self.clip_id = torch.zeros(frames, dtype=torch.long)

    def goal(self, f, frames):
        return self.q.new_zeros((*f.shape, 39 * frames))


def mdp(dtype=torch.float32):
    return signals.VelocityDodgeMDP(
        Kinematics(), Motion(dtype=dtype), torch.zeros(12, dtype=dtype), obs_mode="relative_orientation")


def simulator_class():
    """Execute the actual simulator methods with inert physics dependencies."""
    path = REPO / "source/trackerLab/trackerLab/tracker_env/supermimic/velocity_dodge.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VelocityDodgeSim")
    scope = dict(vars(signals), torch=torch, flatten_state=flatten_state, TrackingSim=object)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    return scope["VelocityDodgeSim"]


def fake_simulator(eval_mode=False, termination="none"):
    cls = simulator_class()
    env = cls.__new__(cls)
    env.num_envs, env.device, env.eval_mode = 1, "cpu", eval_mode
    env.task, env.mdp, env.motion = VelocityDodgeCfg(), mdp(), Motion()
    env._S = state()
    env.state = lambda: env._S
    env.f, env.motion_id = torch.tensor([0]), torch.tensor([0])
    env.xi, env.a_prev = torch.zeros(1, 16), torch.zeros(1, 12)
    env.xi[:, 0] = 1
    env.default_q, env.action_scale, env.joint_ids = torch.zeros(12), torch.full((12,), .25), list(range(12))
    env.done_eval, env.failed_eval = torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
    env.env_origins = torch.tensor([[12., -8., 0.]])
    env.scenario = signals.make_scenario(env._S, 1., torch.tensor([1]))
    env.scenario[:, 8], env.scenario[:, 6] = 2, .7
    env.scenario[:, 3:6] = torch.tensor([[4., 4., .12]])
    env.ep_step, env.ep_return = torch.zeros(1, dtype=torch.int32), torch.zeros(1)
    env.max_episode_steps = 1 if termination == "timeout" else 800
    env.physics_dt, env.control_decimation = .005, 4
    for name in ("encounter_collided", "encounter_passed", "collided", "ever_fallen"):
        setattr(env, name, torch.zeros(1, dtype=torch.bool))
    env.encounters, env.successes = torch.zeros(1), torch.zeros(1)
    env.last_crossing_time = torch.zeros(1)
    env.reset_calls = 0
    bar_data = SimpleNamespace(root_link_pos_w=torch.zeros(1, 3),
                               root_link_quat_w=torch.tensor([[1., 0, 0, 0]]))
    bar = SimpleNamespace(data=bar_data, velocity=None, poses=[], velocities=[])

    def write_pose(pose, env_ids=None):
        bar.poses.append(pose.clone())
        bar_data.root_link_pos_w = pose[:, :3].clone()
        bar_data.root_link_quat_w = pose[:, 3:].clone()

    def write_velocity(velocity):
        bar.velocity = velocity.clone()
        bar.velocities.append(velocity.clone())

    bar.write_root_pose_to_sim, bar.write_root_velocity_to_sim = write_pose, write_velocity
    contact = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(1, 1, 3)),
                              reset=lambda env_ids: None)

    class Scene(dict):
        def write_data_to_sim(self):
            pass

        def update(self, dt):
            pass

    env.scene = Scene(bar=bar, bar_contact=contact)
    env.robot = SimpleNamespace(set_joint_position_target=lambda target, joint_ids: None)

    def physics_step(render=False):
        env._S[0][:, 0] += .001
        if termination == "fall":
            env._S[0][:, 2] = .09

    env.sim = SimpleNamespace(step=physics_step)

    def reset(self, ids):
        if ids.numel():
            self.reset_calls += 1
            self.a_prev[ids] = 0
            self.scenario[ids] = -99
            self.ep_step[ids] = 0
            self.ep_return[ids] = 0
            self.encounters[ids] = 0
            self.successes[ids] = 0
            self.last_crossing_time[ids] = 0

    env._reset = MethodType(reset, env)
    return env


class VelocityTaskTests(unittest.TestCase):
    def test_config_ranges_and_bound(self):
        cfg = VelocityDodgeCfg()
        self.assertAlmostEqual(cfg.max_step_reward, .1)
        self.assertAlmostEqual(cfg.value_bound(.99), 10.)
        self.assertEqual(cfg.difficulty(0), 0.)
        self.assertEqual(cfg.difficulty(6300), 1.)
        for changes in (
            {"vx_range": (1, -1)}, {"control_dt": .01}, {"bar_speed_range": (-1, 2)},
            {"period_range": (.1, 2)}, {"low_bar_range": (.01, .1)},
            {"high_bar_range": (float("nan"), .4)}, {"height_weight": -1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(cfg, **changes)

    def test_sampling_ranges_and_horizontal_sectors(self):
        cfg, S = VelocityDodgeCfg(), state(2048)
        seeds = torch.arange(1, 2049)
        for difficulty in (0., .5, 1.):
            c = signals.make_scenario(S, difficulty, seeds, cfg)
            self.assertTrue(bool(torch.isfinite(c).all()))
            self.assertTrue(bool(((c[:, 0] >= -.6) & (c[:, 0] <= 1.5)).all()))
            self.assertTrue(bool((c[:, 1].abs() <= difficulty * .8 + 1e-6).all()))
            self.assertTrue(bool((c[:, 2].abs() <= difficulty * 1.5 + 1e-6).all()))
            if difficulty == 0:
                torch.testing.assert_close(c[:, :3], torch.tensor([.83, 0., 0.]).expand(2048, 3))
                self.assertTrue(bool((c[:, 8] == 0).all()))
            if difficulty == 1:
                self.assertEqual(set(c[:, 8].tolist()), {0., 1., 2.})
                self.assertTrue(bool(((c[:, 7] >= 1.8) & (c[:, 7] <= 3.6)).all()))
                self.assertTrue(bool(((c[:, 10] >= 2.2-1e-5) & (c[:, 10] <= 3.+1e-5)).all()))
                self.assertTrue(bool(((c[c[:, 8] == 1, 5] >= .31) & (c[c[:, 8] == 1, 5] <= .39)).all()))
                self.assertTrue(bool(((c[c[:, 8] == 2, 5] >= .08) & (c[c[:, 8] == 2, 5] <= .18)).all()))
                sectors = torch.floor(torch.remainder(c[:, 6] + math.pi, 2*math.pi) / (math.pi/4))
                self.assertEqual(len(sectors.unique()), 8)

    def test_bar_pose_geometry_and_normal_crossing_rotate_together(self):
        S = state(dtype=torch.float64)
        S[2][:, 0] = 1
        c = signals.make_scenario(S, 1., torch.tensor([1]))
        c[:, 3:6] = torch.tensor([.4, 0, .35], dtype=c.dtype)
        c[:, 6:10] = torch.tensor([0, 2., 1., 0.], dtype=c.dtype)
        expected = signals.crossing_state(S, c)
        self.assertAlmostEqual(expected[2].item(), .4 / 3)
        for angle in (0., math.pi/2, math.pi, -.7):
            H = signals.yaw_rotation(torch.tensor([angle], dtype=c.dtype))
            Sr = tuple((H @ x.unsqueeze(-1)).squeeze(-1) if index in (0, 2, 3)
                       else H @ x if index == 1 else x for index, x in enumerate(S))
            cr = c.clone()
            cr[:, 3:6] = (H @ c[:, 3:6].unsqueeze(-1)).squeeze(-1)
            cr[:, 6] += angle
            pose = signals.bar_pose(cr)
            center, _, rotation = signals.scenario_state(cr)
            torch.testing.assert_close(pose[:, :3], center)
            torch.testing.assert_close(quat_to_mat(pose[:, 3:]), rotation)
            for x, y in zip(signals.crossing_state(Sr, cr), expected):
                torch.testing.assert_close(x, y)
            torch.testing.assert_close(
                mdp(torch.float64).collision_probability(Sr, cr), mdp(torch.float64).collision_probability(S, c))
            torch.testing.assert_close(
                mdp(torch.float64).desired_height(Sr, cr), mdp(torch.float64).desired_height(S, c))

    def test_reward_bound_command_frame_and_gradients(self):
        torch.manual_seed(24)
        model, S = mdp(), state(128)
        c = signals.make_scenario(S, 1., torch.arange(128))
        R = signals.yaw_rotation(torch.linspace(-math.pi, math.pi, 128))
        body_v = torch.cat((c[:, :2], torch.zeros(128, 1)), -1)
        body_w = torch.stack((torch.zeros(128), torch.zeros(128), c[:, 2]), -1)
        S = (S[0].requires_grad_(), R, (R @ body_v.unsqueeze(-1)).squeeze(-1),
             (R @ body_w.unsqueeze(-1)).squeeze(-1), *S[4:])
        ev, ew = model.command_error(S, c)
        torch.testing.assert_close(ev, torch.zeros_like(ev), atol=2e-6, rtol=0)
        torch.testing.assert_close(ew, torch.zeros_like(ew))
        a = torch.randn(128, 12, requires_grad=True)
        r, terms = model.task_reward(S, torch.zeros(128, dtype=torch.long), a, torch.zeros_like(a), c, terms=True)
        self.assertTrue(bool((r <= model.task_cfg.max_step_reward + 1e-7).all()))
        torch.testing.assert_close(r, sum(terms.values()))
        r.sum().backward()
        self.assertTrue(bool(torch.isfinite(S[0].grad).all()))
        self.assertGreater(a.grad.abs().sum().item(), 0)
        self.assertEqual(model.task_features(S, c).shape, (128, 24))

    def test_jump_reward_terms_match_reward_and_keep_original_upper_bound(self):
        cfg = JumpDodgeCfg()
        model = signals.VelocityDodgeMDP(
            Kinematics(), Motion(), torch.zeros(12), obs_mode="relative_orientation", task_cfg=cfg)
        S = state(32)
        c = signals.make_scenario(S, 0.0, torch.arange(1, 33), cfg)
        action = torch.zeros(32, 12)
        reward, terms = model.task_reward(S, torch.zeros(32, dtype=torch.long), action,
                                           action, c, terms=True)
        self.assertEqual(tuple(terms), signals.TERMS)
        torch.testing.assert_close(reward, sum(terms.values()))
        self.assertTrue(bool((reward <= 0.1 + 1e-7).all()))
        self.assertAlmostEqual(cfg.value_bound(.99), 10.0)
        self.assertTrue(bool(torch.isfinite(terms["jump_clearance"]).all()))

    def test_jump_height_interpolates_to_absolute_clearance_target(self):
        cfg = JumpDodgeCfg()
        model = signals.VelocityDodgeMDP(
            Kinematics(), Motion(), torch.zeros(12), obs_mode="relative_orientation", task_cfg=cfg)
        S = state(301)
        c = signals.make_scenario(S, 0.0, torch.arange(1, 302), cfg)
        c[:, 3] = torch.linspace(-1.0, 2.0, len(c))
        c[:, 4], c[:, 6], c[:, 7], c[:, 9] = 0, 0, 1, 0
        for bar_height in (cfg.initial_low_bar_height, cfg.low_bar_range[1]):
            with self.subTest(bar_height=bar_height):
                c[:, 5] = bar_height
                height = model.desired_height(S, c)
                absolute_target = bar_height + cfg.bar_half_thickness + cfg.normal_height + 0.08
                self.assertTrue(bool((height >= cfg.normal_height).all()))
                self.assertTrue(bool((height <= absolute_target + 1e-6).all()))
                self.assertGreater(height.max().item(), absolute_target - 0.025)
                self.assertAlmostEqual(height[-1].item(), cfg.normal_height, delta=0.006)
                self.assertAlmostEqual(height[0].item(), cfg.normal_height, delta=0.006)
        c[:, 8] = 0
        torch.testing.assert_close(model.desired_height(S, c), c[:, 12])

    def test_encounter_boundary_keeps_old_reward_pose_and_robot_state(self):
        S = state(dtype=torch.float64)
        original = tuple(x.clone() for x in S)
        c = signals.make_scenario(S, 1., torch.tensor([1]))
        c[:, 9] = c[:, 10] - .02
        before = c.clone()
        end, following, changed = signals.advance_scenario(c, S)
        self.assertTrue(changed.item())
        self.assertEqual(end[0, 14].item(), 0)
        self.assertEqual(following[0, 14].item(), 1)
        self.assertEqual(following[0, 9].item(), 0)
        torch.testing.assert_close(end, signals.offset_scenario(c, .02))
        torch.testing.assert_close(c, before)
        for x, y in zip(S, original):
            torch.testing.assert_close(x, y)
        for x, y in zip(signals.advance_scenario(c, S), (end, following, changed)):
            torch.testing.assert_close(x, y)

    def test_respawn_translation_and_yaw_gradients_match_finite_difference(self):
        base = state(dtype=torch.float64)
        c = signals.make_scenario(base, 1., torch.tensor([1]))
        c[:, 9] = c[:, 10] - .02

        def features(translation, yaw):
            p = base[0] + torch.stack((translation, translation * 0, translation * 0))[None]
            R = signals.yaw_rotation(yaw[None])
            S = (p, R, *base[2:])
            _, following, changed = signals.advance_scenario(c, S)
            self.assertTrue(changed.item())
            center, _, rotation = signals.scenario_state(following)
            relative = center - p
            local = (R.transpose(-1, -2) @ relative.unsqueeze(-1)).squeeze(-1)
            return torch.stack((relative[0, 0], relative[0, 1], local[0, 0], rotation[0, 0, 0]))

        x = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
        yaw = torch.tensor(.27, dtype=torch.float64, requires_grad=True)
        values = features(x, yaw)
        epsilon = 1e-5
        for index, value in enumerate(values):
            gx, gy = torch.autograd.grad(value, (x, yaw), retain_graph=True)
            fdx = (features(x.detach()+epsilon, yaw.detach())[index] -
                   features(x.detach()-epsilon, yaw.detach())[index]) / (2*epsilon)
            fdy = (features(x.detach(), yaw.detach()+epsilon)[index] -
                   features(x.detach(), yaw.detach()-epsilon)[index]) / (2*epsilon)
            torch.testing.assert_close(gx, fdx, atol=2e-7, rtol=2e-6)
            torch.testing.assert_close(gy, fdy, atol=2e-7, rtol=2e-6)
        self.assertAlmostEqual(torch.autograd.grad(values[0], x)[0].item(), 0., places=10)

    def test_inclusive_gait_end_is_visited_and_wrapped(self):
        motion = Motion(frames=3)
        f = torch.tensor([0])
        observed = []
        for _ in range(7):
            observed.append(f.item())
            f = signals.next_gait_frame(f, motion)
        self.assertEqual(observed, [0, 1, 2, 0, 1, 2, 0])

    def test_scenario_replay_keeps_encounter_changes_inside_an_episode(self):
        buffer = VelocityReplayBuffer(6, 1, "cpu")
        for index in range(10):
            row = {key: torch.zeros_like(value[0]) for key, value in buffer.data.items()}
            row["ep_step"].fill_(index % 7)
            row["scenario"][:, 9] = index
            row["scenario1"][:, 9] = index + 1
            row["scenario"][:, 14] = index // 2
            row["done"].fill_(index == 6)
            buffer.add(row)
        start, history = buffer.sample_starts(256, 4)
        self.assertTrue(bool((history["scenario"][:, -1, 9] == start["scenario"][:, 9]).all()))
        self.assertTrue(bool((history["scenario"][:, 0, 9] >= 4).all()))
        reset = start["scenario"][:, 9] == 7
        self.assertTrue(reset.any())
        self.assertTrue(bool((history["scenario"][reset, :, 9] == 7).all()))
        across_bar = start["scenario"][:, 9] == 6
        self.assertTrue(bool((history["scenario"][across_bar, 0, 14] <
                              history["scenario"][across_bar, -1, 14]).all()))
        windows = buffer.sample_windows(256, 3)
        torch.testing.assert_close(windows["scenario1"][:, :, 9], windows["scenario"][:, :, 9] + 1)
        terminal = windows["done"][:, 0]
        self.assertTrue(terminal.any())
        torch.testing.assert_close(windows["mask"][terminal], torch.tensor([1., 0., 0.]).expand(int(terminal.sum()), 3))

    def test_simulator_terminal_snapshot_and_explicit_bar_drive(self):
        env = fake_simulator(termination="fall")
        original_s, original_c = flatten_state(env.state()), env.scenario.clone()
        out = env.step(torch.full((1, 12), .7))
        torch.testing.assert_close(out["s"], original_s)
        torch.testing.assert_close(out["scenario"], original_c)
        torch.testing.assert_close(out["a"], torch.full((1, 12), .7))
        self.assertTrue(out["terminated"].item())
        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(out["bar_pose_error"].item(), 0.)
        self.assertLess(out["bar_rotation_error"].item(), 1e-6)
        bar = env.scene["bar"]
        self.assertEqual(len(bar.velocities), 4)
        for index, (pose, velocity) in enumerate(zip(bar.poses, bar.velocities)):
            c = signals.offset_scenario(original_c, (index+1)*.005)
            torch.testing.assert_close(pose[:, :3] - env.env_origins, signals.bar_pose(c)[:, :3])
            torch.testing.assert_close(pose[:, 3:], signals.bar_pose(c)[:, 3:])
            torch.testing.assert_close(velocity[:, :3], signals.scenario_state(c)[1])

    def test_fall_and_timeout_finalize_one_failed_attempt(self):
        for cause in ("fall", "timeout"):
            with self.subTest(cause=cause):
                env = fake_simulator(eval_mode=True, termination=cause)
                out = env.step(torch.zeros(1, 12))
                self.assertTrue(out["done"].item())
                self.assertTrue(out["encounter_end"].item())
                self.assertFalse(out["encounter_natural_end"].item())
                self.assertTrue(out["encounter_failure"].item())
                self.assertFalse(out["encounter_success"].item())
                self.assertEqual(out["encounter_timeout"].item(), cause == "timeout")
                self.assertEqual(out["encounter_fall"].item(), cause == "fall")
                self.assertEqual(out["encounters"].item(), 1)
                self.assertEqual(out["successes"].item(), 0)
                second = env.step(torch.zeros(1, 12))
                self.assertFalse(second["encounter_end"].item())
                self.assertEqual(second["encounters"].item(), 1)

    def test_natural_success_requires_pass_collision_free_and_survival(self):
        for collided, passed in ((False, True), (True, True), (False, False)):
            with self.subTest(collided=collided, passed=passed):
                env = fake_simulator(eval_mode=True)
                env.scenario[:, 6] = 0
                env.scenario[:, 9] = env.scenario[:, 10] - .02
                # Far away in Y keeps the bar clear while avoiding a new pass.
                env.scenario[:, 4] = 20.
                env.encounter_collided.fill_(collided)
                env.encounter_passed.fill_(passed)
                out = env.step(torch.zeros(1, 12))
                self.assertTrue(out["encounter_natural_end"].item())
                self.assertEqual(out["encounter_success"].item(), passed and not collided)
                self.assertFalse(out["done"].item())
                self.assertEqual(env.reset_calls, 0)
                self.assertEqual(env.ep_step.item(), 1)
                self.assertEqual(env.f.item(), 1)

    def test_crossing_interval_uses_episode_clock_and_emits_once_per_bar(self):
        env = fake_simulator(eval_mode=True)
        env.scenario[:, 6] = 0
        env.scenario[:, 3:6] = torch.tensor([[-1., 0., .12]])
        env.scenario[:, 7] = 0
        env.ep_step.fill_(49)
        first = env.step(torch.zeros(1, 12))
        self.assertTrue(first["crossing"].item())
        self.assertFalse(first["crossing_interval_valid"].item())
        self.assertAlmostEqual(env.last_crossing_time.item(), 1.)
        repeat = env.step(torch.zeros(1, 12))
        self.assertFalse(repeat["crossing"].item())
        env.encounter_passed.fill_(False)
        env.ep_step.fill_(79)
        second = env.step(torch.zeros(1, 12))
        self.assertTrue(second["crossing_interval_valid"].item())
        self.assertAlmostEqual(second["crossing_interval_s"].item(), .6, places=5)

    def test_delayed_terminal_attempt_and_expiry_keep_original_event_id(self):
        env = fake_simulator(eval_mode=True, termination="timeout")
        env.scenario[:, 6:8] = 0
        env.scenario[:, 3:6] = torch.tensor([[0., 0., .12]])
        env.scenario[:, 9:11] = torch.tensor([[1.08, 1.]])
        env.scenario[:, 14] = 7
        out = env.step(torch.zeros(1, 12))
        self.assertTrue(out["encounter_deadline_miss"].item())
        self.assertTrue(out["encounter_timeout"].item())
        self.assertFalse(out["encounter_natural_end"].item())
        self.assertEqual(out["encounter_index"].item(), 7)
        self.assertAlmostEqual(out["encounter_clearance_delay_s"].item(), .1, places=5)
        env = fake_simulator(eval_mode=True)
        env.scenario[:, 6:8] = 0
        env.scenario[:, 3:6] = torch.tensor([[1., 0., .12]])
        env.scenario[:, 9] = env.scenario[:, 10] - .02
        env.scenario[:, 14] = 7
        out = env.step(torch.zeros(1, 12))
        self.assertTrue(out["encounter_expired_ahead"].item())
        self.assertTrue(out["encounter_failure"].item())
        self.assertFalse(out["encounter_success"].item())
        self.assertEqual(out["encounter_index"].item(), 7)
        self.assertEqual(out["next_encounter_index"].item(), 8)


if __name__ == "__main__":
    unittest.main()
