"""Rapid-profile configuration, geometry, determinism, and gradients."""

from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerTask.supermimic.go2_velocity_dodge_cfg import (
    JumpDodgeCfg, RapidDodgeCfg, VelocityDodgeCfg, get_task_cfg,
)
from trackerLab.algorithms.supermimic.velocity_buffer import VelocityReplayBuffer
from trackerLab.tracker_env.mdp.supermimic import flatten_state, unflatten_state
import trackerLab.tracker_env.mdp.supermimic_velocity_dodge as signals


def state(batch=1, yaws=None, dtype=torch.float32):
    if yaws is None:
        yaws = torch.zeros(batch, dtype=dtype)
    position = torch.zeros(batch, 3, dtype=dtype)
    position[:, 2] = 0.29
    rotation = signals.yaw_rotation(yaws)
    zeros = torch.zeros(batch, 3, dtype=dtype)
    joints = torch.zeros(batch, 12, dtype=dtype)
    return position, rotation, zeros, zeros, joints, joints


def body_displacement(command, duration):
    vx, vy, wz = command.unbind(-1)
    turn = wz * duration
    sinc = torch.sinc(turn / math.pi)
    cosc = 0.5 * turn * torch.sinc(turn / (2 * math.pi)).square()
    return torch.stack((
        duration * (sinc * vx - cosc * vy),
        duration * (cosc * vx + sinc * vy),
        torch.zeros_like(vx),
    ), -1)


def twist_state(S, c, duration):
    rotation = S[1] @ signals.yaw_rotation(c[..., 2] * duration)
    displacement = (S[1] @ body_displacement(c[..., :3], duration).unsqueeze(-1)).squeeze(-1)
    velocity = (rotation @ torch.cat((
        c[..., :2], torch.zeros_like(c[..., :1]),
    ), -1).unsqueeze(-1)).squeeze(-1)
    angular = torch.stack((torch.zeros_like(c[..., 2]), torch.zeros_like(c[..., 2]),
                           c[..., 2]), -1)
    return S[0] + displacement, rotation, velocity, angular, *S[4:]


class RapidTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_profiles_keep_standard_and_configure_rapid_curriculum(self):
        standard = get_task_cfg()
        rapid = get_task_cfg("rapid")
        self.assertIs(type(standard), VelocityDodgeCfg)
        self.assertIs(type(rapid), RapidDodgeCfg)
        self.assertEqual((standard.initial_high_bar_height, standard.initial_low_bar_height),
                         (0.65, 0.04))
        self.assertEqual((rapid.initial_high_bar_height, rapid.initial_low_bar_height),
                         (0.38, 0.08))
        self.assertEqual(rapid.initial_period, (1.0, 1.2))
        self.assertEqual(rapid.period_range, (0.5, 1.0))
        self.assertEqual((rapid.curriculum_delay, rapid.curriculum_iterations), (100, 4000))
        self.assertEqual(rapid.command_hold_encounters, 4)
        self.assertEqual(rapid.control_dt, 0.02)
        self.assertEqual((rapid.task_feature_dim, rapid.scenario_dim), (24, 15))
        self.assertAlmostEqual(rapid.max_step_reward, standard.max_step_reward)
        self.assertAlmostEqual(rapid.value_bound(0.99), standard.value_bound(0.99))
        with self.assertRaises(ValueError):
            get_task_cfg("unknown")

    def test_standard_profile_matches_pre_profile_scenario_values(self):
        yaw = torch.tensor([0.0, 0.7, -1.2])
        S = state(3, yaw)
        actual = signals.make_scenario(S, 0.4, torch.tensor([1, 2, 3]), get_task_cfg())
        expected = torch.tensor([
            [0.58051372, 0.09052411, 0.23371576, 2.45302033, -0.82292306, 0.09163213,
             -0.66446525, 1.09958088, 2.0, 0.0, 3.86, 0.4, 0.29, 1.0, 0.0],
            [0.93167061, -0.30248129, 0.23398913, 5.04584503, 2.08551836, 0.08982004,
             0.29140386, 2.01885700, 0.0, 0.0, 4.10, 0.4, 0.29, 2.0, 0.0],
            [0.88611126, 0.07217446, 0.35665950, 1.45804286, -3.27326751, 0.05761706,
             -1.34006834, 1.45393050, 2.0, 0.0, 3.52, 0.4, 0.29, 3.0, 0.0],
        ])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)

    def test_config_validates_period_height_speed_and_control_rate(self):
        rapid = RapidDodgeCfg()
        for changes in (
            {"period_range": (0.49, 1.0)},
            {"initial_period": (0.49, 1.2)},
            {"initial_low_bar_height": 0.02},
            {"initial_high_bar_height": 0.07, "initial_low_bar_height": 0.08},
            {"control_dt": 0.01},
            {"vy_max": 2.0},
            {"bar_speed_range": (6.0, 7.1)},
            {"lead_fraction": 0.3},
            {"command_hold_encounters": 0},
            {"command_hold_encounters": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(rapid, **changes)
        with self.assertRaises(ValueError):
            replace(VelocityDodgeCfg(), initial_high_bar_height=float("nan"))
        with self.assertRaises(ValueError):
            replace(VelocityDodgeCfg(), period_range=(0.5, 1.0))

    def test_jump_profile_is_front_only_low_and_preserves_reward_ceiling(self):
        cfg = get_task_cfg("jump")
        self.assertIsInstance(cfg, JumpDodgeCfg)
        self.assertTrue(cfg.jump_only)
        self.assertTrue(cfg.front_only)
        self.assertEqual(cfg.max_step_reward, VelocityDodgeCfg().max_step_reward)
        samples = signals.make_scenario(state(1024), 0.0, torch.arange(1, 1025), cfg)
        self.assertTrue(bool((samples[:, 8] == 2).all()))
        self.assertTrue(bool((samples[:, 6].abs() < 1e-7).all()))
        self.assertTrue(bool(((samples[:, 5] >= cfg.initial_low_bar_height) &
                              (samples[:, 5] <= cfg.initial_low_bar_height)).all()))
        self.assertLessEqual(cfg.value_bound(.99), VelocityDodgeCfg().value_bound(.99))
        progressed = signals.make_scenario(state(1024), 1.0, torch.arange(1, 1025), cfg)
        sectors = torch.floor(torch.remainder(progressed[:, 6] + math.pi, 2 * math.pi) /
                              (math.pi / 4))
        self.assertEqual(set(sectors.tolist()), set(range(8)))

    def test_jump_clearance_signal_and_measurement_use_kinematic_feet_and_calves(self):
        motion = SimpleNamespace(q=torch.zeros(8, 12), clip_id=torch.zeros(8, dtype=torch.long),
                                 clip_start=torch.tensor([0]), clip_end=torch.tensor([7]))
        model, S = signals.VelocityDodgeMDP(
            type("Kin", (), {
                "robot": __import__("trackerTask.supermimic.robots", fromlist=["get_robot"]).get_robot("go2"),
                "q_lo": torch.full((12,), -3.), "q_hi": torch.full((12,), 3.),
                "forward": lambda self, p, R, v, w, q, qd: (
                    p.unsqueeze(-2).expand(*p.shape[:-1], len(self.robot.tracked_bodies), 3),
                    R.unsqueeze(-3).expand(*R.shape[:-2], len(self.robot.tracked_bodies), 3, 3),
                    v.unsqueeze(-2).expand(*v.shape[:-1], len(self.robot.tracked_bodies), 3),
                    w.unsqueeze(-2).expand(*w.shape[:-1], len(self.robot.tracked_bodies), 3),
                ),
            })(), motion, torch.zeros(12), obs_mode="relative_orientation", task_cfg=JumpDodgeCfg()), state()
        c = torch.tensor([[.3, 0., 0., 0., 0., .12, 0., 1., 2., 0., 2.5, 0., .29, 1., 0.]])
        reward = model.jump_clearance(S, c)
        self.assertTrue(torch.isfinite(reward).all())
        self.assertGreaterEqual(reward.item(), 0.)
        clearance, valid = model.jump_clearance_metric(S, c)
        self.assertTrue(torch.isfinite(clearance).all())
        self.assertTrue(bool(valid.all()))
        self.assertAlmostEqual(model.task_cfg.max_step_reward, .1)

    def test_rapid_initial_bars_and_curriculum_interpolate_configured_heights(self):
        cfg = RapidDodgeCfg()
        S = state(1024, dtype=torch.float64)
        for iteration, difficulty in ((0, 0.0), (100, 0.0), (2100, 0.5), (4100, 1.0)):
            with self.subTest(iteration=iteration):
                self.assertEqual(cfg.difficulty(iteration), difficulty)
                c = signals.make_scenario(S, difficulty, torch.arange(1, 1025), cfg)
                self.assertEqual(set(c[:, 8].tolist()), {1.0, 2.0})
                for mode, initial, final in (
                    (1, 0.38, cfg.high_bar_range), (2, 0.08, cfg.low_bar_range),
                ):
                    heights = c[c[:, 8] == mode, 5]
                    lower = initial + difficulty * (final[0] - initial)
                    upper = initial + difficulty * (final[1] - initial)
                    self.assertTrue(bool((heights >= lower - 1e-12).all()))
                    self.assertTrue(bool((heights <= upper + 1e-12).all()))
                period_min = cfg.initial_period[0] + difficulty * (
                    cfg.period_range[0] - cfg.initial_period[0])
                period_max = cfg.initial_period[1] + difficulty * (
                    cfg.period_range[1] - cfg.initial_period[1])
                self.assertTrue(bool((c[:, 10] >= period_min - cfg.control_dt / 2).all()))
                self.assertTrue(bool((c[:, 10] <= period_max + cfg.control_dt / 2).all()))
                self.assertTrue(bool((c[:, 7] <= cfg.max_bar_speed).all()))
        standard = signals.make_scenario(S, 0.0, torch.arange(1, 1025), VelocityDodgeCfg())
        self.assertTrue(bool((standard[:, 8] == 0).all()))

    def test_rapid_difficulty_one_periods_geometry_and_speed_bound(self):
        count = 8096
        yaws = torch.linspace(-math.pi, math.pi, count)
        S = state(count, yaws)
        cfg = RapidDodgeCfg()
        scenario = signals.make_scenario(S, 1.0, torch.arange(1, count + 1), cfg)
        self.assertTrue(bool(torch.isfinite(scenario).all()))
        self.assertEqual(set(scenario[:, 8].tolist()), {1.0, 2.0})
        self.assertGreaterEqual(scenario[:, 10].min().item(), 0.5)
        self.assertLessEqual(scenario[:, 10].max().item(), 1.0)
        self.assertLessEqual(scenario[:, 7].max().item(), cfg.max_bar_speed)
        self.assertTrue(bool(((scenario[scenario[:, 8] == 1, 5] >= 0.31) &
                              (scenario[scenario[:, 8] == 1, 5] <= 0.39)).all()))
        self.assertTrue(bool(((scenario[scenario[:, 8] == 2, 5] >= 0.08) &
                              (scenario[scenario[:, 8] == 2, 5] <= 0.18)).all()))

        normal = signals.yaw_rotation(scenario[:, 6])[:, :, 0]
        lead = cfg.lead_fraction * scenario[:, 10]
        D_lead = (S[1] @ body_displacement(scenario[:, :3], lead).unsqueeze(-1)).squeeze(-1)
        D_end = (S[1] @ body_displacement(scenario[:, :3], scenario[:, 10]).unsqueeze(-1)).squeeze(-1)
        start_distance = ((scenario[:, 3:6] - S[0]) * normal).sum(-1)
        end_bar = scenario[:, 3:6] - scenario[:, 7:8] * scenario[:, 10:11] * normal
        end_root = S[0] + D_end
        end_distance = ((end_bar - end_root) * normal).sum(-1)
        lead_bar = signals.scenario_state(signals.offset_scenario(scenario, lead))[0]
        torch.testing.assert_close(lead_bar[:, :2], (S[0] + D_lead)[:, :2], atol=2e-6, rtol=0)
        self.assertTrue(bool((start_distance >= cfg.respawn_clearance + 0.3 - 2e-5).all()))
        self.assertTrue(bool((end_distance <= -(cfg.respawn_clearance + 0.1) + 2e-5).all()))
        # The signed normal distance changes sign within the requested period.
        self.assertTrue(bool((start_distance > 0).all() and (end_distance < 0).all()))
        S_before = twist_state(S, scenario, scenario[:, 10] - cfg.control_dt)
        _, _, early = signals.advance_scenario(
            signals.offset_scenario(scenario, scenario[:, 10] - 2 * cfg.control_dt),
            S_before, cfg)
        self.assertFalse(bool(early.any()))
        S_end = twist_state(S, scenario, scenario[:, 10])
        end, following, changed = signals.advance_scenario(
            signals.offset_scenario(scenario, scenario[:, 10] - cfg.control_dt), S_end, cfg)
        self.assertTrue(bool(changed.all()))
        torch.testing.assert_close(end[:, 9], scenario[:, 10])
        torch.testing.assert_close(following[:, 14], torch.ones(count))

    def test_short_period_endpoints_and_multisector_commands(self):
        count = 2048
        headings = torch.linspace(-math.pi, math.pi, count, dtype=torch.float64)
        S = state(count, headings, dtype=torch.float64)
        seeds = torch.arange(20, 20 + count)
        for period in (0.5, 1.0):
            cfg = replace(RapidDodgeCfg(), initial_period=(period, period),
                          period_range=(period, period))
            scenario = signals.make_scenario(S, 1.0, seeds, cfg)
            torch.testing.assert_close(scenario[:, 10], torch.full((count,), period, dtype=S[0].dtype))
            relative_angle = scenario[:, 6] - headings
            sectors = torch.floor((relative_angle + math.pi) / (math.pi / 4)).long()
            self.assertEqual(set(sectors.tolist()), set(range(8)))
            self.assertEqual(set(scenario[:, 8].tolist()), {1.0, 2.0})
            self.assertTrue(bool((scenario[:, 7] <= 7).all()))
            normals = signals.yaw_rotation(scenario[:, 6])[:, :, 0]
            start = ((scenario[:, 3:6] - S[0]) * normals).sum(-1)
            end_scenario = signals.offset_scenario(scenario, period)
            S_end = twist_state(S, scenario, torch.full((count,), period, dtype=S[0].dtype))
            end = ((signals.scenario_state(end_scenario)[0] - S_end[0]) * normals).sum(-1)
            self.assertTrue(bool((start >= 1.1 - 1e-12).all()))
            self.assertTrue(bool((end <= -0.9 + 1e-12).all()))

    def test_commands_hold_four_encounters_bars_remain_per_encounter(self):
        cfg = RapidDodgeCfg()
        S = state(8)
        c = torch.zeros(8, cfg.scenario_dim)
        c[:, 11] = 1.0
        c[:, 13] = 71
        c[:, 14] = torch.arange(8)
        scenarios = signals._spawn_encounter(S, c, cfg)
        torch.testing.assert_close(scenarios[:4, :3], scenarios[:1, :3].expand(4, 3))
        torch.testing.assert_close(scenarios[4:, :3], scenarios[4:5, :3].expand(4, 3))
        self.assertFalse(torch.equal(scenarios[0, :3], scenarios[4, :3]))
        self.assertGreater(scenarios[:4, 6].unique().numel(), 1)
        self.assertGreater(scenarios[:4, 10].unique().numel(), 1)
        torch.testing.assert_close(signals._spawn_encounter(S, c, cfg), scenarios, atol=0, rtol=0)
        for index in range(8):
            actual = signals._spawn_encounter(tuple(x[index:index + 1] for x in S),
                                             c[index:index + 1], cfg)
            torch.testing.assert_close(actual, scenarios[index:index + 1], atol=0, rtol=0)
        standard = signals._spawn_encounter(S, c, VelocityDodgeCfg())
        self.assertEqual(standard[:, :3].unique(dim=0).shape[0], 8)

    def test_two_respawn_boundaries_are_replayable_and_clear_gated(self):
        cfg = RapidDodgeCfg()
        S = state(dtype=torch.float64)
        c0 = signals.make_scenario(S, 1.0, torch.tensor([313]), cfg)
        c0[:, 9] = c0[:, 10] - cfg.control_dt
        end0, c1, changed0 = signals.advance_scenario(c0, S, cfg)
        self.assertTrue(changed0.item())
        self.assertEqual(c1[0, 14].item(), 1)
        c1 = c1.clone()
        c1[:, 9] = c1[:, 10] - cfg.control_dt
        _, c2, changed1 = signals.advance_scenario(c1, S, cfg)
        self.assertTrue(changed1.item())
        self.assertEqual(c2[0, 14].item(), 2)
        _, replay, replay_changed = signals.advance_scenario(c1, S, cfg)
        self.assertTrue(replay_changed.item())
        torch.testing.assert_close(c2, replay)
        self.assertEqual(end0[0, 14].item(), 0)

        blocked = c1.clone()
        blocked[:, 6] = 0
        blocked[:, 8] = 1
        blocked[:, 9] = blocked[:, 10] - cfg.control_dt
        # At the deadline, this bar still overlaps the root in local X/Y.
        blocked[:, 3:6] = S[0]
        blocked[:, 3] += blocked[:, 7] * blocked[:, 10]
        old_end, same, changed = signals.advance_scenario(blocked, S, cfg)
        self.assertFalse(changed.item())
        self.assertEqual(same[0, 14].item(), blocked[0, 14].item())
        torch.testing.assert_close(same, old_end)

    def test_spawn_difficulty_changes_only_the_new_encounter(self):
        cfg = JumpDodgeCfg()
        S = state(128)
        c = signals.make_scenario(S, 0.0, torch.arange(1, 129), cfg)
        c[:, 9] = c[:, 10] - cfg.control_dt
        normal = signals.yaw_rotation(c[:, 6])[:, :, 0]
        c[:, 3:5] = normal[:, :2] * (c[:, 7:8] * c[:, 10:11] + 2.0)
        _, following, changed = signals.advance_scenario(c, S, cfg, spawn_difficulty=1.0)
        self.assertTrue(bool(changed.all()))
        torch.testing.assert_close(following[:, 11], torch.ones(128))
        torch.testing.assert_close(following[:, 14], torch.ones(128))
        self.assertTrue(bool((following[:, 6].abs() > 0).all()))
        self.assertTrue(bool(((following[:, 5] >= cfg.low_bar_range[0]) &
                              (following[:, 5] <= cfg.low_bar_range[1])).all()))

    def test_constant_twist_rollout_has_repeated_rapid_cycles_and_replay_alignment(self):
        cfg = RapidDodgeCfg()
        count, steps = 64, 160
        anchor = state(count, torch.linspace(-math.pi, math.pi, count))
        S = anchor
        c = signals.make_scenario(S, 1.0, torch.arange(1, count + 1), cfg)
        replay = VelocityReplayBuffer(steps, count, "cpu")
        last_crossing = torch.zeros(count)
        crossed = torch.zeros(count, dtype=torch.bool)
        intervals, periods = [], []
        for index in range(steps):
            age = c[:, 9] + cfg.control_dt
            S1 = twist_state(anchor, c, age)
            end, c1, changed = signals.advance_scenario(c, S1, cfg)
            self.assertTrue(torch.equal(changed, age >= c[:, 10] - 1e-5))
            crossing = ~crossed & (signals.crossing_state(S1, end)[0] <= 0)
            valid_interval = crossing & (last_crossing > 0)
            intervals.extend(((index + 1) * cfg.control_dt -
                              last_crossing[valid_interval]).tolist())
            last_crossing = torch.where(crossing, (index + 1) * cfg.control_dt, last_crossing)
            crossed = (crossed | crossing) & ~changed
            periods.extend(end[changed, 9].tolist())
            for old, updated in zip(S, S1):
                self.assertTrue(bool(torch.isfinite(updated).all()))
            anchor = tuple(torch.where(
                changed.reshape(count, *(1 for _ in x.shape[1:])), x, old,
            ) for x, old in zip(S1, anchor))
            row = {name: torch.zeros_like(value[0]) for name, value in replay.data.items()}
            row.update(s=flatten_state(S), s1=flatten_state(S1), scenario=c, scenario1=c1)
            row["ep_step"].fill_(index)
            replay.add(row)
            S, c = S1, c1
        self.assertTrue(bool((c[:, 14] >= 3).all()))
        self.assertGreater(len(periods), 3 * count)
        self.assertGreater(len(intervals), 2 * count)
        self.assertGreaterEqual(min(periods), 0.5 - 2e-6)
        self.assertLessEqual(max(periods), 1.0 + 2e-6)
        self.assertGreaterEqual(min(intervals), 0.5 - 2e-6)
        self.assertLessEqual(max(intervals), 1.0 + 2e-6)

        windows = replay.sample_windows(256, 4)
        _, recomputed, _ = signals.advance_scenario(
            windows["scenario"], unflatten_state(windows["s1"]), cfg)
        torch.testing.assert_close(windows["scenario1"], recomputed, atol=2e-6, rtol=0)
        torch.testing.assert_close(windows["scenario1"][:, :-1], windows["scenario"][:, 1:],
                                   atol=0, rtol=0)
        self.assertTrue(bool((windows["mask"] == 1).all()))

    def test_rapid_respawn_translation_and_yaw_gradients_match_finite_difference(self):
        cfg = RapidDodgeCfg()
        base = state(dtype=torch.float64)
        original = signals.make_scenario(base, 1.0, torch.tensor([811]), cfg)
        original[:, 9] = original[:, 10] - cfg.control_dt

        def features(translation, yaw):
            p = base[0] + torch.stack((
                translation, 0.3 * translation, torch.zeros_like(translation)
            )).unsqueeze(0)
            R = signals.yaw_rotation(yaw.reshape(1))
            S = (p, R, *base[2:])
            _, following, changed = signals.advance_scenario(original, S, cfg)
            self.assertTrue(changed.item())
            features_per_boundary = []
            for boundary in range(2):
                normal = signals.yaw_rotation(following[:, 6])[:, :, 0]
                relative = following[:, 3:6] - S[0]
                local = (S[1].transpose(-1, -2) @ relative.unsqueeze(-1)).squeeze(-1)
                features_per_boundary.append(torch.stack((
                    local[0, 0], local[0, 1], following[0, 7], following[0, 6],
                    normal[0, 0], normal[0, 1], relative[0, 0],
                )))
                if boundary == 0:
                    S = twist_state(S, following, following[:, 10])
                    ending = signals.offset_scenario(
                        following, following[:, 10] - cfg.control_dt)
                    _, following, changed = signals.advance_scenario(ending, S, cfg)
                    self.assertTrue(changed.item())
            return torch.cat(features_per_boundary)

        x = torch.tensor(0.11, dtype=torch.float64, requires_grad=True)
        yaw = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
        values = features(x, yaw)
        epsilon = 1e-5
        for index, value in enumerate(values):
            gx, gy = torch.autograd.grad(value, (x, yaw), retain_graph=True)
            fdx = (features(x.detach() + epsilon, yaw.detach())[index] -
                   features(x.detach() - epsilon, yaw.detach())[index]) / (2 * epsilon)
            fdy = (features(x.detach(), yaw.detach() + epsilon)[index] -
                   features(x.detach(), yaw.detach() - epsilon)[index]) / (2 * epsilon)
            self.assertTrue(bool(torch.isfinite(gx) and torch.isfinite(gy)))
            torch.testing.assert_close(gx, fdx, atol=3e-7, rtol=3e-6)
            torch.testing.assert_close(gy, fdy, atol=3e-7, rtol=3e-6)


if __name__ == "__main__":
    unittest.main()
