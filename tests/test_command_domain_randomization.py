"""CPU checks for privileged command-domain randomization."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.command_domain_randomization import (
    apply_command_domain_scales,
    command_domain_randomization_contract,
    command_domain_scales,
)
from trackerLab.tracker_env.supermimic.terrain import TerrainSim


class CommandDomainRandomizationTests(unittest.TestCase):
    def test_context_maps_to_bounded_privileged_physical_scales(self):
        xi = torch.zeros(4, 33)
        xi[0, 1:5] = -.01
        xi[1, 1:5] = .01
        xi[2, 1:5] = torch.tensor([-.005, .0025, .0075, -.01])

        scales = command_domain_scales(xi, 29)

        torch.testing.assert_close(scales[0], torch.tensor([.9, .85, .85, .9]))
        torch.testing.assert_close(scales[1], torch.tensor([1.1, 1.15, 1.15, 1.1]))
        torch.testing.assert_close(scales[2], torch.tensor([.95, 1.0375, 1.1125, .9]))
        torch.testing.assert_close(scales[3], torch.ones(4))

    def test_rejects_bad_context(self):
        for xi in (
            torch.zeros(2, 32),
            torch.zeros(2, 33, dtype=torch.long),
            torch.full((2, 33), float("nan")),
            torch.zeros(33),
        ):
            with self.subTest(shape=tuple(xi.shape)), self.assertRaises(ValueError):
                command_domain_scales(xi, 29)

    def test_selected_worlds_scale_from_baselines_without_compounding(self):
        body_mass = torch.tensor([[2., 3.], [5., 7.], [11., 13.]])
        body_inertia = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) + 1
        gain = torch.ones(3, 4, 10)
        bias = torch.zeros_like(gain)
        bias[..., 1], bias[..., 2] = -2., -3.
        force = torch.ones(3, 4, 2)
        base = tuple(value.clone() for value in (body_mass, body_inertia, gain, bias, force))
        ids, actuators = torch.tensor([0, 2]), torch.tensor([1, 3])
        first = torch.tensor([[1.1, .85, 1.15, .9], [.9, 1.15, .85, 1.1]])

        def apply(scales):
            apply_command_domain_scales(
                ids, actuators, scales,
                body_mass=body_mass, body_inertia=body_inertia,
                actuator_gainprm=gain, actuator_biasprm=bias,
                actuator_forcerange=force,
                base_body_mass=base[0], base_body_inertia=base[1],
                base_actuator_gainprm=base[2], base_actuator_biasprm=base[3],
                base_actuator_forcerange=base[4],
            )

        apply(first)
        torch.testing.assert_close(body_mass[0], base[0][0] * 1.1)
        torch.testing.assert_close(body_mass[2], base[0][2] * .9)
        torch.testing.assert_close(body_inertia[0], base[1][0] * 1.1)
        torch.testing.assert_close(body_inertia[2], base[1][2] * .9)
        torch.testing.assert_close(gain[1], base[2][1])
        torch.testing.assert_close(bias[1], base[3][1])
        torch.testing.assert_close(force[1], base[4][1])
        torch.testing.assert_close(gain[0, 1, 0], base[2][0, 1, 0] * .85)
        torch.testing.assert_close(gain[0, 3, 0], base[2][0, 3, 0] * .85)
        torch.testing.assert_close(bias[0, 1, 1], base[3][0, 1, 1] * .85)
        torch.testing.assert_close(bias[0, 1, 2], base[3][0, 1, 2] * 1.15)
        torch.testing.assert_close(force[0, 1], base[4][0, 1] * .9)

        second = torch.ones_like(first)
        apply(second)
        torch.testing.assert_close(body_mass[0], base[0][0])
        torch.testing.assert_close(body_inertia[2], base[1][2])
        torch.testing.assert_close(gain[0], base[2][0])
        torch.testing.assert_close(bias[2], base[3][2])
        torch.testing.assert_close(force[2], base[4][2])

    def test_contract_discloses_fixed_world_sampling_and_scale_ranges(self):
        nominal = command_domain_randomization_contract(False)
        randomized = command_domain_randomization_contract(True)
        self.assertFalse(nominal["enabled"])
        self.assertEqual(nominal["scale_ranges"], {})
        self.assertTrue(randomized["enabled"])
        self.assertIn("fixed joint-offset", randomized["source"])
        self.assertFalse(randomized["actor_observes_physical_scales_directly"])
        self.assertEqual(randomized["scale_ranges"]["body_mass"], [.9, 1.1])

    def test_episode_resets_preserve_physical_domain_channels(self):
        xi = torch.zeros(3, 33)
        xi[:, 1:5] = torch.tensor([-.009, .004, -.003, .008])
        xi[:, -3:] = torch.tensor([.01, -.02, .015])
        sim = SimpleNamespace(
            xi=xi, physical_domain_randomization=True, device="cpu", scale=1.,
            spec=SimpleNamespace(num_joints=29),
            geom_friction=torch.ones(3, 2, 3),
            base_geom_friction=torch.ones(3, 2, 3),
            body_ipos=torch.zeros(3, 2, 3),
            base_body_ipos=torch.zeros(3, 2, 3), com_id=1,
        )
        fixed = xi.clone()
        TerrainSim._randomize(sim, torch.tensor([0, 2]))
        torch.testing.assert_close(sim.xi[[0, 2], 1:5], fixed[[0, 2], 1:5])
        torch.testing.assert_close(sim.xi[[0, 2], -3:], fixed[[0, 2], -3:])
        torch.testing.assert_close(sim.xi[1], fixed[1])
        self.assertGreater(sim.xi[[0, 2], 5:-3].abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main()
