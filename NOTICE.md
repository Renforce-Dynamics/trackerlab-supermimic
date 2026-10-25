# Sources And Attribution

## Code

- TrackerLab: https://github.com/Renforce-Dynamics/trackerLab,
  base `1e5ccc062b445712a0aa7308cfb99edd7296cc88`, plus the SuperMimic work
  developed in its local workspace. The parent MIT notice is retained in
  `LICENSE`. This focused distribution is maintained under ZouYan93's account.
- SuperMimic adaptation: https://github.com/zita-ch/supermimic_vibed,
  commit `a1508d42c9d1699a52e8233ab16d08c488441ab9`.
  Copyright (c) 2026 ETH Zurich, Chong Zhang.
  The full MIT notice is retained at
  `source/trackerLab/trackerLab/algorithms/supermimic/LICENSE.upstream`
  and in the Python package.
  World-model/policy architecture, rotations, motion processing, differentiable
  MDP heads and training flow build on that implementation. This is an
  adaptation and extraction, not a claim of independent algorithm authorship.
- Paper: *SuperMimic: Scaling Humanoid Whole-Body Control with One
  Differentiable World Model*, https://arxiv.org/abs/2610.28480,
  version 1. Appendix A / Tables A.1-A.5, Algorithm A.1 and Appendix F.1 guide
  this motion-tracking implementation.

The TrackerLab work adds the PhysX environment adaptation, resource and
validation workflow, episode-boundary handling, reward-component metrics,
policy packaging and reproducibility records. Public-release packaging
removes machine-specific paths and unrelated TrackerLab tasks.
The version 0.2 refactor adopts TrackerLab's module ownership and source
layout, adds a reference buffer and motion manager, and provides a backed-up
overlay installer for existing TrackerLab checkouts. Algorithm authorship
and inherited licenses remain as described above.

The Go2 extension adds immutable task-selected robot descriptions, variable
joint dimensions, procedural foot trajectories and bounded numerical inverse
kinematics using MuJoCo Jacobians and SciPy. It uses the existing SuperMimic
learner. Go2 actuator parameters follow IsaacLab v2.3.2's Unitree Go2 DC motor
configuration. Generated gait references have no LAFAN1 dependency.
The default Go2 resource importer reads pinned Kine2Go retargeted references,
validates raw DOF ordering with full-frame FK and converts numeric NPY files
to TrackerLab CSVs. The optional procedural IK generator remains available.

## External Resources

- Robot asset repository: https://github.com/Renforce-Dynamics/assetslib,
  revision `9df792784e3642163f03390bf6ae16cf88b222a5`.
  Robot files and upstream notices remain in the external checkout.
- Retargeted motion source:
  https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset,
  revision `ce1572906efe6157840e8474d5a0d7aa87481e74`.
  Its README declares LAFAN1 motion data under CC BY-NC-ND 4.0 (Ubisoft
  La Forge). Its repository also has a separate BSD notice. The code license
  does not grant additional motion-data rights. Original CSVs are retained
  unchanged, and neither source nor transformed motions are included in Git.
- IsaacLab and Isaac Sim are installed separately and retain their own
  licenses and terms. Resource scripts do not accept those terms on behalf
  of the person installing them.
- Go2 reference data: `MIMUW-Robotics/kine2go` on Hugging Face, revision
  `7f03a3fb7f6b2e684899530862ba103e81570363`. The selected `ai4_dog_trot_00`
  originates in AI4Animation natural dog mocap, attributed to H. Zhang et al.,
  "Mode-Adaptive Neural Networks for Quadruped Motion Control", SIGGRAPH 2018.
  Its source metadata declares CC BY-NC 4.0. The importer preserves the source
  README, repository license, per-source license and clip metadata.
  The public retargeting pipeline is `nomagiclab/kine2go-pipeline`, revision
  `65b39f104706c1b4307dfc2a7df2aa8bed8d20aa`. Source/converted motions stay
  outside Git; this repository's MIT code license does not relicense data.

The downloader stores the dataset README, repository license and source robot
URDF beside the CSVs. Resampling is performed in memory for simulation.

## Additional Humanoid Resources

- Bumi: https://github.com/Noetix-Robotics/noetix_bumi_lab,
  revision `2454535c967cd1d6f04dfe0f0289f3c6c608e596`.
  Its `bumi.py` identifies Beijing Noetix Robotics TECHNOLOGY CO.,LTD.,
  copyright 2025-2026, under a BSD 3-Clause header. The external checkout
  supplies the Bumi V3 MJCF/meshes, actuation parameters and `2.npz` reference.
  The converter records elbow limit projection and recomputes body targets;
  it does not reuse the older cached body geometry.
- Pi+: https://github.com/HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic,
  revision `df3fcf16bf2cf645b8e3c24ee1e395d453f78f80`.
  The external checkout supplies Pi Plus assets, actuation settings and four
  robot-space CSV motions. Its README displays an MIT badge, while no top-level
  license file is present at the pinned revision. All 22 moving joints,
  including wrists, are retained. Review upstream terms before redistribution.

These assets and motions are downloaded separately and remain subject to
their upstream terms. This repository's MIT code license does not relicense
external datasets or robot assets. The shared learner remains SuperMimic.
