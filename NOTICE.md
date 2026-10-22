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
  `src/trackerlab_supermimic/LICENSE.upstream` and in the Python package.
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

The downloader stores the dataset README, repository license and source robot
URDF beside the CSVs. Resampling is performed in memory for simulation.
