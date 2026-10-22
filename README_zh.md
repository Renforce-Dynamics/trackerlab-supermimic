# TrackerLab SuperMimic

从 TrackerLab 提取的 G1 动作跟踪实现，覆盖原始数据下载、格式检查、
重采样、运动学计算、PhysX 仿真、世界模型训练、策略优化、评测、视频和策略导出。

这是 SuperMimic 的非官方复现，算法实现参考 `supermimic_vibed`，
使用 IsaacLab 2.3.2、Isaac Sim 5.1 和 PhysX。代码保留原作者声明与 MIT 许可证。
本仓库聚焦 40 个 G1 LAFAN1 动作，完整论文的多目标任务、BONES-SEED
实验及实机迁移留待后续验证。

## 使用流程

环境要求为 Linux、Python 3.11 和可运行 Isaac Sim 5.1 的 NVIDIA GPU。
具体安装步骤见 [安装文档](docs/installation.md)。

```bash
git clone https://github.com/ZouYan93/trackerlab-supermimic.git
cd trackerlab-supermimic
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

python scripts/prepare_resources.py --with_reference
python scripts/prepare_resources.py --verify_only --with_reference
python scripts/validate_rewards.py \
  --upstream_mdp data/supermimic_reference/mdp.py \
  --output logs/validation/rewards.json
python scripts/smoke_test.py --headless
python scripts/train.py --config configs/smoke.json --headless \
  --log_dir logs/smoke
```

root 容器运行仿真命令时添加 `--kit_args=--allow-root`。
Isaac Sim 许可由使用者阅读并同意，资源脚本保留源文件的许可声明。

## 训练与产物

```bash
python scripts/supervise.py --config configs/lafan1_8096.json \
  --run_dir logs/lafan1_8096
```

此配置从零初始化，运行 8,096 个环境、100,000 轮。
`configs/lafan1_4096.json` 提供 4,096 环境、320,000 轮配置。
`configs/smoke.json` 提供 4 环境、5 轮的功能测试。
命令行参数可以覆盖 JSON 中的值。

监督程序持续训练，并在指定轮次评测所有动作、记录八项奖励、
生成视频与策略导出、保存文件哈希。使用 `tmux` 或作业调度器运行长期任务。
`job_status.json` 表示运行状态，`milestones/*/ready.json` 表示该轮次产物完成。

`last.pt` 包含优化器，可用于续训；`ckpt_*.pt` 用于评测；
`policy.ts` 用于推理，内含归一化统计。完整命令和输入说明见
[英文说明](README.md)。

## 奖励来源与结果

八项奖励均来自论文附录 A、表 A.2：身体位置、身体姿态、
身体线速度、身体角速度、根部位置、根部姿态、动作变化惩罚、
关节限位惩罚。TrackerLab 早期版本已经包含这八项。
后续工作增加逐项记录与验证，并调整观测和重置配置。

原 TrackerLab 实验第 2,000 轮，8,096 个环境的完整动作平均回报为
**0.5416**，失败 **12/40**，训练耗时 **23.71 分钟**。
这些数据来自提取前的实验，精简版的发布测试见 `docs/validation.md`。
其他实现采用不同仿真后端，公共 checkpoint 尚未在本框架内复测。
分项结果与比较范围见 [实验记录](docs/results.md)。

## 文件布局

| 模块 | 位置 |
| --- | --- |
| 数据处理流程 | `scripts/prepare_data.py`、`src/trackerlab_supermimic/motion.py` |
| 任务与环境 | `task.py`、`env.py` |
| 奖励与观测 | `mdp.py` |
| 算法与网络 | `networks.py`、`buffer.py`、`scripts/train.py` |
| 评测与导出 | `scripts/evaluate.py`、`scripts/export_policy.py` |

详细流程见 [数据文档](docs/data_flow.md)，来源见 [NOTICE.md](NOTICE.md)。
