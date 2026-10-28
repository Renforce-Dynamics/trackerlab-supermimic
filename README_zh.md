# TrackerLab SuperMimic

按照 TrackerLab 的目录与数据流组织 G1、Go2、A3、T1、Bumi 与 Pi+ 动作跟踪代码，同一份源码支持独立运行，
也支持安装到已有 TrackerLab。任务、动作库、参考缓存、动作管理、仿真环境、
算法和指标各有独立模块。

## 2026-10-28 参考轨迹与归一化修复

- 末帧之后的补帧保持位姿，根线速度、根角速度与关节速度置零。
- 真实末帧的速度按原记录保留。
- 已为 50Hz 的 CSV 保留全部源帧。
- 归一化张量采用原位更新，维持 FP32/FP64 内存地址与原计算公式。

共享实现按机器人关节数处理。具体行为与测试见
[修复说明](docs/reference_tail_fix.md)。下方历史视频及分数沿用各自原版本记录。

## 动作演示

以下画面来自 **IsaacLab / PhysX 策略仿真**，按 1 倍速播放。
动作跟踪画面左侧为策略、右侧为参考动作；Go2 过杆展示任务策略。
点击动图可查看对应 MP4。

| G1 舞蹈 · 32,000 轮 | G1 连续跳跃 · 32,000 轮 |
| --- | --- |
| [![G1 舞蹈与参考动作](docs/media/g1_dance_032000.gif)](docs/media/g1_dance_032000.mp4?raw=true) | [![G1 连续跳跃与参考动作](docs/media/g1_jump_032000.gif)](docs/media/g1_jump_032000.mp4?raw=true) |
| **Go2 小跑跟踪 · 10,000 轮** | **Go2 自主过杆 · 10,000 轮任务训练** |
| [![Go2 小跑跟踪](docs/media/go2_trot_010000.gif)](docs/media/go2_trot_010000.mp4?raw=true) | [![Go2 按杆高选择通过动作](docs/media/go2_adaptive_010000.gif)](docs/media/go2_adaptive_010000.mp4?raw=true) |
| **A3 交替出拳 · 10,000 轮** | **T1 舞蹈 · 10,000 轮** |
| [![A3 交替出拳与参考动作](docs/media/a3_punch_010000.gif)](docs/media/a3_punch_010000.mp4?raw=true) | [![T1 舞蹈与参考动作](docs/media/t1_dance_010000.gif)](docs/media/t1_dance_010000.mp4?raw=true) |
| **Bumi 厂商动作跟踪 · 10,000 轮** | **Pi+ 舞蹈 · 10,000 轮** |
| [![Bumi 厂商动作跟踪](docs/media/bumi_motion_010000.gif)](docs/media/bumi_motion_010000.mp4?raw=true) | [![Pi+ 舞蹈与参考动作](docs/media/pi_plus_dance_010000.gif)](docs/media/pi_plus_dance_010000.mp4?raw=true) |

[十二段演示、Pi+ 功夫、G1 起身、分数与数据来源](docs/demos.md)。
完整训练动作库的平均归一化回报：G1 **0.8695**、Go2 小跑 **0.9273**、
A3 **0.9845**、T1 **0.8703**、Bumi **0.9373**、Pi+ **0.8407**。
Pi+ 完成四段参考动作，功夫动作仍有明显跟踪偏差。Bumi 评测覆盖一段 6.82 秒厂商动作；
未测其他动作上的泛化。Go2 过杆在 16 次试验中完成 **157/173** 次尝试，
时限结束时的 16 次未完成尝试计入失败。各任务的数据量、目标与训练历史各有差异。

## 实现范围

这是 SuperMimic 的非官方复现，算法实现参考 `supermimic_vibed`，
使用 IsaacLab 2.3.2、Isaac Sim 5.1 和 PhysX。代码保留原作者声明与 MIT 许可证。
本仓库聚焦 40 个 G1 LAFAN1 动作，完整论文的多目标任务、BONES-SEED
实验及实机迁移留待后续验证。

Go2 扩展复用动作库、参考缓存、管理器、环境和 SuperMimic 算法。
新增 task 配置决定机器人、关节顺序、观测维度与奖励参数。
默认数据为 Kine2Go 公开的 Go2 小跑重定向轨迹，训练预设为
8,096 个环境、10,000 轮。程序生成的四条 IK 轨迹保留用于对照实验。
数据来源、运行命令和验证范围见
[Go2 文档](docs/go2.md)。导出的策略需要参考动作目标与观测历史。

多机器人任务按“任务类型 → 机器人 → 环境与算法配置”组织。
`tracking/a3` 使用智元公开的 20 段机器人动作，`tracking/t1` 使用
Booster 公开的 T1 舞蹈轨迹。两套预设均为 4,096 环境、10,000 轮，
沿用共享动作库、仿真环境、世界模型、策略训练与评测模块。
来源版本、单位转换、电机参数及验证范围见 [多机器人任务说明](docs/multirobot_tasks.md)。

新增 `tracking/bumi` 和 `tracking/pi_plus`，同样使用 4,096 环境、10,000 轮。
Bumi 使用厂商的一段 21 关节参考动作，转换时处理肘部越界并用当前模型重算身体目标；
Pi+ 使用厂商四段动作，保留全部 22 个关节，包含双腕。原始文件、转换记录与来源哈希分别保存。
两项任务的训练与评测命令见上述文档。

```bash
python scripts/rls/supermimic/tasks.py list
python scripts/rls/supermimic/tasks.py prepare --task tracking/a3
python scripts/rls/supermimic/tasks.py prepare --task tracking/t1
python scripts/rls/supermimic/tasks.py prepare --task tracking/bumi
python scripts/rls/supermimic/tasks.py prepare --task tracking/pi_plus
python scripts/rls/supermimic/tasks.py smoke --task tracking/a3
python scripts/rls/supermimic/tasks.py smoke --task tracking/t1
python scripts/rls/supermimic/tasks.py smoke --task tracking/bumi
python scripts/rls/supermimic/tasks.py smoke --task tracking/pi_plus
python scripts/rls/supermimic/tasks.py supervise --task tracking/a3 \
  --run_dir logs/a3_4096_10k --milestones 2000 10000
python scripts/rls/supermimic/tasks.py supervise --task tracking/t1 \
  --run_dir logs/t1_4096_10k --milestones 2000 10000
python scripts/rls/supermimic/tasks.py supervise --task tracking/bumi \
  --run_dir logs/bumi_4096_10k --milestones 2000 10000 --render_clips 2
python scripts/rls/supermimic/tasks.py supervise --task tracking/pi_plus \
  --run_dir logs/pi_plus_4096_10k --milestones 2000 10000 \
  --render_clips pi_plus_dance1_subject2 pi_plus_kungfu pi_plus_right_big_kick_cut
```

A3 采用 29 个运动关节，头部两关节固定；T1 为 23 关节模型。
这些输入均为上游提供的机器人空间轨迹，本地处理包含格式、单位和采样率转换。
仿真验证已覆盖两套模型的运动学、状态重置、回放数据和策略更新；
动作学习效果以训练后的完整轨迹评测为依据。

## 独立运行

MicroDuck 使用单独的 Python 3.12 环境与官方 mjlab/PPO 行走任务，
配置为 4,096 环境、10,000 轮，保留 BAM 电机模型。训练输入为仿真中的
速度与姿态指令，无动作捕捉数据；它与现有 IsaacLab/SuperMimic 模仿任务分开。
任务目录与运行命令见 [MicroDuck 说明](docs/microduck.md)。

环境要求为 Linux、Python 3.11 和可运行 Isaac Sim 5.1 的 NVIDIA GPU。
具体安装步骤见 [安装文档](docs/installation.md)。

```bash
git clone https://github.com/ZouYan93/trackerlab-supermimic.git
cd trackerlab-supermimic
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

python scripts/rls/supermimic/prepare_resources.py --with_reference
python scripts/rls/supermimic/prepare_resources.py --verify_only --with_reference
python scripts/rls/supermimic/validate_rewards.py \
  --upstream_mdp data/supermimic_reference/mdp.py \
  --output logs/validation/rewards.json
python scripts/rls/supermimic/smoke_test.py --headless
python scripts/rls/supermimic/train.py \
  --config scripts/rls/supermimic/configs/smoke.json --headless \
  --motion_config data/configs/supermimic_lafan1.json --log_dir logs/smoke
```

root 容器运行仿真命令时添加 `--kit_args=--allow-root`。
Isaac Sim 许可由使用者阅读并同意，资源脚本保留源文件的许可声明。

## 安装到已有 TrackerLab

在原 TrackerLab Python 环境中，从本仓库运行：

```bash
python scripts/install_trackerlab.py --trackerlab /path/to/trackerLab
python scripts/install_trackerlab.py --trackerlab /path/to/trackerLab --apply
cd /path/to/trackerLab
python -m pip install -r /path/to/trackerlab-supermimic/requirements.txt
python scripts/rls/supermimic/prepare_resources.py
python scripts/rls/supermimic/train.py \
  --config scripts/rls/supermimic/configs/smoke.json --headless \
  --motion_config data/configs/supermimic_lafan1.json --log_dir logs/supermimic_smoke
```

第一条命令显示文件计划，`--apply` 写入扩展。安装器保留原仓库的顶层
`__init__.py` 与 `setup.py`；涉及的父模块改为按需加载原导出内容，
并保存原文件及备份。重复安装会跳过相同文件，扩展文件冲突会报错。
详细行为与恢复方式见 [安装文档](docs/installation.md#existing-trackerlab)。

已有 TrackerLab 环境使用该安装器，独立模式使用根目录的 `pip install -e .`，
两种模式分别管理同名 Python 包。

## LAFAN1 数据

训练使用 `lvhaidong/LAFAN1_Retargeting_Dataset` 的 G1 子集。
上游作者已经通过 Interaction Mesh 与逆运动学，将人体动作转换为 G1
根部位姿及 29 个关节角。当前代码加载这些 CSV，在内存中完成
30 Hz 到 50 Hz 重采样、速度计算与正向运动学计算。

上游 retarget 主要考虑运动学约束；动力学、力矩限制与仿真跟踪效果仍需训练和评测。
下载文件放在 `data/retargeted/lafan1_g1/`，动作清单、原始帧率和关节顺序记录于
`data/configs/supermimic_lafan1.json`。来源和版本详见
[数据文档](docs/data_flow.md) 与 [NOTICE.md](NOTICE.md)。

## 训练与产物

```bash
python scripts/rls/supermimic/supervise.py \
  --config scripts/rls/supermimic/configs/lafan1_8096.json \
  --motion_config data/configs/supermimic_lafan1.json \
  --run_dir logs/lafan1_8096
```

此配置从零初始化，运行 8,096 个环境、100,000 轮。
同目录 `lafan1_4096.json` 提供 4,096 环境、320,000 轮配置，
`smoke.json` 提供 4 环境、5 轮的功能测试。
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
这些数据来自原训练实验，重构版的发布测试见 `docs/validation.md`。
其他实现采用不同仿真后端，公共 checkpoint 尚未在本框架内复测。
分项结果与比较范围见 [实验记录](docs/results.md)。

## 文件布局

| 模块 | 位置 |
| --- | --- |
| 任务与机器人 | `source/trackerTask/trackerTask/supermimic/` |
| 动作库与参考缓存 | `source/trackerLab/trackerLab/motion_buffer/` |
| 参考状态与重置采样 | `source/trackerLab/trackerLab/managers/supermimic.py` |
| 运动学 | `source/trackerLab/trackerLab/kinematics/supermimic/` |
| 仿真环境 | `source/trackerLab/trackerLab/tracker_env/supermimic/` |
| 奖励、观测与终止 | `source/trackerLab/trackerLab/tracker_env/mdp/supermimic.py` |
| 网络、经验缓存与训练器 | `source/trackerLab/trackerLab/algorithms/supermimic/` |
| 指标 | `source/trackerLab/trackerLab/metrics/supermimic.py` |
| 命令、配置与导出 | `scripts/rls/supermimic/` |

SuperMimic 使用专用的 MotionBuffer、MotionManager 与 `InteractiveScene`
环境，保留已有训练的计算方式和终止状态。它们采用 TrackerLab 的职责划分，
独立于原通用管理器的继承体系；训练入口由所附脚本提供。

详细流程见 [数据文档](docs/data_flow.md)，来源见 [NOTICE.md](NOTICE.md)。
