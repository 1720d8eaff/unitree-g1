# G1 物体中心双臂操作仿真

这是可运行的 MuJoCo 控制实验，包含 G1 双臂 14DoF、接触搬箱、物体中心 SE(3) 目标投影、模拟位姿反馈与四组配对评估。它尚未加载 GR00T，也不涉及真机采集、SDK2 新控制器联调或 Dex3 手指控制。

## 启动与复现

在项目根目录双击 `start-demo.cmd`。只有一个原生窗口，限制 30FPS，物理碰撞和显示网格分离。先关闭以前的演示窗口，避免重复渲染。窗口内：

| 操作 | 功能 |
|---|---|
| F8 / Pause | 暂停或继续 |
| F5 / Replay | 重放相同场景与随机种子 |
| F9 / Scene | 下一个场景 |
| F6 / Mode | feedback → raw → smooth → project 循环 |
| F7 / Evaluate | 暂停显示任务，运行 192 次 CPU 仿真评估；结束后可按 F8 继续 |
| Esc / Close | 关闭窗口及该窗口启动的评估子进程 |

中文输入法可能占用字母键，因此保留按钮和 F 功能键。首次显示默认场景；之后循环 24 个正式场景，失败同样显示，不筛选为展示成功率。

PowerShell 在项目根目录运行：

```powershell
& .\.venv\Scripts\python.exe .\work\g1_objectcentric\test_geometry.py
& .\.venv\Scripts\python.exe .\work\g1_objectcentric\validate.py
& .\.venv\Scripts\python.exe .\work\g1_objectcentric\evaluate.py
& .\.venv\Scripts\python.exe .\work\g1_objectcentric\export_demo.py --episodes 2
```

也可从仓库根目录运行 `python evaluate.py` 和 `python export_dataset.py`。评估重新生成对应文件；不要同时运行多个评估进程或在评估途中修改控制代码。

## 控制链路

```text
已有 CEM 参考轨迹 → FK 生成候选末端动作块
                      ↓ 添加声明的双手差异误差与共同偏差
              raw / 平滑 / SE(3) 投影
                      ↓ 完整组加入带延迟的物体位置反馈
                 IK + 关节限位与命令速度限制
                      ↓
                  500Hz 关节 PD + 偏置力补偿
                      ↓
                    MuJoCo 接触物理
                      ↓
                 状态记录、任务评估、失败诊断
```

刚体约定：`G_i=T_object_hand_i`，`O_i=H_i G_i⁻¹`。共享物体位姿通过加权位置均值和 SO(3) 局部均值求解，输出 `H_i*=O*G_i`。抓持后仅在 LIFT/CARRY/LOWER 开启约束；接近与释放保持独立动作，释放继续位置补偿以减少跳变。

本地相对动作契约是 body-frame `T_next=T_obs Delta`，每个动作都相对同一个观测姿态。不要直接把它当成 GR00T processor 的相对动作定义。动作代理每 200ms 预测 40 个 20ms 控制点；执行约 10 步再更新。

完整组使用模拟传感器提供的延迟、噪声和掉帧位姿，与同一时间戳参考运动比较，作限幅、低通的共享位置修正。只实现位置反馈，没有完整旋转反馈或基于图像的跟踪器。位姿年龄超过 180ms 时不再使用旧误差；动作观测年龄超过 300ms 时保持上一关节目标。

## 模型和数据边界

- 使用原始 G1 手臂运动学和完整显示外观，身体与腿固定。没有平衡控制、行走、柔性物体或 Dex3 手指关节。
- 箱子为自由刚体，episode 期间不设置物体位姿，不焊接、不吸附。移动来自接触与摩擦。
- 尺寸变化通过重新编译轻量 XML 模型实现并缓存，避免仅改 `geom_size` 留下旧碰撞包围盒。显示模型只复制 qpos，不参与物理。
- 参考参数来自已有 CEM 参数搜索；夹持高度通过独立开发场景标定。当前运行没有神经网络训练，也没有视觉／语言条件动作预测。
- “自碰撞检查”仅限双掌代理距离与有限接触诊断，不是全身安全证明。过期处理也不保证物体不会掉落。
- 模拟位姿反馈的来源是 MuJoCo 真值加噪声，不是 RGB-D 视觉跟踪。导出 RGB-D 是真实渲染，但尚未用于识别或跟踪。

## 评估口径

`scenarios.py` 定义 3 初始偏航×4 目标位置×2 尺寸，共 24 个场景；每场景 2 次重复、4 组配置，合计 192 次。开发场景在 `development()` 中单列。各配置使用相同参考参数、场景、随机种子与成功标准。

默认扰动：物体位姿延迟 60ms，动作观测延迟 40ms；共同动作偏差标准差 4mm，结构化左右差异扰动标称 9mm；位姿观测含 0.5mm 高斯噪声与标定偏差；3.0s 起模拟 160ms 遮挡。旋转差异扰动另在代码中明确给出。这些是用于诊断的合成扰动，不是从 GR00T 预测残差拟合的真实误差分布。

成功要求：抬升至少 10cm、双掌接触累计至少 0.5s、未跌落；释放后桌面接触、XY 误差<3.5cm、Z 误差<2cm、Rzz>0.96、速度<3.5cm/s、双掌均未接触且连续稳定至少 1s。

三个几何指标不能混淆：

1. `commanded_inconsistency_m`：投影后笛卡尔目标隐含的左右物体位置差，发生在 IK 之前；不是实际机器人运动误差。
2. `ik_command_inconsistency_m`：限位／速度约束后的候选关节命令经 FK 重建，再计算相同残差；记录位置在双掌碰撞回退之前。
3. `grasp_drift_m`：稳定搬运 CARRY 阶段双掌均接触时，实际手相对物体位置相对首个合格时刻的平均漂移。先对每条轨迹平均，再对轨迹平均；无合格样本为 null，报告合格轨迹数量。

`results/episodes.csv` 保存所有成功和失败；`results/traces/*.jsonl.gz` 保存每条轨迹的控制、物体姿态和观测时间戳；`manifest.json` 保存参数与代码／模型散列。汇总报告含 Wilson 区间、配对成功计数及按场景聚类的 bootstrap 差值区间。有限样本与有限扰动不能证明真机泛化。

## 仿真数据导出与 GR00T 准备

导出使用 PyArrow 与 FFmpeg 生成 Parquet 和 H264 MP4，元数据遵循查证的 LeRobot v2.1 结构并包含 GR00T `modality.json`。默认 **2 条仿真轨迹，174 帧**，不是 174 条真机示范。

每帧 state/action 各 26 维：左右末端各 6 维 XYZ+旋转向量、双臂关节 14 维。磁盘存绝对末端和关节目标；`gr00t_config.py` 设置 RELATIVE，由官方 processor 完成变换。相机为两个固定仿真视角，额外保存 OpenGL 外参、视锥参数、米制深度 NPZ、物体真值和采样时间。真实手眼标定尚未完成。

`dataset/validation.json` 证明本地表维度、时间采样、H264 解码与视频帧数检查通过；**没有运行官方 GR00T loader**。RGB 视频没有自动提供物体位姿，视觉跟踪需后续开发。

自定义固定身体、固定掌心不等于预训练 `REAL_G1` 的模态，使用 `NEW_EMBODIMENT`。该配置是待在正式 GR00T 环境验证的接入文件，当前没有训练 checkpoint。

在满足官方依赖和权重访问要求的服务器上，可按以下顺序继续：

1. 锁定 NVIDIA/Isaac-GR00T 的 N1.7 源码版本并完成官方安装。
2. 运行其数据 loader 与 stats，检查当前自定义配置；修复 schema／旋转表示／视频时间轴问题。小样例只适合验证加载链路，不足以证明策略训练有效。
3. 采集更丰富的示范，按场景或 episode 划分数据，并生成相对动作统计。
4. 用 `launch_finetune.py` 微调，保留超参数、checkpoint、验证集与训练日志；先检查官方 CLI `--help`。
5. 通过官方 `PolicyClient` 远程推理，在动作回放、离线输出检查后接入本地控制。核对 processor 输出的绝对／相对表示，按数据帧率重采样到 50Hz，再做时效与物理闭环验证。

官方参考命令形状（服务器使用，尚未在本机执行）：

```bash
uv run python gr00t/data/stats.py --dataset-path /path/to/dataset \
  --embodiment-tag NEW_EMBODIMENT --modality-config-path /path/to/gr00t_config.py
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B --dataset-path /path/to/dataset \
  --embodiment-tag NEW_EMBODIMENT --modality-config-path /path/to/gr00t_config.py \
  --output-dir /path/to/checkpoints --num-gpus 1 --global-batch-size 8
```

## 文件说明

| 文件 | 用途 |
|---|---|
| geometry.py | SE(3) 投影、SO(3) 数值方法、相对动作编解码 |
| actions.py | 动作形状、坐标系与时间戳契约 |
| task.py | 接触任务、IK、阶段门控、反馈与指标 |
| scenarios.py | 开发与正式场景 |
| reference-parameters.json | 冻结的共同参考参数 |
| develop.py | 开发场景夹持高度探索；不用于正式评估 |
| evaluate.py | 四组 192 次配对仿真 |
| validate.py / test_geometry.py | 物理负对照、时效处理、尺寸接触与数值验证 |
| export_demo.py | 小规模双视角 RGB-D 与 Parquet/MP4 导出 |
| gr00t_config.py | 待在官方运行时验证的自定义本体配置 |
| demo.py / launch.py | 原生 30FPS G1 展示窗口 |

## 资料

- [GR00T 官方仓库](https://github.com/NVIDIA/Isaac-GR00T)
- [模态与相对动作配置](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_config.md)
- [数据准备要求](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)
- [Unitree SDK2 Arm7 示例](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/g1/high_level/g1_arm7_sdk_dds_example.cpp)
