# 物理仿真数据与 RGB-D 控制验证

这批数据在 2026-10-19 实际运行 MuJoCo 得到，动作源为 CEM 加模拟位姿反馈，来源为 `simulation`。它不是真机示范，没有运行 GR00T 权重，也没有替换原来的 2 条轨迹 / 174 帧或 192 次归档评测。

## 本次实跑结果

| 项目 | 实际数量 |
|---|---:|
| 预先冻结的候选 episode | 220 |
| 达到物理末端成功条件 | 189 |
| 按预定配额保留 | 174 |
| 物理条件失败 | 31 |
| 已合格但因配额排除 | 15 |
| 全部候选帧数 | 19,140 |
| 保留帧数 | 15,138 |
| 训练 / 验证 episode | 139 / 35 |
| 完整解码校验的双视角视频 | 348 |

**174 是保留配额，不是反向指定成功数。** 220 条全部执行并保留原始轨迹、100Hz 物理遥测、参数、种子、判定条件与排除原因。189/220 是采集筛选通过比例，不是独立留出集上的策略成功率；174/220 也不是成功率。

每条轨迹含 87 个 10Hz 观测 / 动作样本。RGB 来自模型的真实 MuJoCo 渲染，不再使用随机像素；视频为 160×120 H.264。状态和动作各 26 维：左 / 右末端世界坐标 XYZ + 旋转向量各 6 维，以及 14 维双臂关节。动作是该时刻控制器给出的绝对关节目标，不能直接解释为 body-frame 相对动作。机器人仍为固定身体、刚性掌心模型。

数据包见 [physical-collection-220](../examples/physical-collection-220/README.md)。内部包括 LeRobot v2.1 形状的 Parquet / MP4 / meta 导出，训练与验证为两个独立根目录。只用训练 episode 拟合统计，验证复用同一统计。官方 GR00T loader、模型训练及机器人泛化尚未验证；当前 episode 划分也不是未见场景留出评测。

```bash
python -m pip install -r requirements-dataflow.txt
python -m dataflow.collect --output-dir artifacts/collection --episodes 220 --keep 174
python -m dataflow.inventory artifacts/collection
python -m dataflow.validate artifacts/collection --output artifacts/collection/validation.json
```

输出目录必须不存在。校验会重算文件哈希、物理遥测成功条件、episode / frame 数、分区、归一化和完整视频帧数。采集器内部多记录一个初始 t=0 判定样本；原始遥测从 0.01s 的 REACH 开始，校验明确检查这一计数差异。

## 从 RGB-D 得到反馈

`dataflow.rgbd` 使用双视角颜色分割、标定反投影和顶面矩形拟合，从渲染 RGB + 米制深度计算箱体中心和模 π 偏航。它假设已知橙色、尺寸、直立箱体与模拟相机标定，不是一般物体的学习式六自由度跟踪。

```bash
python -m dataflow.vision_run --output-dir artifacts/rgbd-scene0 --scene 0 --seed 20261019
```

单场景实跑得到 86 次测量，中心位置平均评估误差 0.6116cm、最大 1.6618cm，末端放置条件通过；这只是一次控制验证，不是全面成功率。观测函数不读取当前物体位姿真值，真值仅用于评估误差。控制动作仍由 CEM 产生，控制反馈只使用 XYZ；初始场景和抓持先验仍来自仿真。原来的归档评测反馈来源不因此改变。

检查源码和结果：[dataflow](../dataflow)、[RGB-D 本次记录](../examples/rgbd-control-probe/summary.json)。
