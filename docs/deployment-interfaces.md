# GR00T、SDK2 与可核查的模拟演示

本次接口新增于2026-10-19。归档的192次CEM控制仿真、2条轨迹/174帧数据及其散列保持不变。新增演示使用独立目录，拒绝覆盖已有文件。

后续新增的真实物理轨迹导出和受限 RGB-D 反馈实现见[物理数据流](physical-dataflow.md)。下文的 `collection_fixture` 仍是随机像素接口样例；不要与新物理集合混用。`deployment.simulation` 的 GR00T 动作入口与 `dataflow.vision_run` 的 CEM / RGB-D 控制入口分别验证，各自没有证明 GR00T 视觉闭环有效。

## 已实现的链路

| 模块 | 实现 | 本地验证范围 |
|---|---|---|
| `deployment.policy` | 本地`Gr00tPolicy`与远程`PolicyClient`；校验模态、历史长度、动作H×14、有限值 | 模拟策略接口、格式和异常测试；未加载真实GR00T权重 |
| `deployment.simulation` | 双视角MuJoCo渲染和状态输入；使用策略返回的绝对关节目标；可选SE(3)投影 | synthetic动作源完整任务时长冒烟运行；不报告GR00T性能 |
| `deployment.sdk2` | SDK2 `rt/lowstate`订阅和`rt/arm_sdk`发布；ID15–28双臂，ID29控制权重；CRC | 内存传输、消息字段、限速、超时与来源隔离测试；未连接真机 |
| `deployment.demo` | 合成观测→合成策略→限速控制→内存驱动 | 虚拟时钟；没有模型、任务成功率或实测延迟 |
| `deployment.collection_fixture` | 实际生成独立合成episode文件及清单，episode和frame分别计数 | 220条计划中按固定种子选择174条；接受决定为模拟规则，非质量评估 |

## 无设备的演示

```bash
python -m deployment.demo --output-dir artifacts/interface-demo
python -m deployment.collection_fixture --output-dir artifacts/synthetic-collection
python -m unittest discover -s tests -v
```

集合中每份NPZ和每行清单都保留`synthetic`来源，图像是随机像素。它不是LeRobot训练数据、物理仿真数据或174条真机示范。其总帧数由实际文件计算；原来的174帧仍然是174帧。

## 真实模型入口

先在独立GR00T N1.7环境安装官方依赖并取得匹配本仓库固定掌心模态的微调检查点。保持`NEW_EMBODIMENT`，不将固定掌心输入冒充`REAL_G1`。本接口只支持video/front、side；state/left_eef、right_eef、arms；language/annotation.human.task_description。观测历史为当前一帧，动作horizon从模型配置读取。

```bash
# GR00T GPU环境运行；checkpoint须来自匹配自定义模态的实际训练。
uv run python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT \
  --model-path /path/to/checkpoint --host 127.0.0.1 --port 5555 --strict

# 安装MuJoCo依赖与GR00T客户端的本仓库环境运行。
python -m deployment.simulation --policy remote --host 127.0.0.1 --port 5555 \
  --checkpoint /path/to/checkpoint --sample-dt 0.1 --duration 12 \
  --output-dir artifacts/gr00t-run

# 本地进程加载权重亦可。
python -m deployment.simulation --policy local --checkpoint /path/to/checkpoint \
  --sample-dt 0.1 --duration 12 --output-dir artifacts/gr00t-local
```

`sample-dt`必须等于该模型训练数据的动作周期。模型处理器应把`arms`还原为14维绝对关节弧度；本地body-frame相对末端动作不能直接接入。EEF输入是世界坐标XYZ/旋转向量，状态顺序与已有26维数据一致。底层用返回的arms执行，忽略模型额外的EEF动作，避免混合两个命令源。

此入口复用既有仿真场景、初始姿态、任务时长和PD，但在线目标不来自CEM参考轨迹。投影只在双掌接触时启用；它使用场景提供的抓持变换，未实现RGB-D视觉跟踪或学习的接触状态。图像来自轻量物理模型，训练/推理视觉域必须另行检查。推理期间物理时间暂停，因此不能用仿真时间宣称端到端延迟或实时频率。完整评测还需固定检查点/模型资产、种子、场景与消融协议。

## SDK2接线

`UnitreeTransport(interface, enable_hardware=True, robot_dof=29)`只创建网络传输；控制器需要操作者提供平台实测关节限位、速度和增益。无默认通用硬件参数，不控制腰部、腿、平衡、VR、相机或Dex3手指。

```python
from deployment.sdk2 import UnitreeTransport, ArmController, WatchdogLoop

transport = UnitreeTransport(interface, enable_hardware=True, robot_dof=29)
controller = ArmController(transport, lower=lower, upper=upper,
    max_speed=max_speed, kp=kp, kd=kd)
with WatchdogLoop(controller):
    # observation来自真实且已标定的同步RGB-D、FK和关节采集器。
    plan = model.predict(observation)
    controller.submit(plan.actions[0], observation.timestamp,
        source=plan.source, observation_source=observation.source)
```

以上为接线接口，变量由调用方提供。必须在同一个单调时钟域标记传感器和推理观测；驱动低状态使用本机接收时刻。相机/低状态同步和手眼标定由真实采集器负责，不能用合成观测代替。只有真实观测派生的模型指令可进入真传输；合成与仿真来源会被拒绝。看门狗独立于模型线程，过期或缺失状态/命令时通过权重0释放SDK控制权。释放权重不等于保证物体不掉落或整机安全；此行为需结合平台控制器验收。

SDK2默认示例针对G1 29DoF，23DoF的腕关节布局不同。该接口未做硬件验收、CRC真实消息运行或Linux网络验证。远程来源只能证明调用了声明的服务器，不能证明服务器实际加载了哪个权重；正式记录还需保存服务器版本、检查点散列和实测时间线。

参考：[GR00T官方Policy API](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md)、[SDK2官方G1 Arm7示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/high_level/g1_arm7_sdk_dds_example.py)。

**模拟数据仅用于接口和流程演示，未经过GR00T模型性能或真机验证。**
