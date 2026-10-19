# 实际执行的 MuJoCo 仿真采集

来源为 **simulation**；非真机、非人工示范、未运行GR00T权重。

冻结220条候选，实际189条通过终末物理条件，按预定配额保留174条、15,138帧。31条物理失败、15条因配额排除，全部原始轨迹和100Hz遥测均保留。训练139条，验证35条，统计只用训练部分。189/220只是采集筛选结果，不是独立任务评测。

[完整数据包](dataset.zip)，SHA256：`770c5da6d8be5c6dbba0e98c446018e73c42329d961fa8aafcaed85019d2de24`。压缩包包含220条原始NPZ、逐条物理遥测与排除记录、174条Parquet和348份双视角真实MuJoCo渲染视频、相机模拟标定、分区元数据和文件散列。RGB不是随机像素。

```bash
python -m zipfile -e examples/physical-collection-220/dataset.zip artifacts/physical-collection
python -m dataflow.validate artifacts/physical-collection
```

独立校验通过：文件散列、物理成功条件重放、episode/帧数、训练/验证隔离、训练统计、时间轴及完整视频解码。导出遵循LeRobot v2.1文件形状，未运行官方GR00T loader或训练，不能称为已完成REAL_G1适配。

本次运行环境：Windows / Python 3.12、MuJoCo 3.3.6、NumPy 2.5.4、OpenCV 4.11.0、PyArrow 26.0.0、imageio-ffmpeg 0.6.0。来源与动作定义见[数据流说明](../../docs/physical-dataflow.md)。旧2条轨迹174帧及归档评测保持不变。
