# 2026-10-19 配对复现

完整24场景×2重复×4配置，共192次CPU仿真。动作仍为CEM代理，反馈仍为模拟位姿；未加载GR00T，没有真机执行。每组48次，raw/smooth/project/feedback成功次数分别为37/37/35/43；本次末端稳定成功与历史最长稳定区间成功的计数一致。

manifest.json在运行前写入计划、主要终点、运行时版本及源码散列；episodes.jsonl记录每次结果、物理结束时间/步数及遥测散列；completed.json绑定三个主文件。telemetry.zip是运行结束后的无损打包，内含192个原始gzip文件；每个gzip的SHA256仍与episodes.jsonl逐条对应。它不是额外192次实验。

运行时新增runner尚未提交，所以manifest中的git_commit是当时Git基点，不是包含新增runner的发布提交。精确运行源码由source_sha256确定，当前仓库逐文件匹配该清单；回放命令也会核查这些散列。散列不能证明历史时间或作者真实性。

从仓库根目录运行`python -m benchmark.replay results/2026-10-19`即可直接读取压缩包，独立重算完整成功条件。原始冻结批次仍位于work/g1_objectcentric/results。相同场景和种子的复现不扩大统计独立样本量。
