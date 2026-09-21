# Qwen3.5 hybrid 训练

入口为 `examples/drmas_trainer/run_math_qwen35.sh`。支持官方完整、非量化的
Qwen3.5 dense / MoE ConditionalGeneration checkpoint 的纯文本 Math 全参数训练。
默认 Solver 和 Verifier 是两个独立的 `Qwen/Qwen3.5-4B` Actor，独立语义编码器也为
`Qwen/Qwen3.5-4B`。语义编码器冻结，只训练其语义 MLP 头。Math LOTO / team-event GAE、
两阶段优势缩放、语义驱动独立熵约束保持原方法。
全量 rollout、W&B 指标、每 10 步 checkpoint 的配置继续生效。

## 环境

在新建的 Linux / NVIDIA CUDA 环境中安装，推荐 Python 3.11 或 3.12：

```bash
python -m pip install -r requirements_qwen35.txt
python -m pip install --no-build-isolation -r requirements_qwen35_kernels.txt
python scripts/check_qwen35_runtime.py --require-kernels --check-cuda
```

核心固定为 SGLang 0.5.10、Transformers 5.3.0、PyTorch 2.9.1；SGLang 版本的
[官方依赖](https://github.com/sgl-project/sglang/blob/v0.5.10/python/pyproject.toml)
是选择这一组合的依据。扩展需要与安装的 PyTorch/CUDA 匹配，源代码构建需要 CUDA 编译工具。
不要同时安装旧 `requirements_sglang.txt` 或 `[sglang]` extra；旧 Qwen3 环境仍可单独使用。
Qwen3.5 启动脚本先核对依赖版本，再启动训练；`DRY_RUN=True` 只输出参数。

## 启动与模型选择

从项目根目录运行（数据路径默认沿用原 hybrid）：

```bash
VALUE_CHECKPOINT=/models/semantic_qwen35/semantic_value.pt \
RUN_NAME=math_qwen35_4b \
bash examples/drmas_trainer/run_math_qwen35.sh train
```

更换稠密模型或 MoE，只需设置两个 Actor 的模型目录或 Hugging Face ID。小模型也必须采用
同一 Qwen3.5 适配路径，不能启用旧 Qwen3 attention monkey patch。

```bash
SOLVER_MODEL=Qwen/Qwen3.5-9B \
VERIFIER_MODEL=Qwen/Qwen3.5-9B \
VALUE_CHECKPOINT=/models/semantic_qwen35/semantic_value.pt \
bash examples/drmas_trainer/run_math_qwen35.sh train
```

`Qwen/Qwen3.5-35B-A3B` 等 MoE 也使用相同入口，但显存按全部专家权重、优化器、
两个 Actor 和 rollout 实例计算，不能按每 token 激活参数量计算。

默认物理 16 卡中 15 卡作为 Actor pool，1 卡给价值网络，rollout TP=1。
TP 必须整除 Actor 总卡数。例如双机每台 8 卡，TP=2 可采用：

```bash
NNODES=2 N_GPUS_PER_NODE=8 AGENT_GPUS_PER_NODE='[6,8]' \
ROLLOUT_TP=2 TRAIN_BATCH_SIZE=28 RAY_ADDRESS=auto \
SOLVER_MODEL=Qwen/Qwen3.5-35B-A3B VERIFIER_MODEL=Qwen/Qwen3.5-35B-A3B \
VALUE_CHECKPOINT=/models/semantic_qwen35/semantic_value.pt \
bash examples/drmas_trainer/run_math_qwen35.sh train
```

这是资源布局示例，不承诺任意 16 卡硬件均能容纳此 MoE。更大的 TP 同样要调整 Actor 布局；
单节点可见设备及 attention heads 也必须满足后端分片要求。每个 rollout TP 组必须落在同一台机器，
因此上述布局每个节点的 Actor 卡数均可被 TP 整除；训练侧 FSDP 仍可跨机器。多机 checkpoint 路径必须共享可见。

## 训练执行约束

- 使用原生 HF、普通 padding、SDPA、FSDP2、SP=1；视觉塔冻结，训练文本部分。
- FP32 master/FSDP 参数保持递归参数精度，Actor 前向在 BF16 autocast 中执行。
  这比全 BF16 参数通信占用更多显存和带宽。
- 修正 HF 5.3 单样本时忽略 padding mask 的线性注意力路径。
- 明确拒绝 remove-padding、旧 Ulysses SP>1、旧 fused/Liger 前向，避免跨样本状态污染。
  当前不提供 Qwen3.5 LoRA、量化权重训练或图像/视频任务适配。
- 沿用 `enable_thinking=False` 和 Solver `\boxed{}` / Verifier 判定协议。
- 不改变 Top16 熵的定义和独立 Actor 全词表熵约束。更换模型后重建熵校准状态。
- Qwen3.5 不套用旧 Transformer FLOPs 公式：记录 `mfu_supported=0`，不把伪造的 MFU=0
  当成设备效率；训练耗时和 token 吞吐仍可观测。

这些约束是在这份定制项目内验证的执行范围，不表示 Qwen3.5 架构本身不支持其他优化。

## 语义编码器与恢复

先用完整历史 Math 轨迹，为新的 Qwen3.5 编码器重新训练语义 MLP。输入包含完整原题、
真实 Solver 预算、顺序动作及最终成败；旧模型生成的轨迹也可作为监督数据。

```bash
VALUE_INPUT=/data/history/rollouts.jsonl \
VALUE_OUTPUT=/models/semantic_qwen35/semantic_value.pt \
bash examples/drmas_trainer/pretrain_semantic_value_qwen35.sh
```

新入口默认不加载旧头，从随机初始化的 MLP 开始；编码器使用官方预训练文本权重并保持冻结。
该入口与在线入口默认都使用 `Qwen/Qwen3.5-4B`、BF16、SDPA 及相同的头结构与前缀配置。
它复用基础预训练 YAML，并通过 `--model-path` 显式覆盖模型名。若指定本地模型路径或其他尺寸，
预训练和在线训练两次命令都要设置相同的 `VALUE_ENCODER`。旧 Qwen3 语义 checkpoint 不能作为
新编码器的 `VALUE_CHECKPOINT` 或 `VALUE_WARM_START`，即使两者隐藏维度相同也会被严格校验拒绝。

预训练输出 checkpoint 和 `.report.json`；通过资格返回 0，未通过仍保存候选头并返回 2。
在线默认 `VALUE_INIT_MODE=candidate`：用新预训练头初始化候选网络，在在线策略轨迹上重新验证，
取得语义资格后才驱动独立熵约束。显式的 `qualified` 模式只应用于已在目标策略数据上验证的头。

换编码器后使用新的 `RUN_DIR`，不要设置旧实验的 `RESUME_FROM`。同一编码器、同一实验的恢复
仍使用原 `RESUME_FROM` 机制，同时恢复语义网络与控制器状态。完整官方模型加载后会提取真实的
`model.language_model`；跨模型 checkpoint 校验继续严格执行。

## 验证边界

测试包括真实小型随机 Qwen3.5 dense/MoE 的因果性、左 padding、样本隔离、反向传播、
原生 checkpoint 保存加载，以及生产 Actor 前向上的 PPO + 独立熵梯度（CPU 和可用 CUDA）。
语义前缀及 SGLang 权重转换/返回协议另有回归测试；权重映射测试执行了固定版本的真实 SGLang loader，
覆盖 dense/MoE 完整 HF state dict 与专家矩阵切片。
本地 Windows 环境未启动 Linux SGLang 服务或多卡 FSDP，也未下载完整官方模型权重。
单卡 FSDP2 集成测试已加入，但本机 Gloo 不支持所需 CUDA 通信，执行时明确跳过。
因此这些检查不等同于正式规模训练、分布式同步或收敛验收。部署后需首先运行短训练，
检查训练/rollout log-prob 一致性、更新后权重生效、轨迹完整性及保存恢复。

参考：[HF Qwen3.5](https://huggingface.co/docs/transformers/v5.3.0/en/model_doc/qwen3_5)、
[SGLang 模型实现](https://github.com/sgl-project/sglang/blob/v0.5.10/python/sglang/srt/models/qwen3_5.py)。
