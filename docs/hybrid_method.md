# Hybrid：zxj 优势 + mix 两阶段信用 + 语义熵约束

本项目的 Math 训练使用 zxj 原有事件链、Math answer LOTO 价值、GAE 和题目／角色标准化，得到每个真实 action 的基础优势。随后执行 mix 的熵排名与稀疏时序信用转移，最终系数只乘一次：

\[
A_e^{base}=\operatorname{Normalize}_{task,role}(\operatorname{TeamGAE}(V_{LOTO})),
\qquad A_e^{final}=c_e A_e^{base},\quad c_e\in[0.8,1.2].
\]

奖励、returns、LOTO value 和原始优势诊断保持不变；缩放后不再次标准化。第二阶段默认只作用于 Solver，并检查相邻基础优势的实际符号。复制样本复用相同系数，Actor 继续使用 zxj 的 UID 调度和逆副本权重。

零优势恢复、虚拟奖励和恢复权重预算均已移除。全成功／全失败组在 zxj 默认条件下仍为零优势，两阶段乘法不能恢复它们。

辅助价值网络只保留冻结编码器与可训练语义头，预测完整交互前缀的最终成功概率；绝对熵和时序熵预测分支均已移除。它不参与 GAE 或优势缩放，只驱动独立的 Actor 熵约束：

\[
L=L_{PPO}(A^{final})+0.0025\,\operatorname{mean}_{valid\ unique\ actions}
\left[\operatorname{stopgrad}(g_e)\,(\bar H_\theta(e)-H_e^{cap})_+\right].
\]

`g_e` 由语义价值下降、熵风险、语义模型自身资格和渐进启用共同决定。控制只作用于非终局、有效且未截断的动作；终局前缀仍可用于价值训练与预测。训练使用当前已部署语义头生成冻结控制信号，两个 Actor 更新后才训练候选语义头，通过验证后从下一轮部署。零优势动作仍可能获得独立熵正则梯度。

Qwen3.5 系列的环境安装、模型切换和训练入口见 [Qwen3.5 训练说明](qwen35_training.md)。
该入口的 Actor 和独立语义编码器默认均为 Qwen3.5-4B；请用
`pretrain_semantic_value_qwen35.sh` 重新训练匹配的语义头。下方保留 Qwen3 通用入口的用法。

## 预训练与初始化

以下命令在已安装项目依赖的 Linux 训练环境、项目根目录运行。历史输入必须包含完整原题、真实 Solver 预算、按实际执行顺序记录的动作和终局成败；不要用截断的 Actor prompt 替代原题。

```bash
VALUE_INPUT=/data/history/rollouts.jsonl \
VALUE_OUTPUT=/models/semantic_value.pt \
VALUE_ENCODER=Qwen/Qwen3-4B \
bash examples/drmas_trainer/pretrain_semantic_value.sh
```

脚本默认加载 [pretrain_semantic_value.yaml](../examples/drmas_trainer/pretrain_semantic_value.yaml)，与在线启动脚本的语义配置一致，编码器默认 `bfloat16`。可通过 `VALUE_CONFIG` 指定自定义配置；在线的编码器、revision、dtype、头结构及相关配置须与 checkpoint 一致。`VALUE_DEVICE=cpu` 可切换预训练设备，`VALUE_WARM_START=/models/old.pt` 可显式初始化语义候选权重。

预训练输出 checkpoint 和 `.report.json`。通过资格时退出码为 0；未通过资格时仍保存候选 checkpoint，并返回 2。在线默认 `VALUE_INIT_MODE=qualified`，要求使用已通过验证的语义 checkpoint。

若需要从未通过资格的语义权重或旧 mix v2 checkpoint 开始，必须显式设置 `VALUE_INIT_MODE=candidate` 和 `VALUE_CHECKPOINT`。该模式只提取语义候选头，重新验证；不会沿用旧熵分支或旧部署资格。它仍检查编码器及结构兼容性，不等价于恢复旧训练。

## 新训练

单机 16 卡默认使用 15 卡 Actor 共享资源池，另留 1 卡给独立语义编码器；Solver／Verifier 保持各自模型。默认题目 batch 为 30，每题 8 条轨迹，最多 3 次 Solver 调用。

```bash
VALUE_CHECKPOINT=/models/semantic_value.pt \
RUN_DIR=/runs/hybrid_new \
bash examples/drmas_trainer/run_math_hybrid.sh train
```

双机各 8 卡时，先让两个节点加入同一个 Ray 集群，再在头节点启动。默认 Actor 布局为 `[7,8]`；调度器选择一个合适节点承载训练协调器，并在该节点留 1 卡给语义编码器：

```bash
RAY_ADDRESS=auto NNODES=2 N_GPUS_PER_NODE=8 \
VALUE_CHECKPOINT=/models/semantic_value.pt \
RUN_DIR=/shared/runs/hybrid_new \
bash examples/drmas_trainer/run_math_hybrid.sh train
```

多节点路径应在各节点可访问。可设置 `TRAIN_DATA`、`VAL_DATA`、`SOLVER_MODEL`、`VERIFIER_MODEL`、`VALUE_ENCODER` 等；更换 batch 时必须是实际 Actor world size 的整数倍。加 `DRY_RUN=True` 可查看最终命令。新训练要求空 `RUN_DIR`；旧 mix 候选初始化示例是在以上命令同时设置 `VALUE_INIT_MODE=candidate VALUE_CHECKPOINT=/models/old_mix_value.pt`。

## 显式恢复与评估

训练不会自动选取旧目录继续。恢复时明确指定完整 `global_step_N` 目录，并保持方法配置一致：

```bash
RESUME_FROM=/runs/hybrid_new/global_step_10 \
RUN_DIR=/runs/hybrid_new \
bash examples/drmas_trainer/run_math_hybrid.sh train
```

完整 hybrid 恢复必须包含 `semantic_value.pt`、`semantic_entropy_controller.json`、`hybrid_state.json` 三个状态文件，以及正常的角色 Actor checkpoint 和 `data.pt`。缺失状态会报错，不会静默降级为新训练；恢复时不使用 `VALUE_CHECKPOINT` 代替这些状态。

```bash
RESUME_FROM=/runs/hybrid_new/global_step_10 \
bash examples/drmas_trainer/run_math_hybrid.sh eval
```

评估默认读取 `data/drmas_math/test.parquet` 全量测试集，并关闭语义训练与熵控制；训练期间默认验证集为 `test_sampled.parquet`。

## 全部轨迹与 checkpoint

默认训练入口每个训练 step、每个验证 batch 保存全部真实轨迹，不按成败、角色、长度或 W&B 展示数量抽样。每行 JSONL 为一条完整轨迹，`actions` 按全局 `event_index` 排序；只删除 `event_uid` 相同的训练补齐副本，文本相同但事件不同的动作仍保留。

| 路径（相对 `RUN_DIR`） | 内容 |
| --- | --- |
| `rollouts/train_step_00000010_part_00000.jsonl` | 全部训练轨迹：原题、真实预算、角色与事件身份、完整原始/执行文本、Actor 观测、实际 token IDs、生成 log probabilities、真实奖励和终局结果；附带基础/最终优势、returns、两阶段系数、语义预测及控制信号 |
| `validation/val_step_00000010_part_00000.jsonl` | 本次验证实际生成的全部轨迹；同一步的后续 batch 自动增加 `part`，不会覆盖前一批。验证未计算 PPO 优势和语义预测，这些数值不写入 |
| `event_traces/` | zxj 原始事件与采样 manifest，在训练 batch 调整前保存，包含精确生成 token IDs |
| `credit_traces/` | zxj LOTO/GAE 信用诊断，保存原始值估计、TD 残差和标准化信息 |
| `global_step_10/`、`global_step_20/` 等 | 完整 checkpoint：两个角色 Actor、语义网络、熵控制器、合版配置与 dataloader 状态 |

轨迹导出发生在当前批 Actor 更新之前，其 `actor_update_count` 标识生成该批数据的策略版本。文件保留完整文本，不额外截断；若 Actor 输入或生成本身已触发长度上限，保留真实输入、生成结果及截断标记，不补造模型未生成的内容。训练文件的 canonical `actions` 格式可以直接用于语义预训练。

checkpoint 默认每 **10 个全局训练 step** 保存一次，训练最后一步也保存。`SAVE_FREQ` 或 Hydra override 可显式覆盖该值。轨迹保存不受 checkpoint 频率限制。

## W&B 观测指标

`run_math_hybrid.sh` 默认 `trainer.logger=[console,wandb]`，所有训练 scalar 每 step 发送到 W&B；验证指标在每次验证后发送。项目名为 `DrMAS_math_hybrid`，run 名由 `RUN_NAME` 决定。开始训练前配置 W&B 登录；多节点可通过 `WANDB_API_KEY` / `WANDB_ENTITY` 环境变量显式传递给远程日志协调器。`LOGGERS=[console]` 或 `WANDB_MODE=offline` 仅在用户主动设置时关闭线上上传。

| 指标前缀 | 主要观察内容 |
| --- | --- |
| `episode/`、`val/` | 训练 `pass@8` / `avg@8`、验证 `pass@k` / `avg@k`、奖励、任务长度和数据源分项；验证 k 取实际 `VAL_GROUP_SIZE` |
| `actor/<worker>/` | PPO loss、clip fraction、KL、梯度范数、学习率、真实策略熵，以及独立熵约束损失和有效动作数 |
| `credit/<worker>/` | zxj 事件原始优势与有效唯一事件数 |
| `entropy_credit/` | Top16 归一化熵、覆盖率、第一/第二阶段和最终缩放系数、边界命中率 |
| `pure_entropy/` | 第二阶段候选/激活配对、实际优势符号不匹配、样本不足或阈值导致的跳过 |
| `hybrid/<role>/advantage/` | 按唯一事件统计基础/最终优势的均值、标准差、绝对均值、零比例、符号翻转和实际缩放幅度 |
| `hybrid/<role>/zero_groups/` | `(task, role)` 零优势组数及缩放后保持零的比例；`final_nonzero_count` 应为 0 |
| `semantic_value/score/`、`semantic_value/train/` | 当前部署版本与角色资格、预测覆盖、BCE、候选/部署头的非终局角色 AUC/Brier/ECE、部署/停用状态及训练/留出样本数量 |
| `entropy_control/`、`hybrid/<role>/semantic/` | 每角色/轮次的 cap、risk、ramp、语义价值变化；符合条件→预测可用→实际激活的数量和比例，以及零优势动作的独立熵激活比例。熵约束使用全词表熵，单位为 nats |
| `trajectory_dump/`、`timing_s/` 等 | 已写入的轨迹/事件数量、训练阶段耗时和吞吐 |

新增 hybrid 统计按 `event_uid` 去重，避免 batch padding 改变曲线。空范围只报告计数，不伪造条件均值或比例。`semantic_value/score` 是本批使用的部署头状态，`semantic_value/train` 是本批所有 Actor 更新后的候选训练/部署结果。

## 垂域范围

**当前完整合版仅支持 Math。** 优势入口明确要求 `math_answer_loto`，语义前缀和资格检查绑定 Solver/Verifier 角色及 Math 停止规则。

SearchQA 环境与旧入口仍保留，但 `run_search.sh` 运行的是原有 GRPO 路径，不是完整 hybrid。底层 zxj Search event GAE 代码存在；要把完整方法接到 SearchQA，还需选择并接入 Search 值估计，适配 Verifier/Search/Answer 三角色的语义前缀、终局规则与控制资格，不能只切换 `env_name`。
