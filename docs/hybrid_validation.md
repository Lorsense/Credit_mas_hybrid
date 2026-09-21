# Hybrid 合版验证记录

验证日期：2026-09-20。项目代码改动仅在 `Credit_mas_hybrid`；`mas_zxj/DrMAS` 与 `Credit_mas_mix` 保持原样。临时备份与隔离测试环境位于工作区 `.hybrid_merge_work`。

## 数值来源与合版边界

- `verl/trainer/ppo/core_algos.py` 与 zxj 逐字节一致，Math LOTO 和 team event GAE 的算法实现未改写。
- `verl/utils/entropy_credit.py`、`sparse_entropy_credit.py` 与 mix 逐字节一致。
- `hybrid_credit.py` / `hybrid_training.py` 连接两套方法：原始事件上固定熵排名；zxj 完成一次优势标准化；实际优势符号确定第二阶段门控；最终系数只乘 Actor 优势。
- `returns`、真实奖励、LOTO 值和 zxj 诊断张量保持不变。`hybrid_base_advantages` 保存缩放前优势。
- 纯语义预测仅驱动独立熵损失；当前批部署头固定，所有角色更新后才训练候选头。没有熵预测分支、零优势恢复或虚拟奖励。

## 已执行验证

工作区隔离环境：Python 3.13.9、PyTorch 2.9.1+cu126。数值和梯度测试使用真实 CPU tensors；分布式归一化测试验证分片求和公式，没有启动多卡 FSDP。

1. zxj 原始 Math LOTO / GAE / 事件链与 boundary value：**131 passed**。
2. mix 熵信用工具、hybrid 信用集成、Actor 与 rollout、纯语义网络与控制器：**121 passed**。
3. 启动参数与异构资源分配：**8 passed**。使用真实 launcher 参数数组进行 Hydra compose；用延迟调度模拟器执行生产资源池与 WorkerGroup 方法，验证 `[7,8]` 分配顺序和正确的 15 个 Actor rank。
4. Driver 生命周期与 checkpoint 集成：**7 passed**。使用真实语义模块与同步 Ray 代理验证冻结预测、更新入口、终局排除、状态恢复、配置不匹配拒绝和旧方案初始化边界。

以上合计 **267 项通过，无跳过**。新增生产模块以及已修改的 Actor、Driver、资源调度代码还通过了未定义名称等静态检查。启动脚本的参数和配置已校验，未执行真实 Bash 启动流程。

关键覆盖：历史精确数值与去重不变性；关闭调制等于 zxj；优势以外的张量不变；全成功或全失败组的零优势不恢复；第二阶段使用真实优势符号；padding 副本的系数及 sample weight；语义 BCE、冻结 encoder、部署时序、资格与 checkpoint；独立熵梯度和微批/DP 分母。

运行上述测试，在项目根目录执行：

```bash
python -m pytest -q tests/test_team_event_math_value.py tests/test_team_event_math_gae.py tests/test_math_event_trace.py tests/test_team_event_boundary_value.py
python -m pytest -q tests/utils/test_entropy_credit.py tests/utils/test_sparse_entropy_credit.py tests/utils/test_hybrid_actor.py tests/utils/test_hybrid_rollout_metadata.py tests/trainer/ppo/test_hybrid_credit_integration.py tests/workers/test_semantic_value.py tests/utils/test_semantic_credit.py tests/utils/test_semantic_entropy_control.py
python -m pytest -q tests/utils/test_hybrid_launch.py tests/utils/test_hybrid_resources.py
python -m pytest -q tests/trainer/ppo/test_hybrid_training.py
```

## 验证范围

未下载或加载真实 Qwen3 权重，未启动 SGLang / Ray 多节点 / FSDP 完整训练，未评估合版后的训练收敛或最终准确率。当前本地环境缺少完整推理服务依赖；实际训练需要原项目的 Linux/CUDA/SGLang 环境、模型、数据和语义初始 checkpoint。CPU 测试通过不代表已经完成多卡训练验收。

## 全量轨迹与 W&B 观测补充

随后增加完整轨迹导出、按唯一事件的 hybrid 指标、默认 W&B 日志，并统一 checkpoint 默认间隔为 10。相关 **69 项测试通过**，包括：完整原文和 token IDs、失败轨迹保留、只去 padding 副本、同一步多验证批次不覆盖、真实预训练回读、实际 Driver 每 step 保存入口、指标去重及不修改训练张量、真实 Tracking 适配器的模拟 W&B 转发、配置和原合版生命周期回归。

测试从项目根目录运行：

```bash
python -m pytest -q tests/utils/test_hybrid_trajectory.py tests/utils/test_hybrid_observability.py tests/utils/test_hybrid_launch.py tests/utils/test_hybrid_wandb.py tests/trainer/ppo/test_hybrid_credit_integration.py tests/trainer/ppo/test_hybrid_training.py tests/workers/test_semantic_value.py tests/utils/test_hybrid_rollout_metadata.py
```

W&B 测试未连接线上服务；真实训练时由 `Tracking` 每步调用 W&B。测试计数包含对既有相关测试的重跑，不应与前面的 267 直接相加。
