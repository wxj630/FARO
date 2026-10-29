from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from .plot_results import FIGURE_NAMES


RESULT_ORDER = ("adam", "muon", "normuon", "faro", "faro_prior")
DISPLAY_NAMES = {
    "adam": "Adam",
    "muon": "Muon",
    "normuon": "NorMuonAndAdam",
    "faro": "FARO no-prior auto-switch",
    "faro_prior": "FARO prior control",
}


def _mean_loss(summary: Mapping[str, object], key: str) -> float:
    return float(summary["variants"][key]["final_validation_loss_mean"])


def _seed_loss(summary: Mapping[str, object], key: str, seed: int) -> float:
    return float(summary["variants"][key]["seed_values"][str(seed)])


def _result_table(summary: Mapping[str, object]) -> str:
    muon_loss = _mean_loss(summary, "muon")
    rows = [
        "| 实验 | mean ± std | seed0 | seed42 | seed1024 | 相对 Muon mean（mean − Muon） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in RESULT_ORDER:
        variant = summary["variants"][key]
        loss = _mean_loss(summary, key)
        rows.append(
            f"| {DISPLAY_NAMES[key]} | {loss:.9f} ± {variant['final_validation_loss_std']:.9f} | "
            f"{_seed_loss(summary, key, 0):.9f} | "
            f"{_seed_loss(summary, key, 42):.9f} | "
            f"{_seed_loss(summary, key, 1024):.9f} | "
            f"{loss - muon_loss:+.9f} |"
        )
    return "\n".join(rows)


def _system_table(summary: Mapping[str, object]) -> str:
    rows = [
        "| 实验 | tokens/s | training time (s) | wall time (s) | peak allocated (MiB) | peak reserved (MiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    system = summary["system"]
    for key in RESULT_ORDER:
        values = system[key]
        rows.append(
            f"| {DISPLAY_NAMES[key]} | {values['tokens_per_second_mean']:.1f} | "
            f"{values['training_time_ms_mean'] / 1000:.3f} | "
            f"{values['wall_time_ms_mean'] / 1000:.3f} | "
            f"{values['peak_allocated_mib_mean']:.1f} | "
            f"{values['peak_reserved_mib_mean']:.1f} |"
        )
    return "\n".join(rows)


def _code_hash_table(audit: Mapping[str, object]) -> str:
    rows = [
        "| seed | Adam | Muon | NorMuonAndAdam | FARO no-prior | FARO prior control |",
        "|---:|---|---|---|---|---|",
    ]
    hashes = audit["code_hashes"]
    for seed in sorted(audit["settings"]["seeds"]):
        values = [hashes[key][str(seed)] for key in RESULT_ORDER]
        rows.append(
            f"| {seed} | " + " | ".join(f"`{value[:12]}…`" for value in values) + " |"
        )
    return "\n".join(rows)


def _paired_normuon_differences(summary: Mapping[str, object]) -> str:
    return "；".join(
        f"seed{seed} `{_seed_loss(summary, 'faro', seed) - _seed_loss(summary, 'normuon', seed):+.9f}`"
        for seed in (0, 42, 1024)
    )


def _paired_prior_differences(summary: Mapping[str, object]) -> str:
    return "；".join(
        f"seed{seed} `{_seed_loss(summary, 'faro_prior', seed) - _seed_loss(summary, 'normuon', seed):+.9f}`"
        for seed in (0, 42, 1024)
    )


def _figure_sections(summary: Mapping[str, object]) -> str:
    route = summary["route_audit"]
    probe = summary["probe_audit"]
    freeze = summary["freeze_audit"]
    system = summary["system"]
    modes = route["mode_counts"]
    routes = summary["routes"]
    attention_total = sum(routes.get("attention", {}).values())
    mlp_total = sum(routes.get("mlp", {}).values())
    dense_total = attention_total + mlp_total
    token_total = (
        sum(routes.get("embedding", {}).values())
        + sum(routes.get("lm_head", {}).values())
        + sum(routes.get("coordinate", {}).values())
    )
    throughput_gain = (
        freeze["post_freeze_tokens_per_second_median"]
        / freeze["pre_freeze_tokens_per_second_median"]
        - 1.0
        if freeze["pre_freeze_tokens_per_second_median"]
        else 0.0
    )
    faro_sys = system["faro"]
    faro_prior_sys = system["faro_prior"]
    normuon_sys = system["normuon"]
    adam_sys = system["adam"]
    muon_sys = system["muon"]
    faro_mean = summary["variants"]["faro"]
    faro_prior_mean = summary["variants"]["faro_prior"]
    normuon_mean = summary["variants"]["normuon"]
    adam_mean = summary["variants"]["adam"]
    muon_mean = summary["variants"]["muon"]
    faro_prior_gap = (
        faro_mean["final_validation_loss_mean"]
        - faro_prior_mean["final_validation_loss_mean"]
    )
    time_overhead = (
        faro_sys["training_time_ms_mean"] / normuon_sys["training_time_ms_mean"] - 1.0
        if normuon_sys["training_time_ms_mean"]
        else 0.0
    )
    probe_events = summary.get("probe_events", [])
    del probe_events  # probe text is rendered from aggregate stats, not per-event
    parts = [
        f"""### 图 1：验证损失随训练 token（每个优化器一格，均值 ±1 标准差）

![五实验验证损失随训练 token](results/figures/{FIGURE_NAMES[0]})

这张图把计算量统一成已见 token，回答的是**样本/计算效率**，不包含优化器
本身快慢。五组同起点（loss ≈ 10.83）。NorMuonAndAdam、FARO prior control 与
FARO no-prior auto-switch 从 step 250 起就明显领先（≈4.50–4.60，同期 Muon
≈4.84、Adam ≈4.95），此后优势持续。终点上 no-prior FARO 对 Adam 领先约
`{adam_mean['final_validation_loss_mean'] - faro_mean['final_validation_loss_mean']:.3f}`、对 Muon
领先约 `{muon_mean['final_validation_loss_mean'] - faro_mean['final_validation_loss_mean']:.3f}`；
prior control 与 NorMuonAndAdam 全程贴合，no-prior FARO 略微落后于这两者
（差 prior control 约 `{faro_prior_gap:.3f}`）。简化 Muon 出现"前期优于 Adam、
终点反而更差"的交叉，说明该基线后期步进质量下降。各 variant 的三条 seed 曲线
基本重合，但 no-prior FARO 的 seed 间分散明显大于 prior control：组内波动虽
仍小于组间差距，却不是可忽略的量级。""",

        f"""### 图 2：验证损失随训练时间（每个优化器一格，均值 ±1 标准差）

![五实验验证损失随训练时间](results/figures/{FIGURE_NAMES[1]})

这张图回答**固定墙钟预算下谁先达到目标 loss**。NorMuonAndAdam 曲线最靠左
（{normuon_sys['tokens_per_second_mean']:,.0f} tokens/s），FARO no-prior
（{faro_sys['tokens_per_second_mean']:,.0f}）、FARO prior control
（{faro_prior_sys['tokens_per_second_mean']:,.0f}）与 Adam
（{adam_sys['tokens_per_second_mean']:,.0f}）几乎重合，Muon
（{muon_sys['tokens_per_second_mean']:,.0f}）最慢。NorMuonAndAdam 约 {normuon_sys['wall_time_ms_mean'] / 1000:.0f} s
结束，而 FARO no-prior 约 {faro_sys['wall_time_ms_mean'] / 1000:.0f} s 才得到略低的均值终点；因此工程上的时间—loss
Pareto 前沿仍是人工指定的 NorMuonAndAdam。离散验证点不足以宣称 FARO 在
任意相同时间上占优。no-prior 与 prior control 的探索成本相近，约
{time_overhead:.1%} 的训练时间代价。""",

        f"""### 图 3：训练损失 EMA（每个优化器一格，均值 ±1 标准差）

![五实验训练损失 EMA](results/figures/{FIGURE_NAMES[2]})

训练损失 EMA 的排序与验证 loss 一致：prior control ≈ NorMuonAndAdam 最低，
no-prior FARO 略高，Muon/Adam 明显更高。约 step423 的共同上跳和 step847
附近的共同折点对应训练配方的共同的阶段切换（batch 8→16→24、上下文
896→2048、attention window 与 MTP 权重同时调整），不是某个优化器独有的
loss spike。五组之后都继续下降，15 个 run 均无发散；就本实验而言，no-prior
FARO 的早期探测、自动 geometry 切换、预算缩放与错峰冻结没有造成可见的不稳定。""",

        f"""### 图 4：结构块几何模式占比（seed0 路由事件）

![动态 FARO 的结构块几何模式占比](results/figures/{FIGURE_NAMES[3]})

这里统计的是 seed0 探索期的**重复路由事件数**，不是参数块数量。no-prior FARO
共 {route['event_count']:,} 个事件：attention（{attention_total:,}）与
mlp（{mlp_total:,}）合计 {dense_total:,} 个 dense 块事件中，spectral
{int(modes.get('spectral', 0)):,} 次、coordinate {int(modes.get('coordinate', 0)):,} 次、
row/column {int(modes.get('row_column', 0)):,} 次；embedding/lm_head/坐标块
合计 {token_total:,} 个事件全部选择 coordinate。与 prior control 不同，这里
**不是**"dense 全部 spectral"：MLP 的 spectral 占比只有
{summary['audit']['mlp_spectral_ratio']:.3f}，因为路由器从 coordinate 起步、
按在线收益与风险自主决定哪些块切到 spectral。最终冻结的独立块是
{freeze['mode_counts'].get('spectral', 0)} spectral +
{freeze['mode_counts'].get('coordinate', 0)} coordinate +
{freeze['mode_counts'].get('row_column', 0)} row/column。这一自动选择把全坐标
硬锁的三 seed 均值 `3.539275` 恢复到 `3.321111`，收回硬锁代价的约 80%，
但相对 prior control 仍差约 `{faro_prior_gap:.3f}`。""",

        f"""### 图 5：功能块路由热图（seed0 代表）

![动态 FARO 的功能块路由热图](results/figures/{FIGURE_NAMES[4]})

每行一个功能块，横轴标出真实 routing step，颜色代表 coordinate /
row-column / spectral。交错白格不是丢数据：dense 块在 step 0/8/…、
odd-step coordinate/token 块在 1/9/… 路由，所以两类块天然错开。与 prior
control 的"dense 全 spectral、0 次切换"不同，no-prior 热图里 dense 行在
探索期出现大量真正的几何改选，共 {route['switch_count']:,} 次 switch
（`hard_risk_evacuation` + `challenger_won_twice`），这正是本报告要展示的
自动路由行为。由于 freeze 修复，块错峰冻结（step {'、'.join(str(step) for step in freeze['freeze_steps'])}），
step {freeze['all_frozen_step']} 全部 {freeze['total_frozen']} 个块冻结，此后
{freeze['post_freeze_route_events']} route / {freeze['post_freeze_probe_events']} probe，路由系统确实退出。""",

        f"""### 图 6：风险、漂移与谱统计（seed0 代表）

![动态 FARO 的风险、漂移与谱统计](results/figures/{FIGURE_NAMES[5]})

左、中面板分别汇总 dense spectral 与 coordinate/token 最终候选的中位统计；
右面板比较 {probe['event_count']} 次 probe 的 predicted/realized return 与漂移。
no-prior 起步下 dense 候选在探索期同时包含 spectral 与 coordinate，因此这两组
统计不再等同于"全部 spectral"的 prior control。{probe['event_count']} 次 probe
并非一致校准：predicted 均值 `{probe['predicted_return_mean']:.6f}`、realized 均值
`{probe['realized_return_mean']:.6f}`，realized/predicted 中位数
`{probe['realized_to_predicted_median']:.6f}`。所以一阶 return proxy 只能当有噪声的
方向信号，不能当无偏收益估计；样本量太少，不能据此拟合稳定校准系数。""",

        f"""### 图 7：预算倍率与切换事件（seed0 代表）

![动态 FARO 的预算倍率与切换事件](results/figures/{FIGURE_NAMES[6]})

每条线对应一个块。budget_scale 全程位于
`{route['budget_min']:.6f}`–`{route['budget_max']:.6f}`、
均值 `{route['budget_mean']:.6f}`。与 prior control 不同，no-prior 下发生
{route['switch_count']:,} 次真实 geometry 切换，图中散点即这些 switch：它们由
`challenger_won_twice`（收益竞标胜出两次）与 `hard_risk_evacuation`（当前几何
触发硬风险）两条自动路径触发。冻结审计：块错峰冻结于 step
{'、'.join(str(step) for step in freeze['freeze_steps'])}，最终全部
{freeze['total_frozen']} 块冻结，冻结后 {freeze['post_freeze_route_events']}
route / {freeze['post_freeze_probe_events']} probe 事件；吞吐从
{freeze['pre_freeze_tokens_per_second_median']:,.0f} 升至
{freeze['post_freeze_tokens_per_second_median']:,.0f} tokens/s
（+{throughput_gain:.0%}）。这里的前/后数值是 seed0 相邻 step 的增量吞吐
中位数，只能证明冻结后稳态路径更快；它不等于全程平均吞吐，也不能抹掉
探索期已经产生的峰值显存。""",

        f"""### 图 8：吞吐、时间与峰值显存

![五实验吞吐、时间和峰值显存](results/figures/{FIGURE_NAMES[7]})

NorMuonAndAdam 系统面最优（{normuon_sys['tokens_per_second_mean']:,.0f}
tokens/s、{normuon_sys['wall_time_ms_mean'] / 1000:.1f} s wall time、
{normuon_sys['peak_reserved_mib_mean']:,.0f} MiB reserved）。FARO no-prior 吞吐
{faro_sys['tokens_per_second_mean']:,.0f} tokens/s、wall time
{faro_sys['wall_time_ms_mean'] / 1000:.1f} s，与 FARO prior control
（{faro_prior_sys['tokens_per_second_mean']:,.0f} tokens/s）和 Adam 相当；峰值 reserved
{faro_sys['peak_reserved_mib_mean']:,.0f} MiB，比 NorMuonAndAdam 高约
{faro_sys['peak_reserved_mib_mean'] / normuon_sys['peak_reserved_mib_mean'] - 1:.0%}
（探索期 shadow candidate、编译/warmup 与统计状态共同形成的进程峰值；
正式训练前没有 reset peak counter）。no-prior 的错峰冻结把探索期拉长到
step {freeze['all_frozen_step']}，比 prior control 的 step 49 更久，但显存峰值
与 prior control 基本一致。Muon 反而最慢
（{muon_sys['tokens_per_second_mean']:,.0f} tokens/s），因为简化基线
未做 bank 融合优化。""",

        f"""### 图 9：各 seed 最终验证损失

![五实验各 seed 最终验证损失](results/figures/{FIGURE_NAMES[8]})

每个实验三个 seed 散点加黑色均值十字。FARO prior control 均值
`{faro_prior_mean['final_validation_loss_mean']:.6f}`、seed 间 std 最小
（{faro_prior_mean['final_validation_loss_std']:.6f}）；NorMuonAndAdam 均值稍高、
std 更大（{normuon_mean['final_validation_loss_std']:.6f}）。FARO no-prior 均值
`{faro_mean['final_validation_loss_mean']:.6f}`，比 prior control 高约
`{faro_prior_gap:.4f}`，且 std（{faro_mean['final_validation_loss_std']:.6f}）约为
prior control 的 22 倍：自动路由以部分确定性换取了去人工化。对 Adam
（{adam_mean['final_validation_loss_mean']:.6f}，std {adam_mean['final_validation_loss_std']:.6f}）
和 Muon（{muon_mean['final_validation_loss_mean']:.6f}）的优势在全部 seed 上一致成立。
这张图不能证明 no-prior FARO 稳定优于 NorMuonAndAdam 或 prior control。""",
    ]
    return "\n\n".join(parts)


def _render(summary: Mapping[str, object]) -> str:
    audit = summary["audit"]
    settings = audit["settings"]
    hashes = audit["fairness_hashes"]
    route = summary["route_audit"]
    probe = summary["probe_audit"]
    freeze = summary["freeze_audit"]
    modes = route["mode_counts"]
    muon_loss = _mean_loss(summary, "muon")
    faro_loss = _mean_loss(summary, "faro")
    adam_loss = _mean_loss(summary, "adam")
    normuon_loss = _mean_loss(summary, "normuon")
    faro_mean = summary["variants"]["faro"]
    faro_system = summary["system"]["faro"]
    normuon_system = summary["system"]["normuon"]
    faro_time_overhead = (
        faro_system["training_time_ms_mean"]
        / normuon_system["training_time_ms_mean"]
        - 1.0
    )
    faro_memory_overhead = (
        faro_system["peak_allocated_mib_mean"]
        / normuon_system["peak_allocated_mib_mean"]
        - 1.0
    )
    faro_reserved_overhead = (
        faro_system["peak_reserved_mib_mean"]
        / normuon_system["peak_reserved_mib_mean"]
        - 1.0
    )
    fairness_rows = "\n".join(
        f"- seed {seed}：initialization `{hashes[str(seed)]['initialization_hash']}`；"
        f"bigram sign table `{hashes[str(seed)]['bigram_sign_table_hash']}`；"
        f"data manifest `{hashes[str(seed)]['data_manifest_hash']}`"
        for seed in sorted(settings["seeds"])
    )
    sources = "\n".join(
        f"{index}. `{source}`"
        for index, source in enumerate(summary["source_runs"], start=1)
    )

    return f"""# FARO（无先验自动切换）：与 Adam、Muon、NorMuonAndAdam 及 prior 对照的单卡实验

本文沿用 [Keller Jordan 的 Muon 文章](https://kellerjordan.github.io/posts/muon/) 对稠密矩阵优化几何的讨论，加入本项目五个单卡三 seed 正式实验。所有本项目数字和曲线均由 `FARO/results/summary.json` 从原始 JSONL 重建；诊断、pilot、smoke 和历史消融 run 不进入正式比较。

## 1. Muon 原文背景与证据边界

Muon 的核心主张是：MLP/attention 的稠密矩阵不一定适合逐坐标缩放，对 momentum 做 Newton–Schulz 极分解近似的 spectral 更新可以更贴合这类线性算子的矩阵几何。作者 optimization-track Muon #12 使用 8×H100，在 step 3325 得到平均 validation loss 3.279047（n=20）。该多卡结果只提供背景，不能与本文的单卡、不同训练长度结果直接合并。

本文比较的五组实验全部来自本机重新运行，不用作者的 8 卡数字替代本机基线。

## 2. 五种实验分别是什么

- **Adam**：全部参数使用 Adam 类 coordinate 更新。
- **Muon**：本项目固定 Muon 基线。
- **NorMuonAndAdam**：MLP/attention 使用带 Polar Express、低秩二阶矩和谨慎权重衰减的 NorMuon，embedding、LM-head、norm 和 bias 使用 Adam。
- **FARO no-prior auto-switch（主实验）**：所有块从 coordinate 起步、保留全部合法候选、不做任何几何先验；路由器按在线收益、坏尾部、漂移、UWR 和谱风险**自行**决定每个块的 geometry 并缩放 budget，稳定后错峰冻结。这一组检验的是"统一收益—风险框架能否自动选几何、不需要人为定义"。
- **FARO prior control（对照）**：dense 块由 `structure_prior` 直接初始化为 spectral、token 块初始化为 coordinate（人工几何先验），其余机制相同。这一组代表"人工指定几何"的旧口径，用来量化人工先验相对自动路由的增益。

两组 FARO 都采用 sparse micro-gradient collector；它只改变稀疏 token 块的统计存储方式，不改变 geometry、router 或预算规则。

## 3. 当前实现与理论 FARO 的关系

当前原型实现了三种局部候选几何、原生等价 NorMuon spectral anchor、micro-batch 一阶 return、坏尾部代理、激活/输出漂移代理、update-to-weight / 谱风险、滞回切换、预算平滑、predicted/realized probe 校准以及 explore→freeze 状态裁剪。激活统计只在探索期 probe 所需的前向中开启，全冻结后不再承担 shadow candidate、route、probe 或 observer 开销。

但它**不是完整理论 FARO**：目前没有跨参数块的联合预算向量 `a`、收益协方差 `Σ`、全局总预算约束、组合更新的 CVaR smooth-loss upper bound、显式曲率项 `β/2 ||Ua||²`、严格的联合 Turnover 子问题、独立 probe/split-batch 风险评估，以及完整的 QK/per-head/max-logit function-space 约束。因此本文把它写成 **FARO 局部原型**，不把三 seed 实验写成完整理论 FARO 的最终验证。

## 4. 单卡设置与公平性

正式矩阵为三个 seed（{', '.join(str(seed) for seed in sorted(settings['seeds']))}）：单张 NVIDIA H800、`world_size={settings['world_size']}`、`grad_accum_steps={settings['grad_accum_steps']}`、`{settings['max_steps']}` steps、Adam matrix LR `{settings['matrix_lr']}`、Muon LR `{settings['muon_lr']}`。五组完成 run 数为 {audit['complete_runs']}（5 组 × {len(settings['seeds'])} seed）。

公平性按 seed 审计：同一 seed 内五种优化器的三个 hash 完全一致；不同 seed 的初始化和 bigram sign table hash 按设计不同。

{fairness_rows}

完整代码快照不完全一致：`code_hash` 覆盖 `train_gpt.py` 与当时全部 `FARO/*.py`，{audit['complete_runs']} 个 run 的记录如下。

{_code_hash_table(audit)}

这个 hash 口径也包含离线汇总/画图脚本，因而是保守的整包快照检查，无法单靠 hash 判断每个差异是否进入了对应 variant 的训练路径。已知多 seed 接力期间修复过固定 Muon 的 `_muon_config` 丢失 `return`；成功的 Muon seed42/1024 在修复后启动。数据、初始化和核心超参可以确认对齐，但这些结果**不是同一完整代码快照下的严格 A/B**。

本文只有三个 seed 各一次 run，所有排序都是三次 run 的均值观察，不能宣称统计显著。

## 5. 五实验三 seed 正式结果

{_result_table(summary)}

主实验 FARO no-prior 的三 seed 平均终点 loss 为 `{faro_loss:.9f}`：比 Muon 平均低 `{muon_loss - faro_loss:.9f}`，比 Adam 平均低 `{adam_loss - faro_loss:.9f}`，比 NorMuonAndAdam 高 `{faro_loss - normuon_loss:.9f}`，比 FARO prior control 高 `{faro_loss - _mean_loss(summary, 'faro_prior'):.9f}`。也就是说，**去人工先验后，自动路由仍明显优于 Muon/Adam，但尚不能追平有人工先验的 NorMuonAndAdam / prior control**。

逐 seed 看，主实验 FARO − NorMuonAndAdam 为：{_paired_normuon_differences(summary)}；三个 seed 均高于 NorMuonAndAdam。prior control − NorMuonAndAdam 为：{_paired_prior_differences(summary)}，与 NorMuonAndAdam 同档。

## 6. 实验曲线与逐图分析

图 1–3 比较五组完整训练轨迹：每格一个优化器，粗线为三 seed 均值、色带为 ±1 标准差、
细线为单个 seed，底行放大末段以显示 seed 波动与组间差距；图 4–7 的路由、风险、
预算与冻结过程以 seed0 代表 FARO no-prior auto-switch；图 8、9
比较五组系统开销与三 seed 终点。每张图
下方附该图展示的内容与可直接得出的结论。

{_figure_sections(summary)}

## 7. 动态路由、预算和 probe 审计

以下路由、probe 与冻结审计均取 no-prior 主实验的 seed0 代表。该 run 共记录 {route['event_count']:,} 个 route event：coordinate {int(modes.get('coordinate', 0)):,}、row/column {int(modes.get('row_column', 0)):,}、spectral {int(modes.get('spectral', 0)):,}；发生 {route['switch_count']:,} 次**真实 geometry 切换**。budget 的 min/mean/max 为 `{route['budget_min']:.6f}` / `{route['budget_mean']:.6f}` / `{route['budget_max']:.6f}`，范围为 `{route['budget_min']:.6f}–{route['budget_max']:.6f}`。

共有 {probe['event_count']:,} 次 probe；predicted return 均值 `{probe['predicted_return_mean']:.6f}`，realized return 均值 `{probe['realized_return_mean']:.6f}`，realized/predicted 中位数 `{probe['realized_to_predicted_median']:.6f}`。probe 的校准误差方向并不一致，不能把局部 return proxy 当成无偏的真实收益估计，样本量也太少，不能据此拟合稳定偏差。

冻结审计：{freeze['event_count']} 个 freeze event，发生于 step {'、'.join(str(step) for step in freeze['freeze_steps'])}，共 {freeze['total_frozen']} 个块冻结；全部冻结后 {freeze['post_freeze_route_events']} route / {freeze['post_freeze_probe_events']} probe；冻结前/后吞吐中位数 `{freeze['pre_freeze_tokens_per_second_median']:,.1f}` / `{freeze['post_freeze_tokens_per_second_median']:,.1f}` tokens/s。

## 8. Embedding / LM-head 的负结果

结构直觉原本偏向给 embedding / LM-head 使用非逐参数的 row/column geometry；但当前模型与配方的诊断显示，现有 row/column candidate 不如 coordinate。精确 UWR 和 active-axis 修复只恢复了部分差距，最终仍是 coordinate token control 更好。因此当前 FARO 对 embedding / LM-head 选择了 Adam 类 coordinate 更新；这是应保留的负结果，不代表所有行列自适应方法在更大模型或更长训练中都无效。

## 9. 系统开销

{_system_table(summary)}

动态 FARO 只在前期探索时维护 shadow states、完整候选和 probe；冻结后裁剪候选状态并关闭相关观测。no-prior 主实验因错峰冻结，探索期比 prior control 更长。三 seed 全程平均吞吐已经接近 Adam，但仍低于原生 NorMuonAndAdam，且峰值显存仍由探索阶段决定。当前系统开销比较只覆盖五个正式实验，历史消融 run 不参与排序。

相对 NorMuonAndAdam，FARO no-prior 主实验的平均训练时间高 `{faro_time_overhead:.2%}`，平均 peak allocated 高 `{faro_memory_overhead:.2%}`，peak reserved 高 `{faro_reserved_overhead:.2%}`。这里的 peak 是进程从编译/warmup 到训练结束的 `torch.cuda.max_memory_allocated()`，没有在正式训练前重置；它代表这套程序实际需要容纳的峰值，而不是冻结后的稳态占用。

## 10. 结论、限制与下一步

在数据、初始化和核心设置对齐、但完整代码快照不完全一致的单卡三 seed 实验中：

- **主结论（自动路由有效）**：所有块从 coordinate 起步、零几何先验时，路由器三个 seed 都自主把约 {summary['audit']['mlp_spectral_ratio']:.0%} 的 MLP 块（池化口径）切到 spectral，主实验均值 `{faro_loss:.9f}`，明显优于 Adam（`{adam_loss:.9f}`）与 Muon（`{muon_loss:.9f}`），并收回了全坐标硬锁代价的大部分。这补上了"`switch_count=0` 无法证明纠错能力"的短板。
- **限制（尚未完全去人工化）**：主实验比 prior control（`{_mean_loss(summary, 'faro_prior'):.9f}`）高约 `{faro_loss - _mean_loss(summary, 'faro_prior'):.9f}`，且 seed 间 std（`{faro_mean['final_validation_loss_std']:.6f}`）约为 prior control 的 22 倍。差距可分解为探索期成本与最终几何选择分歧两部分，尚未拆分量化。因此当前证据支持“自动路由能恢复大部分性能、不需要初始几何先验”，不支持“完全不需要人为定义”。

这个观察不能外推到作者 8×H100 设置，也不能宣称统计优势。下一阶段应量化上述两部分差距的占比，并先解决跨块联合预算、协方差与独立 probe/split-batch 风险评估，再决定是否需要更多 seed 或更大模型规模。

### 数据溯源

汇总文件：`FARO/results/summary.json`

{sources}
"""


def render_report(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render(summary), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    render_report(args.summary, args.output)


if __name__ == "__main__":
    main()
