# v1.2.0 发布核查

核查日期：2026-10-10。范围：`v1.1.0`（`92d96dc`）至 `main` / HEAD（`371046d`），共 9 个提交。开始时工作区干净。

## 提交与实际变化

| Commit | 实际变化 |
| --- | --- |
| `37cd153` | 缺少官方 MegaDetector checkpoint 时自动下载到 Torch Hub 缓存；临时文件下载完成后替换目标，失败清理临时文件。 |
| `568ff5e` | 双语安装指引：PowerShell 无激活用法、设备默认值、输入图片格式和首次下载说明。 |
| `ccdb537` | 正式 Top-1 与 primary/additional 报告；仅 det01 决定 crop primary；margin 0.30；物种显示名、Windows 相对路径匹配、最终失败计数与退出状态、detector 进度；移除 summarize_report.py，新增离线 dual-candidate 评测，相关评测接受无 rank 输入。 |
| `88982a0` | 移除 inspect_embeddings.py 和 inspect_neighbor_consistency.py；澄清保留实验脚本的历史策略与正式流程差异。 |
| `f2b3a32` | 新增最小 bbox 面积过滤，保留原始 detection 编号、补充边界/恢复测试和决策链文档。 |
| `8c918de` | 增加 12 张照片、人工真值、示例说明及截图。 |
| `12de346` | 真值 CSV 权限修正，无内容变化。 |
| `58a795f` | 双语项目介绍文字调整。 |
| `371046d` | 两个 BioCLIP 分支及旧版缓存读取均使用验证/恢复机制；拒绝输入目录内的输出路径。 |

## 核查结论

代码依据：`scan_birds.py` 的 `parse_args`、`select_crop_result`、`select_primary_and_additional`、`main`、`write_reports`、`load_candidate_embedding_cache`、两个 classifier 构建函数和 `validate_output_directory`；`megadetector_utils.py` 的 `official_checkpoint` / `load_detector`。已对照范围内 diff 和 v1.1.0 README，避免把已有 BioCLIP 2.5、默认 crop、候选表与缓存功能重新列为新增。

- 双语 README 的参数、默认值、primary/additional 选择规则和 CSV 字段与当前代码一致。本次只补充遗漏的输出目录限制；决策链文档同步补充该限制。
- `--top-k` 默认 3 → 1；显式增大仍只影响原图内部推理。正式 predictions 无 rank，每图一行；汇总的 top1_count/topk_count 改为 primary_count/additional_count。
- det01 指 animal / confidence 筛选后的第一个 detection，不保证最大或最高置信度。面积过滤在编号后进行，size gate 只约束 det01 primary；det02+ additional 不受 0.08 限制。
- additional 同种保留最高分，不重复 primary；汇总只统计有最终 primary 的图片，max_score/best_image 同时考虑 primary 和 additional。
- 缓存读取验证元数据、二维形状、候选数及有限值；坏缓存记录原因并重建，资源不足异常继续抛出。设备传输、编码、保存位于缓存读取恢复边界之外。已有完整模板旧缓存迁移逻辑被保留。
- 输出目录相同或嵌套于输入目录时提前拒绝，包含路径解析及现有祖先的文件系统身份检查；默认输出也受此限制。重跑仍覆盖三个 CSV 和运行摘要，旧 crop 不自动清理。

## 验证

- `.venv/bin/python -m unittest -v`：40 项通过，退出码 0。覆盖 crop 选择及面积边界、additional 阈值/去重、报告字段和汇总边界、显示名、Windows 路径、最终恢复/失败状态、缓存验证/OOM、checkpoint 下载/失败清理、输出目录限制及参数解析。
- `.venv/bin/python scan_birds.py --help`：退出码 0；所有文档中的 CLI 选项均存在。代码核对 batch-size=16、threshold=0.5、device=mps、model=bioclip25、prompt-count=80，以及 crop 默认开启和按模型分配默认输出路径。
- `git diff --check`：通过。
- 测试使用模拟分类器、detector、Torch/cache 和下载接口；未下载大型模型、未执行完整推理，未验证真实 GPU、模型下载服务或识别准确率。Windows 路径验证为 PureWindowsPath 测试，未运行 Windows 系统。

发布说明见 `release-notes-v1.2.0.md`。发布前核查确认：当前分支、`main` 与 `origin/main` 均为 `371046d`，远程不存在 `v1.2.0` Tag 或 Release；工作区仅含本次 README 英中版、决策链文档及两份发布文档。Release Notes 的 Markdown 结构已检查，`git diff --check` 通过；仓库未配置 Markdown lint 工具。发布说明将以本文件原文作为 GitHub Release 正文。
