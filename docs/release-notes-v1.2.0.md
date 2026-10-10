# BirdScan v1.2.0

## Highlights

- **Primary and additional species:** each successful photo now has one primary Top-1 row. Other detections (det02+) can add distinct species with scores of at least 0.90; repeated species keep their highest score and never replace the primary.
- **Revised crop selection:** crop primary now comes only from the original det01, replacing the previous highest-score crop selection. If det01 is classified successfully and its bbox area ratio is at most 0.08, its Top-1 is used without comparing scores with the original-image baseline. Otherwise, BirdScan keeps the baseline; later detections cannot take over.
- **Small-box filtering:** new `--min-bbox-area-ratio` defaults to 0.0003. Smaller boxes are skipped before crop creation and classification, without renumbering detections. Area is measured before margin expansion; crop margin increases from 0.20 to 0.30.
- **Clearer reports and progress:** species summaries count primary and additional appearances separately and select the highest score and corresponding photo across both. Only photos with a final primary result contribute. Run summaries include small-box filtering statistics, and MegaDetector progress shows processed photos and total detections with less log noise.
- **Startup and recovery fixes:** missing official MegaDetector V6 weights are downloaded automatically. Invalid candidate-embedding caches are logged and rebuilt for both BioCLIP models, without treating memory or device errors as invalid caches. Success/failure counts and exit status now reflect the final primary result, including recovery from baseline failure by det01 crop. Windows subdirectory diagnostics now match report paths correctly.
- **Examples and documentation:** adds 12 real bird photographs with manual ground truth, screenshots, clearer installation guidance, and a selection-pipeline reference. The sample includes difficult and known failure cases and is not an overall accuracy benchmark.

## 中文更新摘要

- **Primary 与 additional 鸟种：**每张成功照片改为一行 primary Top-1。其他检测（det02+）可补充得分至少 0.90 的不同物种；同种保留最高分，不覆盖 primary。
- **Crop 选择调整：**从选择最高分 crop 改为仅由原始 det01 决定 crop primary。det01 分类成功且 bbox 面积占比不超过 0.08 时，直接采用其 Top-1，不与原图 baseline 比分；否则保留 baseline，后续检测不接替。
- **小框过滤：**新增 `--min-bbox-area-ratio`，默认 0.0003。小于门槛的框不生成 crop、不分类，检测编号保持不变；面积按扩 margin 前的 bbox 计算，crop margin 从 0.20 调整为 0.30。
- **报告与进度改进：**物种汇总分别统计 primary / additional，并从两类结果中选取最高分及对应照片；仅有最终 primary 的照片参与汇总。运行摘要增加小框过滤统计，MegaDetector 显示已处理照片数和检测总数，减少逐图日志。
- **启动与恢复修复：**缺少 MegaDetector 官方权重时自动下载；两个 BioCLIP 模型的无效候选 embedding 缓存会记录原因并重建，内存或设备错误不会被当成坏缓存。成功/失败计数及退出状态以最终 primary 结果为准，包括 baseline 失败后由 det01 crop 恢复成功的情况；修复 Windows 子目录诊断与报告路径匹配。
- **示例与文档：**增加 12 张真实观鸟照片及人工真值、截图和决策链文档，完善安装说明。样本包含困难及已知失败案例，不用于估计整体准确率。

## Compatibility notes / 兼容性提示

- **CSV schema changes:** `predictions.csv` removes `rank` and adds `primary_species`, `primary_score`, `additional_species`, `additional_scores`, and `additional_count`. `species_summary.csv` replaces `top1_count` / `topk_count` with `primary_count` / `additional_count`; `max_score` and `best_image` now include additional appearances. Update scripts that rely on the old columns or multiple candidate rows per photo.

  **CSV 字段变化：**`predictions.csv` 移除 `rank`，新增 primary / additional 相关列；物种汇总改用 `primary_count` / `additional_count`，最高分及最佳图片同时考虑 additional。依赖旧列名或每图多行结果的脚本需要适配。

- **Top-K behavior:** `--top-k` defaults to 1 (previously 3). Larger values affect original-image inference only; crop classification requests Top-1, and reports still contain one primary row per successful photo. `--threshold` controls only `uncertain.csv`, based on the final primary score; it does not filter additional species or detections.

  **Top-K 行为：**默认值从 3 改为 1；增大参数只影响原图内部推理，不恢复旧 Top-K 报告。Crop 固定取 Top-1，`--threshold` 仅用于最终 primary 的不确定结果判定。

- **Diagnostic names and crop fields:** species display columns prefer Chinese, then English, then Latin; the primary `拉丁学名` column remains the identifier. `additional_species` and `additional_scores` are pipe-separated in matching order. `selected_crop_file`, `detection_confidence`, and `bbox_area_ratio` are populated only when det01 supplies the primary. `crop_top1_species` / `crop_top1_score` describe det01's classification even when its bbox exceeds the size gate.

  **诊断字段：**物种显示列优先中文名，其次英文名，最后回退拉丁名。Additional 物种与分数以 `|` 按相同顺序分隔。选中 crop 的路径和检测信息仅在采用 det01 时填写；det01 超过面积门槛时，其分类结果仍可记录在 `crop_top1_species` / `crop_top1_score` 中。

- **Output location:** reports must be outside the input photo directory, including after resolving symbolic links. An equal or nested output directory is rejected before scanning or creating output. The default report directory is subject to the same restriction.

  **输出路径：**报告目录不能与输入照片目录相同或位于其子目录中，符号链接解析后的路径也受此限制。默认报告目录同样适用；必要时显式指定外部 `--output-dir`。

- **Removed utilities:** `summarize_report.py`, `experiments/inspect_embeddings.py`, and `experiments/inspect_neighbor_consistency.py` are removed. The main scan directly generates the current species summary. Retained experimental scripts may use historical best-score, margin, or Top-K settings that differ from the production pipeline.

  **工具调整：**旧审阅后处理及两个诊断脚本已移除，主扫描直接生成当前物种汇总。保留的实验脚本可能使用历史 best-score、margin 或 Top-K 设置，与正式流程不同。
