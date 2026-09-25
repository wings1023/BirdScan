# BirdScan 当前识别结果决策链

本文记录 `scan_birds.py` 当前正式扫描流程。这里的 det01 是 MegaDetector 返回、经过 animal 类别与置信度筛选后的**第一个** detection；编号保留检测返回顺序。项目有意把它视为与原图 baseline 对应的主主体 crop 视图，但代码不保证它是最大 bbox 或最高置信度 detection。

## Pipeline 总览

```text
原图 → 图片可读性检查 → 原图 BioCLIP baseline Top-1
     → MegaDetector V6 → animal 类别和 confidence >= 0.15 → 按返回顺序编号 det01、det02…
     → bbox 按原图边界 clamp → 原 bbox 面积占比 < 0.0003 则跳过
     → 对保留的 bbox 四边扩 margin 0.30 并再次按图像边界 clamp
     → 保存 diagnostics/crop_runtime 中的 crop → 同一 BioCLIP 模型批量分类，各 crop 取 Top-1
     → 仅 det01 根据 0.08 size gate 决定是否替代 baseline
     → 仅 det02+ 根据 0.90 分数门槛汇总 additional
     → predictions.csv / species_summary.csv / uncertain.csv / run_summary.txt
```

原图 baseline 先于 detector/crop 推理。`--no-crop` 跳过整个 MegaDetector、面积过滤、crop 与 additional 流程，只使用原图 BioCLIP 结果。默认启用 crop；默认模型为 BioCLIP 2.5，也可选 BioCLIP 2。原图推理使用 `--top-k`（默认 1），最终选择该图得分最高的一行作为 baseline Top-1；crop 推理固定只请求 Top-1。两阶段都通过同一个分类器，并使用失败批次拆分重试。

## Detection、crop 与角色

MegaDetector 返回的结果仅取 `class_id == 0`（animal），随后保留 confidence `>= 0.15` 的 detection；**之后**按现有顺序从 1 编号。这个编号先于 bbox 面积过滤，因此 det01 被过滤后，det02 仍叫 det02，不会改号。

对于原图宽高 `W × H`，先将 detector 的 `(x1, y1, x2, y2)` 限制在图像边界内，并确保 `x2 >= x1`、`y2 >= y1`。代码计算：

```text
bbox_area_ratio = (x2 - x1) × (y2 - y1) / (W × H)
```

这里使用的是 **clamp 后、margin 前**的 detector bbox；小于 `--min-bbox-area-ratio` 的 detection 立即跳过。等于或大于阈值的 detection 才扩边、生成并保存 crop，进入 crop 分类。被跳过者不产生本轮 crop、不进入分类或物种选择；跳过次数计入 `Small bbox detections skipped`。当阈值设为 0 时，零面积框可通过面积门槛，但随后的无效 crop 边界检查仍可能跳过它。

默认 margin `m = 0.30`：左、右分别扩原 bbox 宽度的 `m`，上、下分别扩原 bbox 高度的 `m`。下界取 floor，上界取 ceil，并再次 clamp 到图像边界。未触边时，几何上的 crop 宽高各约为 bbox 的 `1 + 2m = 1.6` 倍；像素取整会带来小幅差异。margin 不参与 `bbox_area_ratio`。有效 crop 保存到 `diagnostics/crop_runtime/`，文件名保留原始 detXX 序号。crop 文件保存成功但分类没有返回结果时，该 detection 不成为有效分类候选；分数低本身不算分类失败。

**det01** 是唯一能决定 crop primary 的候选。它有分类结果且其 `bbox_area_ratio <= 0.08` 时，crop Top-1 直接成为最终 primary，不比较 baseline 分数，也没有额外的 primary 分数门槛。det01 超过 `0.08`、被前置过滤、crop 无效或分类无结果时，保留 baseline；det01 在任何情况下都不进入 additional。

**det02+** 永远不能接替 primary。其成功分类的 Top-1 仅在物种名非空、分数 `>= 0.90`、物种不同于最终 primary 时进入 additional。同种只保留最高分；additional 不反向改变 primary。`0.08` 不限制 det02+ 的 crop 分类或 additional 资格。

## 关键参数

| 参数或常量 | 当前值 | 实际作用 |
| --- | ---: | --- |
| MegaDetector confidence threshold | `0.15` | detector 调用及 animal detection 筛选；低于它的不编号、不生成 crop。 |
| `--min-bbox-area-ratio` | `0.0003` | crop 模式的前置硬过滤：小于阈值的原 bbox 不扩边、不保存、不分类、不参与结果；等于阈值保留。 |
| det01 size gate | `0.08` | 只决定有分类结果的 det01 是否直接替代 baseline；不阻止 crop 生成或推理，也不约束 det02+ additional。 |
| additional score threshold | `0.90` | 只约束 det02+ 的 additional Top-1；等于阈值保留，不影响 primary。 |
| crop margin | `0.30` | bbox 四边各扩对应宽/高的 30%，然后边界 clamp；不参与面积占比计算。 |
| `--threshold`（uncertain） | `0.5` | 最终 primary score 低于它才写入 `uncertain.csv`；不改变 detection 或 primary 选择。 |
| `--batch-size` | `16` | BioCLIP 原图及 crop 分类批次大小；失败批次递归拆分重试。 |
| `--prompt-count` | `80` | BioCLIP 2.5 每种候选物种的文本模板数；BioCLIP 2 分支不使用该值构造候选 embedding。 |

### `--min-bbox-area-ratio` 的经验说明

这是原始 detector bbox 面积占整张原图面积的**相对阈值**，决定多小的 detection 仍进入 crop classification。默认值 `0.0003` 主要来自 Sony A7M5 约 33MP 原图的实际测试经验，尚未经过系统的跨机型、跨分辨率验证。对于约 3300 万像素的原图，它对应约 `3300 万 × 0.0003 = 9900` 像素的原始 bbox 面积，约为一万像素量级。

**分辨率因素：**更高分辨率的图像中，可酌情调低阈值，以保留仍有足够细节的小目标；更低分辨率的图像中，可酌情调高阈值，减少极小目标带来的噪音和高置信误判。这是经验性调参方向，并非已验证的跨分辨率规律。

**additional 候选偏好：**较低阈值会让更多微小 detection 进入 crop 分类，有机会发现远距离、占比很小的其他物种，也可能增加错误或重复的原始候选及人工复核负担；输出中的同种 additional 仍按既有规则去重。较高阈值可减少这类噪音与复核工作，但可能漏掉可识别的小目标。偏向召回更多 additional 候选时可酌情调低；偏向减少噪音时可酌情调高。这些方向同样属于经验建议，尚未经系统验证。

该参数不是物种置信度阈值，也不直接决定 additional 是否写入结果。通过面积过滤的 det02+ 仍须满足后续 Top-1 score `>= 0.90`、物种不同于 final primary 等既有条件。

## 决策表

以下面积判断均针对 clamp 后、margin 前的原 bbox，且以默认阈值说明。

| 场景 | 最终 primary | crop / additional 行为 |
| --- | --- | --- |
| 无 detection | baseline；若 baseline 无结果则该图失败 | 无 crop、无 additional。 |
| det01 ratio `< 0.0003` | baseline；det02+ 不接替 | det01 不生成 crop；其余 detection 各自继续过滤与分类。 |
| det01 ratio `0.0003` 至 `0.08`，含两端，且分类有结果 | det01 crop Top-1 | det01 不进 additional；合格 det02+ 可进入。 |
| det01 ratio `> 0.08`，且分类有结果 | baseline | det01 仍生成并分类 crop，但不进 additional；合格 det02+ 可进入。 |
| det01 crop 无效、保存失败或分类无结果 | baseline | det02+ 不接替 primary；有成功分类者仍可按规则进入 additional。 |
| det02+ Top-1 score `< 0.90` | 不改变当前 primary | 不进入 additional。 |
| det02+ Top-1 score `>= 0.90` | 不改变当前 primary | 物种非空、与 primary 不同且按同种最高分去重后进入 additional。 |
| det02+ ratio `> 0.08` | 不改变当前 primary | 仍生成、分类 crop；满足 `0.90` 等规则可进入 additional。 |
| 所有 detection 都因最小面积门槛跳过 | baseline；若 baseline 无结果则该图失败 | 不生成 crop、不执行 crop 分类；跳过数逐 detection 计入 summary。 |
| baseline 无结果，但 det01 ratio `<= 0.08` 且分类成功 | det01 crop Top-1 | 可恢复为成功图片；det02+ 仍只参与 additional。 |
| baseline 无结果，且 det01 未形成可用 primary | 无最终 primary，该图计为失败 | det02+ 不接替；该图不进入正式 `predictions.csv` / 物种汇总。 |
| crop inference 批次失败 | 有效 det01 结果仍可成为 primary，否则回退 baseline | 批次递归拆分到单个 crop；仍失败的 crop 无分类结果，不参与选择。 |

detector 初始化失败会导致本次 crop 模式运行报错；逐图 detection 或 crop 生成异常会跳过该图剩余 crop 生成并继续最终选择，已有 baseline 可作为该图 primary。

## 报告字段

`predictions.csv` 对每张**有最终 primary** 的原图写一行；`file_name` 始终是相对照片目录的原图路径。`鸟种编号`、`中文名`、`拉丁学名`、`英文名称` 来自最终 primary 对应的候选物种表记录；`score` 是该 primary 的 BioCLIP 分数。下列 `*_species` 诊断列使用显示名：中文名优先，其次英文名，最后拉丁名；内部分组与去重使用拉丁学名。

| 字段 | 语义 |
| --- | --- |
| `baseline_top1_species` / `baseline_top1_score` | 原图 BioCLIP 的最高分候选与分数；无结果时为空。 |
| `crop_top1_species` / `crop_top1_score` | **原始 det01** 有效 crop 的最高分候选与分数；即使 `> 0.08` 未被采用，也可记录；无 det01 分类结果时为空。 |
| `primary_species` / `primary_score` | 最终 primary 的物种与分数；与该行物种列及 `score` 对应。 |
| `final_source` / `crop_used` | 分别为 `crop` / `true` 或 `baseline` / `false`，表示最终 primary 来源。 |
| `selected_crop_file` | 仅在 det01 crop 成为 primary 时，记录其相对 output-dir 的 crop 路径；否则为空。 |
| `detection_confidence` / `bbox_area_ratio` | 仅在 det01 crop 成为 primary 时，记录其 detector confidence 与原 bbox 面积占比；否则为空。 |
| `additional_species` / `additional_scores` / `additional_count` | det02+ 合格物种、逐项对应的 Top-1 分数和去重后数量；前两列用 `|` 分隔。 |

`species_summary.csv` 只汇总有正式 `predictions.csv` 行的图片：按拉丁学名统计 `primary_count`、`additional_count`，以及 primary/additional 出现中的最高 `max_score` 和对应原图 `best_image`。`uncertain.csv` 只收录最终 primary 的报告分数（写出时保留六位小数）严格低于 `--threshold` 的成功图片。`run_summary.txt` 记录扫描、成功、失败、不同 primary 物种数、各阶段时间、`Min bbox area ratio` 和 `Small bbox detections skipped`；skipped 是 detection 数量，不是图片数量。前置过滤的 detection 不会进入这些物种结果。

## 输出目录的当前行为

默认 output-dir 随模型选择：BioCLIP 2.5 为 `reports/bird_report_bioclip25/`，BioCLIP 2 为 `reports/bird_report/`。同一 output-dir 重跑时，三个 CSV 和 `run_summary.txt` 会重写覆盖；`diagnostics/crop_runtime/` 只创建目录，不自动清理。旧 crop 可能残留，因此正式 benchmark 或前后对比运行宜使用独立的 `--output-dir`。
