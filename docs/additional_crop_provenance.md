# additional 来源追踪

检测及裁剪链路：`main` 调用 MegaDetector `single_image_detection` → `extract_animals` → 检测置信度至少 0.15 → 框坐标截断 → 框面积至少 `--min-bbox-area-ratio`（默认 0.0003）→ `padded_box`（margin 0.30）→ 保存裁剪文件。检测编号保留原编号，过滤后不会重排。文件名是原照片绝对路径 SHA-1 前 12 位加 `__detXX`，路径相对于报告目录。

原图通过 `predict_resiliently` 分类；所有有效裁剪同样分类，crop Top-K 固定为 1。以返回结果中的实际文件路径关联 `record`，不通过列表顺序关联。`select_primary_and_additional` 仅允许原始 det01 作为 primary：面积不超过 0.08 时采用裁剪，即使其分数低于原图；否则采用原图。det01 缺失也不会让 det02 顶替。

additional 只使用 det02+ 的 Top-1：排除空物种、primary 同种及分数低于 0.90 的候选。按 detection 编号遍历，以拉丁名去重，严格更高分才替换整个候选；相同分数保留较早检测。导出顺序为物种首次合格出现的顺序，替换来源不改变物种位置。分数来自该裁剪的分类结果，不是检测置信度。

去重结果一直保留 `prediction` 和 `record.crop_file`，本次仅补充导出该字段。`additional_species`、`additional_scores`、`additional_crop_files` 从同一 additional 列表依序生成；显示名转换不改变顺序。最后一列路径相对于当前报告目录；无 additional 时为空。不改变分类、阈值、去重或 primary 逻辑。

旧报告只有原图/primary/det01 诊断及裁剪图片，没有每个 det02+ 的分类输出，不能可靠恢复分数对应的裁剪目标，必须重新分类。已有裁剪文件名只能定位检测目标，无法证明其物种。此次选择在独立目录重跑完整流程。

兼容性检查：仓库实验读取器通过 DictReader 和所需列集合读取 CSV，可忽略追加列；原有字段顺序及值结构不变。历史发布说明保留原样，中文及英文 README 和 selection_pipeline 已补充新列。回归测试从真实 main 聚合路径检查列尾位置、三列长度、分数/物种与源检测对应及文件存在，覆盖多目标、重复最高分、同分、primary 同种过滤、阈值过滤、缺失分类、单目标及 det01 面积过滤。
