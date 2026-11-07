# BirdScan v1.2.1 (planned / 拟发布)

## Update / 更新

- `predictions.csv` appends `additional_crop_files`. Its pipe-separated paths match `additional_species` and `additional_scores` in order and are relative to the report directory. Each path identifies the crop that supplied the retained score. No new option is required; the field is empty when there are no additional species. Identification and selection rules are unchanged.
- `predictions.csv` 末尾新增 `additional_crop_files`。该列路径以 `|` 分隔，与 `additional_species`、`additional_scores` 逐项对应，相对于报告目录；每个路径指向最终采用分数的裁剪图。无需新参数；没有 additional 时为空。识别和筛选规则不变。

## Compatibility / 兼容性

Existing column names and order are preserved, with one extra final column. Consumers that require an exact column count or header must be updated. Historical reports do not gain the field automatically; existing crop images alone cannot reliably establish the species-to-crop mapping.

保留原有列名与顺序，末尾追加一列。要求固定列数或完整表头的下游程序需要适配。历史报告不会自动获得新列，仅凭已有裁剪图片无法可靠还原物种与裁剪图的对应关系。

Details / 详细机制：[selection pipeline](selection_pipeline.md).
