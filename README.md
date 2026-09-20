# BirdScan

一个只读原始照片的批量鸟类识别命令行工具。它使用 BioCLIP 2 在候选物种表的拉丁学名范围内分类，并把结果映射回中文名、英文名和鸟种编号。

## 环境

macOS Apple Silicon 示例：

```bash
cd ~/BirdScan
source .venv/bin/activate
python --version
python -m pip show pybioclip
```

已安装的 `pybioclip 2.1.6` 在 Python 中的导入名是 `bioclip`。脚本默认使用 `--device mps`。候选表支持 UTF-8 CSV 或标准 `.xlsx`，不需要 `openpyxl`。

候选表第一行必须有以下四个字段，且每行的“拉丁学名”必须唯一：

```text
鸟种编号,中文名,拉丁学名,英文名称
```

## 使用

```bash
cd ~/BirdScan
.venv/bin/python scan_birds.py "/path/to/photos" \
  --species-file "/path/to/species.xlsx"
```

可选参数：

```text
--top-k 5             每张照片输出的候选数，默认 5
--batch-size 16       每批推理照片数，默认 16
--threshold 0.5       Top-1 分数低于此值时进入 uncertain.csv
--device mps          默认 mps，也可指定 cpu
--output-dir bird_report  报告目录
```

脚本递归读取 `.jpg`、`.jpeg`、`.png`。它不会修改、移动、重命名原照片，也不会自动裁鸟或调用 MegaDetector。

首次运行会在 `~/.cache/birdscan/` 写入候选拉丁学名的 BioCLIP 文本向量缓存。缓存键包含 BioCLIP 模型、预训练标识和完整且有顺序的候选拉丁学名列表；候选表不变时，后续运行会直接复用它。终端和 `run_summary.txt` 会分别列出候选表读取、模型加载、候选文本编码或缓存加载、图片推理、报告写出和总耗时。

## 输出

默认在当前目录生成 `bird_report/`：

- `predictions.csv`：每张成功照片的 Top-K 结果，包含相对文件名、排名、物种四字段及分数。
- `species_summary.csv`：按 Top-K 出现物种汇总 Top-1/Top-K 次数、最高分及对应最佳照片。
- `uncertain.csv`：Top-1 分数低于阈值的照片。
- `run_summary.txt`：扫描数、成功数、失败数、Top-1 不同物种数、耗时、速度；若有损坏文件或孤立的 BioCLIP 推理错误，也会列在这里。

分数是在候选物种表的所有拉丁学名之间归一化的概率，不是“照片中一定有鸟”的判定。第一版不检测或裁切鸟，因此鸟很小、被遮挡或照片没有鸟时，应重点人工复核 `uncertain.csv`。

## 汇总审阅后处理

`summarize_report.py` 只读取已有的 `species_summary.csv`，不会重新运行 BioCLIP，也不会修改输入文件：

```bash
.venv/bin/python summarize_report.py path/to/species_summary.csv
```

默认会在同一目录写出 `species_summary_reviewed.csv`。它保留全部原字段，并新增：

- `top5_only_count`：`top5_count - top1_count`。
- `top5_top1_ratio`：`top5_count / max(top1_count, 1)`。
- `confidence_level`：`high`、`medium` 或 `review`，只表达物种是否具有较稳定的存在信号。
- `special_flag`：`mixed_candidate`、`rare_candidate`、组合值 `mixed_candidate;rare_candidate` 或 `none`，表达是否值得优先人工查看特殊模式。

`confidence_level` 和 `special_flag` 都只是人工审阅辅助规则，不表示 BioCLIP 的真实准确率或概率。两列彼此独立：一个物种可以同时具有 `high` 和 `mixed_candidate`，也可以同时具有 `mixed_candidate;rare_candidate`。

## 故障处理

- 空文件夹会报错，不会启动模型。
- 损坏图片会记录为失败，其他图片继续处理。
- 非图片文件会被忽略。
- 如果一个推理 batch 出错，脚本会拆分 batch 继续处理，直到只把失败图片记录下来。
- `.xls` 旧格式请先在 Excel 中另存为 `.xlsx`，或导出为 UTF-8 CSV。
