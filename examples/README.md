# BirdScan Sample12 示例数据集

## 中文

`sample12_jpeg80/` 包含 12 张真实观鸟照片，选自更大的 test31 测试集。

这些照片并非随机抽样，而是人工挑选，覆盖常见鸟种、小型或远距离目标、多鸟种画面、较难鉴定案例，以及至少一个已知识别失败案例。

图片保留原始像素尺寸，仅重新编码为 JPEG quality 80，以降低下载和仓库体积。

对应人工真值：`ground_truth_sample12.csv`。

这个样本集主要用于：

- 演示 BirdScan 的实际运行流程；
- 复现 primary species 输出；
- 复现 additional species 输出；
- 帮助新用户快速确认安装和推理流程是否正常。

它不是统计意义上的正式 benchmark，也不应使用这 12 张照片估计 BirdScan 的整体准确率。

### 示例命令

```bash
python scan_birds.py "./examples/sample12_jpeg80" \
  --species-file "./species.xlsx" \
  --output-dir "./reports/sample12_example"
```

## English

`sample12_jpeg80/` contains 12 real bird photographs selected from a larger test31 field-test set.

The images were manually selected rather than randomly sampled. They include common species, small or distant birds, multi-species scenes, difficult identification cases, and at least one known failure case.

The original pixel dimensions are preserved. The images were only re-encoded at JPEG quality 80 to reduce repository and download size.

Manual ground truth: `ground_truth_sample12.csv`.

This sample set is intended for:

- demonstrating BirdScan's real workflow;
- reproducing primary species output;
- reproducing additional species output;
- helping new users quickly verify that installation and inference work correctly.

It is not a statistically representative benchmark and should not be used to estimate BirdScan's overall accuracy.

### Example command

```bash
python scan_birds.py "./examples/sample12_jpeg80" \
  --species-file "./species.xlsx" \
  --output-dir "./reports/sample12_example"
```
