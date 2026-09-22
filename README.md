# BirdScan

**语言：[English](README_EN.md) | 中文**

一个只读原始照片的批量鸟类识别命令行工具。它使用 BioCLIP 2 在候选物种表的拉丁学名范围内分类，并将结果关联回候选表中的物种信息。BirdScan 用于 AI 辅助初筛，不是权威鸟种鉴定工具。

要求：Python 3.10 或更高版本。目前已在 `pybioclip 2.1.6` 下验证；这不是唯一受支持版本，`requirements.txt` 保持未锁定的常规依赖范围。

## 快速开始

先克隆仓库并进入目录：

```bash
git clone https://github.com/wings1023/BirdScan.git
cd BirdScan
```

仓库根目录附有 `species.xlsx` 候选表。准备一个照片目录后，按设备选择下列其中一条路径。

### Windows + NVIDIA CUDA

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

普通的 `pip install torch` 可能会安装 CPU 版 PyTorch。使用 NVIDIA GPU 时，请先按照 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)针对当前系统、Python 版本和 CUDA 平台生成的命令安装 `torch` / `torchvision`，再安装 BirdScan 的其余运行依赖：

```powershell
python -m pip install pybioclip Pillow
python scan_birds.py ".\photos" --species-file ".\species.xlsx" --device cuda
```

安装后可检查 CUDA 是否可用：

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

不要在 README 中固定某个 CUDA wheel 版本；请以 PyTorch 官方当前安装命令为准。Windows NVIDIA 用户不建议直接执行 `pip install -r requirements.txt` 来决定 PyTorch 的 CUDA 版本，因为裸 `torch` 依赖可能得到 CPU wheel。

### macOS Apple Silicon + MPS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scan_birds.py "./photos" --species-file "./species.xlsx" --device mps
```

### 通用 CPU

macOS 或 Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scan_birds.py "./photos" --species-file "./species.xlsx" --device cpu
```

Windows PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scan_birds.py ".\photos" --species-file ".\species.xlsx" --device cpu
```

`--device` 的默认值是 `mps`，脚本不会自动回退到 CPU。没有可用 MPS 或 CUDA 设备时，必须显式传入 `--device cpu`。

## 候选物种表

候选表支持 UTF-8 CSV 或标准 `.xlsx`，不需要 `openpyxl`。候选表保持以下四列表格格式。四个字段都必须存在；其中“鸟种编号”和“拉丁学名”必须非空，“拉丁学名”必须唯一。“鸟种编号”只是用户自定义的辅助字段，不要求唯一，也不限制具体格式；“中文名”和“英文名称”可以留空：

```text
鸟种编号,中文名,拉丁学名,英文名称
```

拉丁学名是 BioCLIP 的候选分类标签。候选池不包含的物种不会被输出；应按拍摄地区、季节和栖息地缩小候选范围。候选池过宽、包含大量不可能物种时，会增加近缘种和错误候选的误判风险。

## 运行与参数

```bash
python scan_birds.py "./photos" --species-file "./species.xlsx" --device mps
```

可选参数：

```text
--top-k 5             每张照片输出的候选数，默认 5
--batch-size 16       每批推理照片数，默认 16
--threshold 0.5       Top-1 分数低于此值时进入 uncertain.csv
--device mps          默认 mps，也可指定 cuda 或 cpu
--output-dir bird_report  报告目录
```

照片目录必须存在。脚本会递归读取其中的 `.jpg`、`.jpeg`、`.png`，不会修改、移动、重命名原照片，也不会自动裁鸟或调用 MegaDetector。

## 首次运行、缓存与性能

BirdScan 使用 BioCLIP 2（通过 `pybioclip` 导入名 `bioclip`）加载 `hf-hub:imageomics/bioclip-2`。首次运行需要联网从 Hugging Face 下载模型；模型下载后会由底层依赖复用本地缓存。候选拉丁学名的文本向量缓存单独保存在 `~/.cache/birdscan/`，候选表不变时会直接复用。

运行时间、内存和显存占用取决于设备、照片数量、图像尺寸、批量大小和候选物种数量。首次模型下载及首次候选文本编码通常会比后续运行更慢。终端和 `run_summary.txt` 会分别列出候选表读取、模型加载、候选文本编码或缓存加载、图片推理、报告写出和总耗时。

## 输出

默认在当前目录生成 `bird_report/`：

- `predictions.csv`：每张成功照片的 Top-K 结果，包含相对文件名、排名、物种四字段及分数。
- `species_summary.csv`：按 Top-K 出现物种汇总 Top-1/Top-K 次数、最高分及对应最佳照片。
- `uncertain.csv`：Top-1 分数低于阈值的照片。
- `run_summary.txt`：扫描数、成功数、失败数、Top-1 不同物种数、耗时、速度；若有损坏文件或孤立的 BioCLIP 推理错误，也会列在这里。

上述报告始终分别输出“鸟种编号”“中文名”“拉丁学名”“英文名称”四列；缺失的中文名或英文名称保持为空，不会用其他字段替代。

分数是在候选物种表的所有拉丁学名之间归一化的概率，不是“照片中一定有鸟”的判定。第一版不检测或裁切鸟，因此鸟很小、被遮挡或照片没有鸟时，应重点人工复核 `uncertain.csv`。

## 使用边界与人工复核

BirdScan 是鸟类识别与初筛辅助工具，不应作为最终物种鉴定依据。请结合原始照片、地点、时间、行为和可靠的鸟类资料进行判断，必要时咨询有经验的观察者或专家。

以下情况尤其需要人工复核：

- 近缘种；
- 远距离或模糊照片；
- 遮挡或非完整个体；
- 稀有种；
- 一张照片里有多个物种。

模型分数只表示候选物种在当前候选表中的相对排序，不能替代现场证据或正式鉴定。候选表越贴合地点、季节和栖息地，结果越适合人工复核；它不应被视为对未列入候选池物种的否定判断。

## Citation

如果 BirdScan 的结果或工作流对研究、报告或公开项目有帮助，请同时引用 `pybioclip` 软件包和其默认使用的 BioCLIP 2 模型。`pybioclip` 的版本化软件引用请以其仓库中的 `CITATION.cff` 或 Zenodo DOI 为准；BioCLIP 2 可引用：

```bibtex
@inproceedings{gu2025bioclip,
  title={BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning},
  author={Jianyang Gu and Samuel Stevens and Elizabeth G. Campolongo and Matthew J. Thompson and Net Zhang and Jiaman Wu and Andrei Kopanev and Zheda Mai and Alexander E. White and James Balhoff and Wasila Dahdul and Daniel Rubenstein and Hilmar Lapp and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=yPC9zmkQgG}
}
```

请以 [pybioclip 官方仓库的 Citation 说明](https://github.com/Imageomics/pybioclip#citation) 和 [BioCLIP 2 项目](https://github.com/Imageomics/bioclip-2)中的最新信息为准；若改用其他模型，也应按实际使用的模型引用对应论文。

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
