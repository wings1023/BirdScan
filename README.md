# BirdScan

**语言：中文 | [English](README_EN.md)**

BirdScan 是一个面向观鸟人的**批量鸟种初筛工具**，默认使用 BioCLIP 2.5 和 MegaDetector V6 crop 流程，对整批拍鸟照片进行候选鸟种识别和整理。候选物种表可以根据拍摄地点、季节和生境自行缩小，帮助从几百甚至上千张照片中，系统筛出值得进一步人工复核的鸟种、照片和异常结果。BirdScan 只读取、不修改原始照片，是“批量初筛 + 批次汇总 + 人工复核”的辅助工具，不替代人工鉴定。

## 为什么会有 BirdScan

第一次认真拍鸻鹬后，我带回了七八百张照片。很多照片里不止一只鸟，有的主体很小、很远，或已经模糊；而我对不少鸻鹬并不熟悉。逐张使用现有识鸟软件或通用 AI 虽然方便，但面对几百上千张照片仍然很慢，也很难系统地回头检查整批结果。

BirdScan 想解决的是整批照片的筛查和整理：先让 AI 把照片跑完，完成系统性的第一轮筛查，再把人工精力集中到更值得看的候选、照片和异常结果上。重点不是争夺“哪张单图认得最准”，而是帮助观鸟人更有效地开展后续复核。

## 默认工作流

~~~text
原图
→ BioCLIP baseline 分类
→ MegaDetector V6 检测动物并生成多个 crop
→ 使用同一个 BioCLIP 模型对所有 crop 分类
→ 选择 BioCLIP Top-1 score 最高的 crop（best_score）
→ 检查该 crop 对应 bbox 的面积比例
→ bbox area ratio <= 0.08：采用 crop 结果
→ 否则：保留原图 baseline
→ 无检测或该原图没有可分类的 crop：回退 baseline
~~~

Crop 默认开启。MegaDetector detection threshold 默认是 0.15，crop margin 默认是 0.20，size gate 默认是 0.08。使用 --no-crop 可只对原图运行 BioCLIP；--crop 可显式启用默认 crop 流程。阈值和 margin 当前不是普通 CLI 参数。

## 推荐硬件

默认工作流使用 BioCLIP 2.5 和 MegaDetector crop，建议使用 GPU 加速设备。以下是稳妥的推荐配置，不是严格最低要求：

- **macOS：** Apple Silicon，推荐 24GB 或更多统一内存。
- **Windows / Linux：** NVIDIA GPU，推荐 16GB 或更多独立显存。

低于上述配置仍可能运行。可以选择 BioCLIP 2.0、减小 batch size，或减少 BioCLIP 2.5 的 prompt count。CPU 设备受支持，主要用于兼容、调试或小规模测试，不建议用于大批量照片处理。

建议显式指定设备；当前默认值为 `mps`。

- Apple Silicon：--device mps
- Windows / Linux NVIDIA：--device cuda
- CPU：--device cpu

## 安装

需要 Python 3.10 或更高版本。

### macOS / Linux

~~~bash
git clone https://github.com/wings1023/BirdScan.git
cd BirdScan
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Linux NVIDIA 用户请确认安装的是适用于本机 CUDA 平台的 PyTorch。必要时先按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)安装对应的 PyTorch，再安装 BirdScan 依赖。

### Windows PowerShell

~~~powershell
git clone https://github.com/wings1023/BirdScan.git
cd BirdScan
python -m venv .venv
.venv\Scripts\Activate.ps1
~~~

某些系统的 PowerShell execution policy 可能会阻止激活脚本。无需修改策略，可以跳过激活并直接使用虚拟环境中的 Python，例如：

~~~powershell
.\.venv\Scripts\python.exe scan_birds.py "D:\Photos" --species-file species.xlsx --device cuda
~~~

使用 NVIDIA GPU 时，先通过 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)选择适合本机的 CUDA 安装命令，再安装项目依赖：

~~~powershell
python -m pip install -r requirements.txt
~~~

如果没有激活虚拟环境，可在安装依赖时将 `python` 替换为 `.\.venv\Scripts\python.exe`。

## 运行

主流程是：安装 BirdScan → 准备候选鸟种表（可先用仓库提供的 species.xlsx）→ 运行 scan_birds.py → 查看输出报告。一次扫描会直接生成主要结果；通常不需要再运行其他脚本。BioCLIP 2.5 和 crop 均为默认设置，无需额外指定：

照片目录支持 JPG、JPEG 和 PNG；Sony ARW 等 RAW 文件当前不支持。

### macOS Apple Silicon

~~~bash
python scan_birds.py "/path/to/photos" --species-file species.xlsx --device mps
~~~

### Windows + NVIDIA

~~~bash
python scan_birds.py "D:\Photos" --species-file species.xlsx --device cuda
~~~

### 常用选项

- --model bioclip25：默认模型，也是当前推荐模型。
- --model bioclip2：BioCLIP 2.0，较低资源配置下可选。
- --crop：显式启用默认 crop 流程。
- --no-crop：关闭 crop，直接使用原图 baseline。
- --prompt-count N：BioCLIP 2.5 每个候选物种使用的模板数，范围为 1–80，默认 80。
- --batch-size N：每批分类的图片数，默认 16；资源较少时可调低。
- --top-k N：每张照片保存的候选数，默认 Top-3；可指定其他大于等于 1 的数量。
- --threshold N：Top-1 score 低于该值的照片写入 uncertain.csv，默认 0.5。
- --output-dir DIR：报告目录。默认 BioCLIP 2.5 写入 reports/bird_report_bioclip25/；BioCLIP 2.0 写入 reports/bird_report/。

参数名和其他选项可通过以下命令查看：

~~~bash
python scan_birds.py --help
~~~

## 候选鸟种表

你可以使用自定义候选表；仓库中的 species.xlsx 是一个可直接参考的示例。候选表支持 UTF-8 CSV 和标准 XLSX，表格必须包含以下四列：

~~~text
鸟种编号,中文名,拉丁学名,英文名称
~~~

鸟种编号和拉丁学名必须非空，拉丁学名必须唯一；鸟种编号不要求唯一，中文名和英文名称可以留空。BioCLIP 使用拉丁学名作为候选标签。候选池越贴近拍摄地区和季节，通常越适合初筛；不在候选表中的物种不会出现在结果中。

首次运行需要联网。本机没有对应缓存时，程序会下载 BioCLIP 模型；默认 crop 流程首次使用时还会下载 MegaDetector 权重，并为候选物种构建文本 embedding cache。第一次运行通常明显慢于后续运行；之后会复用已有模型、权重和 embedding 缓存。候选 embedding 缓存在用户目录下的 .cache/birdscan/，相同模型、候选物种及 prompt 配置可复用缓存。

## 输出文件

默认报告写入 reports/。该目录已加入 Git 忽略规则，属于本地运行输出。

第一次使用建议先查看 predictions.csv，再查看 run_summary.txt，确认扫描数量、失败数量和耗时。

- **predictions.csv：** 每张成功照片的最终 Top-K 结果，含 file_name、rank、鸟种编号、中文名、拉丁学名、英文名称、score，以及以下 crop 诊断字段：
  - final_source、crop_used
  - selected_crop_file、detection_confidence、bbox_area_ratio
  - baseline_top1_species、baseline_top1_score
  - crop_top1_species、crop_top1_score
- **species_summary.csv：** 按物种汇总 top1_count、topk_count、max_score 和 best_image。
- **uncertain.csv：** Top-1 score 低于 threshold 的照片。
- **run_summary.txt：** 扫描与失败数量、各阶段耗时和整体速度。

predictions.csv 的 file_name 始终指向原始照片。启用 crop 时，final_source 表示最终采用 baseline 还是 crop；crop_used 表示是否采用 crop。selected_crop_file、detection_confidence 和 bbox_area_ratio 记录最高分 crop 的检测信息。使用 --no-crop 时，final_source 为 baseline，crop 专属字段留空。

## 可选：离线审阅与后处理

如需对已有扫描结果进行进一步物种审阅或筛选，可以选择运行 summarize_report.py。它直接读取已有的 species_summary.csv，不会重新运行模型或重新扫描照片；适合对历史结果重新计算审阅字段：

~~~bash
python summarize_report.py reports/bird_report_bioclip25/species_summary.csv
~~~

这是可选后处理，不属于 scan_birds.py 主流程的必需步骤。脚本不会覆盖输入文件；默认在同一目录生成 species_summary_reviewed.csv，并增加 topk_only_count、topk_top1_ratio、confidence_level 和 special_flag，供人工审阅批次结果使用。

## 项目文件

- scan_birds.py：主入口，扫描照片、运行原图与 crop 分类并写出报告。
- megadetector_utils.py：MegaDetector V6 加载及 detection / crop 支持。
- summarize_report.py：读取物种汇总并生成审阅版报告。
- test_scan_birds.py：自动化测试。

~~~text
experiments/  研究、诊断和评测工具，包括 crop、fusion、size gate 等脚本
benchmarks/   人工标注 ground truth，用于实验策略的回归验证
reports/      本地运行输出，Git ignored
~~~

## 人工复核与问题反馈

模型输出用于批量初筛，不是最终鉴定。近缘种、远距离或模糊照片、遮挡个体、稀有种和一张照片中出现多个物种的情况尤其需要查看原图。模型 score 表示候选表内的排序结果，不是鸟种确定存在的证明。

如遇到问题或有功能建议，欢迎通过 [GitHub Issues](https://github.com/wings1023/BirdScan/issues) 反馈。
