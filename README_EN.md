# BirdScan

**Language: English | [中文](README.md)**

A batch species screening tool for birders.
BirdScan uses BioCLIP 2, a powerful biological AI vision model, to identify bird-species candidates across a whole batch of photos using a candidate species table you define. You can organize that table around location, season, and even habitat, helping narrow the candidate range substantially and making the results more useful in practice.
It is designed to systematically screen hundreds or thousands of photos and organize results for further manual review; it only reads and never modifies the original photos.

## Why BirdScan exists

It began with a very specific bird-photography problem. After taking shorebird photography seriously for the first time, I came home with seven or eight hundred photos. Many frames held more than one bird; some subjects were tiny, distant, or blurred. I was not yet familiar with many of the sandpipers and plovers. Sending photos one at a time to an existing bird-ID app or general-purpose AI is convenient, but it is still slow with hundreds or thousands of images—and makes it hard to inspect the whole batch systematically afterwards.

That is why these two small scripts exist. The point is not to compete over which model recognizes a single image most accurately. It is to let AI complete a first, batch-wide and systematic pass, then focus human attention on the candidates, photos, and odd results that deserve it most. BirdScan is a helper for batch screening, batch-level summarization, and manual review; it does not replace human identification.

## Two-step workflow

```text
Photos
  ↓
scan_birds.py
  ↓
predictions.csv
species_summary.csv
  ↓
summarize_report.py
  ↓
species_summary_reviewed.csv
  ↓
Manual review
```

- `scan_birds.py`: Uses BioCLIP 2 to run a batch Top-K first pass against your custom candidate species table. It writes per-photo results to `predictions.csv` and a batch summary to `species_summary.csv`.
- `summarize_report.py`: The second step in the workflow. It further organizes and flags the existing scan results without running BioCLIP 2 again, helping reveal stable candidates, questionable candidates, and results worth returning to the original photos to review first.

Requirement: Python 3.10 or newer. The project is currently verified with `pybioclip 2.1.6`; this is not the only supported version, and `requirements.txt` keeps the regular dependencies unpinned.

## Quick start

Clone the repository and enter the project directory:

```bash
git clone https://github.com/wings1023/BirdScan.git
cd BirdScan
```

The repository root includes the `species.xlsx` candidate table. Prepare a photo directory, then choose one of the following paths for your device.

### Windows + NVIDIA CUDA

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

A regular `pip install torch` may install a CPU-only PyTorch build. For an NVIDIA GPU, first use the [official PyTorch installation page](https://pytorch.org/get-started/locally/) to generate the command for your system, Python version, and CUDA platform to install `torch` / `torchvision`, then install the remaining BirdScan runtime dependencies:

```powershell
python -m pip install pybioclip Pillow
python scan_birds.py ".\photos" --species-file ".\species.xlsx" --device cuda
```

You can check CUDA availability after installation:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Do not pin a particular CUDA wheel version in this README; use the current command from the official PyTorch page. Windows NVIDIA users should not directly run `pip install -r requirements.txt` to choose the PyTorch CUDA version, because the unqualified `torch` dependency may install a CPU wheel.

### macOS Apple Silicon + MPS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scan_birds.py "./photos" --species-file "./species.xlsx" --device mps
```

### General CPU

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scan_birds.py "./photos" --species-file "./species.xlsx" --device cpu
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scan_birds.py ".\photos" --species-file ".\species.xlsx" --device cpu
```

The default value of `--device` is `mps`, and the script does not automatically fall back to CPU. If MPS or CUDA is unavailable, you must explicitly pass `--device cpu`.

## Candidate species table

The candidate table supports UTF-8 CSV or standard `.xlsx` files and does not require `openpyxl`. It keeps the following four-column format. All four fields must exist; `鸟种编号` and `拉丁学名` must be non-empty, and `拉丁学名` must be unique. `鸟种编号` is a user-defined auxiliary field: it does not need to be unique and has no format restrictions. `中文名` and `英文名称` may be blank:

```text
鸟种编号,中文名,拉丁学名,英文名称
```

The Latin name is the BioCLIP candidate classification label. A species not included in the candidate pool cannot be output. Narrow the candidate pool by region, season, and habitat. An overly broad pool containing many unlikely species increases the risk of confusion between related species and incorrect candidates.

## Running and parameters

```bash
python scan_birds.py "./photos" --species-file "./species.xlsx" --device mps
```

Optional parameters:

```text
--top-k 5             Number of candidates saved per photo; default: 5
--batch-size 16       Images per inference batch; default: 16
--threshold 0.5       Top-1 scores below this value go to uncertain.csv
--device mps          Default mps; cuda or cpu can also be specified
--output-dir <dir>      Report directory (default: reports/bird_report/; BioCLIP 2.5 default: reports/bird_report_bioclip25/)
```

The photo directory must exist. The script recursively reads `.jpg`, `.jpeg`, and `.png` files. It does not modify, move, or rename source photos, and does not automatically detect or crop birds or call MegaDetector.

## First run, caching, and performance

BirdScan uses BioCLIP 2 through the `bioclip` import provided by `pybioclip`, loading `hf-hub:imageomics/bioclip-2`. The first run requires an internet connection to download the model from Hugging Face; after download, the underlying dependencies reuse the local cache. Candidate Latin-name text embeddings are cached separately in `~/.cache/birdscan/` and reused when the candidate table is unchanged.

Runtime, memory use, and GPU memory use depend on the device, number of photos, image dimensions, batch size, and number of candidate species. The first model download and first candidate-text encoding are usually slower than later runs. The terminal and `run_summary.txt` report candidate-table loading, model loading, candidate-text encoding or cache loading, image inference, report writing, and total time.

## Scan output

By default, BioCLIP 2 reports are written to `reports/bird_report/`; BioCLIP 2.5 uses `reports/bird_report_bioclip25/`:

- `predictions.csv`: Top-K results for each successful photo, including the relative filename, rank, four species fields, and score.
- `species_summary.csv`: Per-species Top-1/Top-K counts (the `top1_count` and `topk_count` fields), maximum score, and corresponding best photo.
- `uncertain.csv`: Photos whose Top-1 score is below the threshold.
- `run_summary.txt`: Scan count, success count, failure count, number of distinct Top-1 species, timings, and speed; damaged files and isolated BioCLIP inference errors are also listed.

The three CSV reports—`predictions.csv`, `species_summary.csv`, and `uncertain.csv`—always output `鸟种编号`, `中文名`, `拉丁学名`, and `英文名称` as separate columns. Missing Chinese or English names remain blank; no other field is used as a fallback. `run_summary.txt` is a plain-text runtime summary and is outside this field description.

Scores are probabilities normalized across all Latin names in the candidate table, not a determination that a bird is certainly present in the photo. The first version does not detect or crop birds, so small, obscured, or bird-free photos should receive careful manual review in `uncertain.csv`.

## Summary review: workflow step two

Once `scan_birds.py` has written `species_summary.csv`, run `summarize_report.py` to organize the results from that batch. It does not run BioCLIP 2 again and does not modify its input file:

```bash
python summarize_report.py path/to/species_summary.csv
```

By default, it writes `species_summary_reviewed.csv` beside the input. It preserves all original fields and adds or organizes:

- `topk_only_count`: The number of times a species appeared only in Top-K, rather than as Top-1.
- `topk_top1_ratio`: The ratio between Top-K appearances and Top-1 appearances.
- `confidence_level`: Whether the candidate shows a relatively stable presence signal within this batch.
- `special_flag`: A marker for patterns that deserve priority manual inspection.

These fields are not new model judgments, nor do they express real accuracy or probability. They support a batch-level review: finding repeatedly stable candidates; candidates that often enter Top-K but rarely reach Top-1; unusual or questionable results; and the species or photos most worth checking against the originals first.

## Usage boundaries and manual review

BirdScan is an AI-assisted bird-identification and screening tool and should not be used as a final species determination. Combine the result with the original photo, location, date, behavior, and reliable bird references; consult an experienced observer or expert when necessary.

Manual review is especially important for:

- closely related species;
- distant or blurry photos;
- obscured or incomplete individuals;
- rare species;
- photos containing multiple species.

Model scores represent only the relative ranking within the current candidate table and cannot replace field evidence or formal identification. The closer the candidate table matches the location, season, and habitat, the more useful the result is for manual review; it should not be interpreted as evidence against species omitted from the candidate pool.

## Citation

If BirdScan results or workflows contribute to research, reports, or public projects, cite both the `pybioclip` package and its default BioCLIP 2 model. For the versioned software citation of `pybioclip`, refer to its `CITATION.cff` or Zenodo DOI; BioCLIP 2 can be cited as follows:

```bibtex
@inproceedings{gu2025bioclip,
  title={BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning},
  author={Jianyang Gu and Samuel Stevens and Elizabeth G. Campolongo and Matthew J. Thompson and Net Zhang and Jiaman Wu and Andrei Kopanev and Zheda Mai and Alexander E. White and James Balhoff and Wasila Dahdul and Daniel Rubenstein and Hilmar Lapp and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=yPC9zmkQgG}
}
```

Use the latest information in the [official pybioclip repository's citation instructions](https://github.com/Imageomics/pybioclip#citation) and the [BioCLIP 2 project](https://github.com/Imageomics/bioclip-2). If you use another model, cite the corresponding paper for the model actually used.

## Troubleshooting

- An empty folder raises an error and does not start the model.
- Damaged images are recorded as failures while other images continue.
- Non-image files are ignored.
- If an inference batch fails, the batch is split and retried until only the failed image is recorded.
- For the old `.xls` format, save it as `.xlsx` in Excel or export it as UTF-8 CSV.
