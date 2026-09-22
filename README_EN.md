# BirdScan

**Language: English | [中文](README.md)**

A batch bird-identification command-line tool that only reads source photos. It uses BioCLIP 2 to classify within the Latin names in a candidate species table and associates results with the species information in that table. BirdScan is an AI-assisted screening tool, not an authoritative species identification tool.

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
--output-dir bird_report  Report directory
```

The photo directory must exist. The script recursively reads `.jpg`, `.jpeg`, and `.png` files. It does not modify, move, or rename source photos, and does not automatically detect or crop birds or call MegaDetector.

## First run, caching, and performance

BirdScan uses BioCLIP 2 through the `bioclip` import provided by `pybioclip`, loading `hf-hub:imageomics/bioclip-2`. The first run requires an internet connection to download the model from Hugging Face; after download, the underlying dependencies reuse the local cache. Candidate Latin-name text embeddings are cached separately in `~/.cache/birdscan/` and reused when the candidate table is unchanged.

Runtime, memory use, and GPU memory use depend on the device, number of photos, image dimensions, batch size, and number of candidate species. The first model download and first candidate-text encoding are usually slower than later runs. The terminal and `run_summary.txt` report candidate-table loading, model loading, candidate-text encoding or cache loading, image inference, report writing, and total time.

## Output

By default, `bird_report/` is created in the current directory:

- `predictions.csv`: Top-K results for each successful photo, including the relative filename, rank, four species fields, and score.
- `species_summary.csv`: Per-species Top-1/Top-K counts, maximum score, and corresponding best photo.
- `uncertain.csv`: Photos whose Top-1 score is below the threshold.
- `run_summary.txt`: Scan count, success count, failure count, number of distinct Top-1 species, timings, and speed; damaged files and isolated BioCLIP inference errors are also listed.

The reports always output `鸟种编号`, `中文名`, `拉丁学名`, and `英文名称` as separate columns. Missing Chinese or English names remain blank; no other field is used as a fallback.

Scores are probabilities normalized across all Latin names in the candidate table, not a determination that a bird is certainly present in the photo. The first version does not detect or crop birds, so small, obscured, or bird-free photos should receive careful manual review in `uncertain.csv`.

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

## Summary review post-processing

`summarize_report.py` only reads an existing `species_summary.csv`; it does not run BioCLIP again or modify the input file:

```bash
.venv/bin/python summarize_report.py path/to/species_summary.csv
```

By default, it writes `species_summary_reviewed.csv` beside the input. It preserves all original fields and adds:

- `top5_only_count`: `top5_count - top1_count`.
- `top5_top1_ratio`: `top5_count / max(top1_count, 1)`.
- `confidence_level`: `high`, `medium`, or `review`; expresses only whether the species has a relatively stable presence signal.
- `special_flag`: `mixed_candidate`, `rare_candidate`, the combined value `mixed_candidate;rare_candidate`, or `none`; indicates whether a special pattern merits priority manual review.

`confidence_level` and `special_flag` are manual-review aids only; they do not represent BioCLIP's true accuracy or probability. The two columns are independent: a species can have both `high` and `mixed_candidate`, or both `mixed_candidate;rare_candidate`.

## Troubleshooting

- An empty folder raises an error and does not start the model.
- Damaged images are recorded as failures while other images continue.
- Non-image files are ignored.
- If an inference batch fails, the batch is split and retried until only the failed image is recorded.
- For the old `.xls` format, save it as `.xlsx` in Excel or export it as UTF-8 CSV.
