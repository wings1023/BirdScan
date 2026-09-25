# BirdScan

**Language: English | [中文](README.md)**

BirdScan is a **batch species-screening tool for birders**. Its default workflow uses BioCLIP 2.5 with MegaDetector V6 crops to identify and organize candidate species across a full batch of bird photos. You can tailor the candidate list to the location, season, and habitat to systematically surface birds, photos, and unusual results worth reviewing among hundreds or even thousands of images. BirdScan reads original photos without modifying them. It supports batch screening, batch summaries, and manual review; it does not replace human identification.

## Why BirdScan exists

After my first serious shorebird photography trip, I came home with seven or eight hundred photos. Many showed more than one bird; in others, the subject was tiny, distant, or already blurred. I was also unfamiliar with many of the shorebirds. Identifying photos one by one with existing bird-ID apps or general-purpose AI can be convenient, but it is still slow across hundreds or thousands of images and makes it hard to review a whole batch systematically.

BirdScan is meant to help with that batch-level work. It runs an initial, systematic screening pass across the photos, then helps focus human attention on the candidates, images, and unusual results most worth checking. The goal is not to win a single-image accuracy contest, but to make follow-up review of a large photo batch more manageable.

## Default workflow

~~~text
Original image
→ BioCLIP baseline classification
→ MegaDetector V6 detects animals and generates crops
→ classify all crops with the same BioCLIP model and keep only each detection's Top-1
→ only original det01 can determine primary: use its crop Top-1 directly when classification succeeds and bbox area ratio <= 0.08, without comparing scores with baseline
→ keep the original-image baseline when det01 has area ratio > 0.08, is filtered, or has no crop classification result; det02+ never replace primary
→ list det02+ Top-1 as additional species only when score >= 0.90 and the species is distinct and not repeated
→ additional species never replace the primary
~~~

Crop is enabled by default. MegaDetector detection threshold=0.15 is an internal setting; crop margin defaults to 0.30 and the size gate to 0.08. Detections whose original bbox area is below 0.0003 of the source image are skipped by default. This check uses the detector bbox before margin expansion. Use --no-crop to classify original images only; --crop explicitly enables the default crop workflow. The MegaDetector detection threshold and crop margin are not regular CLI options.

See the [selection pipeline](docs/selection_pipeline.md) for the complete result selection rules, parameter roles, and report field meanings.

## Recommended hardware

The default workflow uses BioCLIP 2.5 with MegaDetector crop, so a GPU-accelerated device is recommended. These are recommended configurations, not strict minimum requirements:

- **macOS:** Apple Silicon, with 24GB or more unified memory recommended.
- **Windows / Linux:** NVIDIA GPU, with 16GB or more dedicated VRAM recommended.

Lower-spec systems may still run. You can choose BioCLIP 2.0, reduce batch size, or use fewer BioCLIP 2.5 prompts. CPU is supported mainly for compatibility, debugging, or small-scale testing, and is not recommended for large photo batches.

Explicitly specifying the device is recommended. The current default is `mps`.

- Apple Silicon: --device mps
- Windows / Linux NVIDIA: --device cuda
- CPU: --device cpu

## Installation

Python 3.10 or newer is required.

### macOS / Linux

~~~bash
git clone https://github.com/wings1023/BirdScan.git
cd BirdScan
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Linux NVIDIA users should confirm that PyTorch is installed for the CUDA platform on their system. If needed, install the appropriate PyTorch build from the [official PyTorch installation page](https://pytorch.org/get-started/locally/) before installing BirdScan dependencies.

### Windows PowerShell

~~~powershell
git clone https://github.com/wings1023/BirdScan.git
cd BirdScan
python -m venv .venv
.venv\Scripts\Activate.ps1
~~~

On some systems, PowerShell's execution policy may block script activation. You can skip activation without changing the policy and call the virtual environment's Python directly, for example:

~~~powershell
.\.venv\Scripts\python.exe scan_birds.py "D:\Photos" --species-file species.xlsx --device cuda
~~~

For an NVIDIA GPU, first use the [official PyTorch installation page](https://pytorch.org/get-started/locally/) to choose the CUDA installation command for your system, then install the project dependencies:

~~~powershell
python -m pip install -r requirements.txt
~~~

If you did not activate the environment, use `.\.venv\Scripts\python.exe` in place of `python` when installing dependencies.

## Run

The main workflow is: install BirdScan → prepare a candidate species table (you can start with the repository's species.xlsx) → run scan_birds.py → review the output. A single scan writes the core reports; you normally do not need to run another script. BioCLIP 2.5 and crop are both enabled by default, so you do not need to specify them:

Photo folders support JPG, JPEG, and PNG. Sony ARW and other RAW files are not currently supported.

### macOS Apple Silicon

~~~bash
python scan_birds.py "/path/to/photos" --species-file species.xlsx --device mps
~~~

### Windows + NVIDIA

~~~bash
python scan_birds.py "D:\Photos" --species-file species.xlsx --device cuda
~~~

### Common options

- --model bioclip25: the default and currently recommended model.
- --model bioclip2: BioCLIP 2.0, an option for systems with fewer resources.
- --crop: explicitly enable the default crop workflow.
- --no-crop: disable crop and classify original images with the baseline.
- --prompt-count N: templates per candidate species for BioCLIP 2.5; valid range 1–80, default 80.
- --batch-size N: images per classification batch, default 16; reduce it on lower-resource systems.
- --top-k N: candidates requested from original-image BioCLIP, default 1 for the formal workflow; explicit values above 1 affect internal inference only, while the user report still has one primary row per photo. Experiment and diagnostic scripts may use Top-K independently.
- --threshold N: used only to decide uncertain.csv membership; a final primary Top-1 score below this value is listed there. Default 0.5. It does not change the MegaDetector detection threshold.
- --min-bbox-area-ratio N: skip detections whose original bbox area divided by original image area is below N; default 0.0003. This applies only when crop is enabled, and detections exactly at the threshold are retained.
- --output-dir DIR: report directory. By default BioCLIP 2.5 writes to reports/bird_report_bioclip25/ and BioCLIP 2.0 writes to reports/bird_report/.

See all options with:

~~~bash
python scan_birds.py --help
~~~

## Candidate species table

You can use a custom candidate table; the repository's species.xlsx is a ready-to-use example. Candidate tables support UTF-8 CSV and standard XLSX files and must contain these four columns:

~~~text
鸟种编号,中文名,拉丁学名,英文名称
~~~

Bird ID and Latin name must be non-empty, and each Latin name must be unique. Bird IDs do not need to be unique; Chinese and English names may be blank. BioCLIP uses the Latin name as the candidate label. Report display names use the Chinese name first, then the English name, and fall back to the Latin name. A pool tailored to the location and season is usually more useful for screening; species outside the table will not appear in the results.

The first run requires an internet connection. If the corresponding cache is not already present, BirdScan downloads the BioCLIP model; the default crop workflow also downloads MegaDetector weights on first use and builds text-embedding cache for the candidate species. The first run is usually noticeably slower than later runs. Later runs reuse available models, weights, and embedding cache. Candidate embeddings are cached under .cache/birdscan/ in your user directory and can be reused for the same model, candidate list, and prompt configuration.

## Output files

Reports are written to reports/ by default. This directory is Git ignored and contains local run output.

For your first run, start with predictions.csv, then check run_summary.txt for the number of scanned photos, failures, and elapsed time.

- **predictions.csv:** one primary Top-1 row per successfully classified photo, with no rank column. It retains file_name, Bird ID, Chinese name, Latin name, English name, and score, and adds primary_species, primary_score, additional_species, additional_scores, and additional_count:
  - additional_species and additional_scores are pipe-separated in matching order and come only from other detections' Top-1.
  - Additional species must score at least 0.90; duplicates and the primary species are excluded, and additional species never change the primary.
  - baseline_top1_species, crop_top1_species, primary_species, and additional_species use Chinese display names when available, then English names, then Latin names.
  - The file also includes these crop diagnostics:
    - final_source, crop_used
    - selected_crop_file, detection_confidence, bbox_area_ratio
    - baseline_top1_species, baseline_top1_score
    - crop_top1_species, crop_top1_score
- **species_summary.csv:** summarizes only successfully classified photos present in predictions.csv, with per-species primary_count, additional_count, max_score, and best_image. max_score is the highest score for that species across primary and additional appearances; best_image is the photo where that score occurred. It no longer uses the legacy Top-K count field.
- **uncertain.csv:** photos whose Top-1 score is below threshold.
- **run_summary.txt:** scan and failure counts, the minimum bbox area ratio and number of skipped detections, stage timings, and overall speed.

The file_name in predictions.csv always points to the original photo. With crop enabled, final_source says whether the baseline or original det01 crop was selected for primary; crop_used indicates whether crop was used. When crop is selected, selected_crop_file, detection_confidence, and bbox_area_ratio describe det01. With --no-crop, final_source is baseline and crop-specific fields are blank.

## Project files

- scan_birds.py: main entry point for scanning photos, classifying originals and crops, and writing reports.
- megadetector_utils.py: MegaDetector V6 loading and detection / crop support.
- test_scan_birds.py: automated tests.

~~~text
experiments/  research, diagnostics, and evaluation tools, including crop, fusion, and size-gate scripts
benchmarks/   manually labeled ground truth for regression checks of experimental strategies
reports/      local run output; Git ignored
~~~

## Manual review and feedback

Model output is for batch screening, not final identification. Closely related species, distant or blurry photos, obscured birds, rare species, and photos with multiple species especially need review against the original image. A model score ranks candidates in the supplied table; it does not prove that a species is present.

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/wings1023/BirdScan/issues).
