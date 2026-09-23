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
→ classify all crops with the same selected BioCLIP model
→ select the crop with the highest BioCLIP Top-1 score (best_score)
→ check the selected bbox area ratio
→ bbox area ratio <= 0.08: use the crop result
→ otherwise: keep the original-image baseline
→ no detection or no classifiable crop for an image: fall back to baseline
~~~

Crop is enabled by default. The MegaDetector detection threshold defaults to 0.15, crop margin to 0.20, and size gate to 0.08. Use --no-crop to classify original images only; --crop explicitly enables the default crop workflow. The threshold and margin are not regular CLI options.

## Recommended hardware

The default workflow uses BioCLIP 2.5 with MegaDetector crop, so a GPU-accelerated device is recommended. These are recommended configurations, not strict minimum requirements:

- **macOS:** Apple Silicon, with 24GB or more unified memory recommended.
- **Windows / Linux:** NVIDIA GPU, with 16GB or more dedicated VRAM recommended.

Lower-spec systems may still run. You can choose BioCLIP 2.0, reduce batch size, or use fewer BioCLIP 2.5 prompts. CPU is supported mainly for compatibility, debugging, or small-scale testing, and is not recommended for large photo batches.

Choose the device explicitly; the program does not select one automatically:

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

For an NVIDIA GPU, first use the [official PyTorch installation page](https://pytorch.org/get-started/locally/) to choose the CUDA installation command for your system, then install the project dependencies:

~~~powershell
python -m pip install -r requirements.txt
~~~

## Run

The main workflow is: install BirdScan → prepare a candidate species table (you can start with the repository's species.xlsx) → run scan_birds.py → review the output. A single scan writes the core reports; you normally do not need to run another script. BioCLIP 2.5 and crop are both enabled by default, so you do not need to specify them:

### macOS Apple Silicon

~~~bash
python scan_birds.py "/path/to/photos" --species-file species.xlsx --device mps
~~~

### Windows / Linux NVIDIA

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
- --top-k N: candidates saved per photo, default Top-3; any value of 1 or greater can be specified.
- --threshold N: photos with a Top-1 score below this value go to uncertain.csv; default 0.5.
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

Bird ID and Latin name must be non-empty, and each Latin name must be unique. Bird IDs do not need to be unique; Chinese and English names may be blank. BioCLIP uses the Latin name as the candidate label. A pool tailored to the location and season is usually more useful for screening; species outside the table will not appear in the results.

The first run may download the BioCLIP model and MegaDetector weights and build text embeddings for the candidate species. Candidate embeddings are cached under .cache/birdscan/ in your user directory and can be reused for the same model, candidate list, and prompt configuration.

## Output files

Reports are written to reports/ by default. This directory is Git ignored and contains local run output.

- **predictions.csv:** final Top-K results for each successfully classified photo, with file_name, rank, Bird ID, Chinese name, Latin name, English name, score, and these crop diagnostics:
  - final_source, crop_used
  - selected_crop_file, detection_confidence, bbox_area_ratio
  - baseline_top1_species, baseline_top1_score
  - crop_top1_species, crop_top1_score
- **species_summary.csv:** per-species top1_count, topk_count, max_score, and best_image.
- **uncertain.csv:** photos whose Top-1 score is below threshold.
- **run_summary.txt:** scan and failure counts, stage timings, and overall speed.

The file_name in predictions.csv always points to the original photo. With crop enabled, final_source says whether the baseline or a crop result was selected; crop_used indicates whether crop was used. selected_crop_file, detection_confidence, and bbox_area_ratio describe the highest-scoring crop candidate. With --no-crop, final_source is baseline and crop-specific fields are blank.

## Optional: Offline review and post-processing

If you want to review or filter species results further, you can optionally run summarize_report.py on an existing species_summary.csv. It reads that file directly and does not rerun the model or rescan photos, so it can also recalculate review fields for historical results:

~~~bash
python summarize_report.py reports/bird_report_bioclip25/species_summary.csv
~~~

This is optional post-processing, not a required step in the scan_birds.py workflow. The script does not overwrite the input; by default, it writes species_summary_reviewed.csv beside it and adds topk_only_count, topk_top1_ratio, confidence_level, and special_flag to support manual batch review.

## Project files

- scan_birds.py: main entry point for scanning photos, classifying originals and crops, and writing reports.
- megadetector_utils.py: MegaDetector V6 loading and detection / crop support.
- summarize_report.py: reads the species summary and creates a reviewed report.
- test_scan_birds.py: automated tests.

~~~text
experiments/  research, diagnostics, and evaluation tools, including crop, fusion, and size-gate scripts
benchmarks/   manually labeled ground truth for regression checks of experimental strategies
reports/      local run output; Git ignored
~~~

## Manual review and feedback

Model output is for batch screening, not final identification. Closely related species, distant or blurry photos, obscured birds, rare species, and photos with multiple species especially need review against the original image. A model score ranks candidates in the supplied table; it does not prove that a species is present.

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/wings1023/BirdScan/issues).
