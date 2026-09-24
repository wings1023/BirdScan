import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from megadetector_utils import MODEL_FILENAME, MODEL_URL, official_checkpoint
from scan_birds import load_species, select_crop_result, DEFAULT_SIZE_GATE, parse_args
from summarize_report import process


class LoadSpeciesTests(unittest.TestCase):
    def write_species_csv(self, directory: Path, rows: list[dict[str, str]]) -> Path:
        path = directory / "species.csv"
        fields = ["鸟种编号", "中文名", "拉丁学名", "英文名称"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_optional_names_may_be_blank_and_numbers_need_not_be_unique(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_species_csv(Path(directory), [
                {"鸟种编号": "A-1", "中文名": "", "拉丁学名": "Ardea alba", "英文名称": ""},
                {"鸟种编号": "A-1", "中文名": "夜鹭", "拉丁学名": "Nycticorax nycticorax", "英文名称": ""},
            ])

            species = load_species(path)

        self.assertEqual(species["Ardea alba"]["鸟种编号"], "A-1")
        self.assertEqual(species["Ardea alba"]["中文名"], "")
        self.assertEqual(species["Ardea alba"]["英文名称"], "")

    def test_number_and_latin_name_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_species_csv(Path(directory), [
                {"鸟种编号": "", "中文名": "", "拉丁学名": "Ardea alba", "英文名称": ""},
            ])
            with self.assertRaisesRegex(ValueError, "鸟种编号"):
                load_species(path)

            path = self.write_species_csv(Path(directory), [
                {"鸟种编号": "A-1", "中文名": "", "拉丁学名": "", "英文名称": ""},
            ])
            with self.assertRaisesRegex(ValueError, "拉丁学名"):
                load_species(path)

    def test_latin_name_must_be_unique(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_species_csv(Path(directory), [
                {"鸟种编号": "A-1", "中文名": "", "拉丁学名": "Ardea alba", "英文名称": ""},
                {"鸟种编号": "A-2", "中文名": "", "拉丁学名": "Ardea alba", "英文名称": ""},
            ])
            with self.assertRaisesRegex(ValueError, "Duplicate Latin name"):
                load_species(path)


class SummarizeReportTests(unittest.TestCase):
    def test_blank_species_names_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "species_summary.csv"
            output_path = Path(directory) / "species_summary_reviewed.csv"
            fields = ["鸟种编号", "中文名", "拉丁学名", "英文名称", "top1_count", "topk_count", "max_score"]
            with input_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "鸟种编号": "A-1",
                    "中文名": "",
                    "拉丁学名": "Ardea alba",
                    "英文名称": "",
                    "top1_count": "1",
                    "topk_count": "1",
                    "max_score": "0.8",
                })

            process(input_path, output_path)

            with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(row["鸟种编号"], "A-1")
        self.assertEqual(row["中文名"], "")
        self.assertEqual(row["拉丁学名"], "Ardea alba")
        self.assertEqual(row["英文名称"], "")


class CropSelectionTests(unittest.TestCase):
    def candidate(self, score: float, area: float, name: str) -> dict:
        return {"record": {"area": area, "crop_file": name},
                "prediction": {"score": score, "classification": name}}

    def test_no_detection_falls_back(self) -> None:
        self.assertEqual(select_crop_result([]), (None, False))

    def test_small_bbox_uses_crop_and_large_bbox_uses_baseline(self) -> None:
        self.assertTrue(select_crop_result([self.candidate(.7, DEFAULT_SIZE_GATE, "small")])[1])
        self.assertFalse(select_crop_result([self.candidate(.9, DEFAULT_SIZE_GATE + .001, "large")])[1])

    def test_multiple_crops_choose_highest_top1_score_not_first(self) -> None:
        best, use_crop = select_crop_result([
            self.candidate(.61, .02, "det01"), self.candidate(.82, .03, "det02")
        ])
        self.assertTrue(use_crop)
        self.assertEqual(best["record"]["crop_file"], "det02")

    def parse(self, extra: list[str] | None = None):
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["scan_birds.py", "photos", "--species-file", "species.csv", *(extra or [])]
            return parse_args()
        finally:
            sys.argv = old_argv

    def test_default_crop_and_model(self) -> None:
        args = self.parse()
        self.assertTrue(args.crop)
        self.assertEqual(args.model, "bioclip25")
        self.assertEqual(args.top_k, 3)

    def test_top_k_can_be_overridden(self) -> None:
        self.assertEqual(self.parse(["--top-k", "7"]).top_k, 7)

    def test_no_crop_flag(self) -> None:
        self.assertFalse(self.parse(["--no-crop"]).crop)

    def test_both_models_accept_crop_and_no_crop_modes(self) -> None:
        for model in ("bioclip2", "bioclip25"):
            self.assertTrue(self.parse(["--model", model, "--crop"]).crop)
            self.assertFalse(self.parse(["--model", model, "--no-crop"]).crop)


class MegaDetectorCheckpointTests(unittest.TestCase):
    def test_existing_official_checkpoint_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoints" / MODEL_FILENAME
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"cached")
            download = Mock()
            torch = SimpleNamespace(hub=SimpleNamespace(get_dir=lambda: directory,
                                                        download_url_to_file=download))
            self.assertEqual(official_checkpoint(torch), checkpoint.resolve())
            download.assert_not_called()

    def test_missing_official_checkpoint_is_downloaded_to_torch_hub_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            def download(url: str, destination: str, progress: bool) -> None:
                self.assertEqual(url, MODEL_URL)
                self.assertTrue(progress)
                Path(destination).write_bytes(b"official weights")

            torch = SimpleNamespace(hub=SimpleNamespace(get_dir=lambda: directory,
                                                        download_url_to_file=Mock(side_effect=download)))
            checkpoint = official_checkpoint(torch)
            self.assertEqual(checkpoint, (Path(directory) / "checkpoints" / MODEL_FILENAME).resolve())
            self.assertEqual(checkpoint.read_bytes(), b"official weights")
            self.assertEqual(list(checkpoint.parent.iterdir()), [checkpoint])

    def test_failed_download_does_not_leave_a_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            def download(url: str, destination: str, progress: bool) -> None:
                Path(destination).write_bytes(b"partial")
                raise OSError("download interrupted")

            torch = SimpleNamespace(hub=SimpleNamespace(get_dir=lambda: directory,
                                                        download_url_to_file=download))
            with self.assertRaisesRegex(OSError, "download interrupted"):
                official_checkpoint(torch)
            self.assertEqual(list((Path(directory) / "checkpoints").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
