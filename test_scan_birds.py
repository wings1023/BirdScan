import csv
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from pathlib import Path, PureWindowsPath
from unittest.mock import Mock, patch

from experiments import evaluate_crop_pipeline as matching
from experiments import inspect_neighbor_consistency
from megadetector_utils import MODEL_FILENAME, MODEL_URL, official_checkpoint
from scan_birds import (
    ADDITIONAL_SCORE_THRESHOLD,
    DEFAULT_CROP_MARGIN,
    DEFAULT_DETECTION_THRESHOLD,
    DEFAULT_SIZE_GATE,
    CROP_TOP_K,
    load_species,
    main,
    parse_args,
    relative_file_key,
    species_display_name,
    select_crop_result,
    select_primary_and_additional,
    write_reports,
)


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


class CropSelectionTests(unittest.TestCase):
    def candidate(self, score: float, area: float, name: str) -> dict:
        detection_index = int(name.rsplit("det", 1)[-1]) if "det" in name else 1
        return {"record": {"area": area, "crop_file": name, "detection_index": detection_index},
                "prediction": {"score": score, "classification": name}}

    def test_no_detection_falls_back(self) -> None:
        self.assertEqual(select_crop_result([]), (None, False))

    def test_small_bbox_uses_crop_and_large_bbox_uses_baseline(self) -> None:
        self.assertTrue(select_crop_result([self.candidate(.7, DEFAULT_SIZE_GATE, "small")])[1])
        self.assertFalse(select_crop_result([self.candidate(.9, DEFAULT_SIZE_GATE + .001, "large")])[1])

    def test_first_detection_is_primary_even_when_later_detection_scores_higher(self) -> None:
        primary, use_crop = select_crop_result([
            self.candidate(.61, .02, "det01"), self.candidate(.99, .03, "det02")
        ])
        self.assertTrue(use_crop)
        self.assertEqual(primary["record"]["crop_file"], "det01")

    def test_additional_species_require_threshold_and_are_deduplicated(self) -> None:
        candidates = [
            self.candidate(.75, .02, "primary-det1"),
            self.candidate(.8999, .02, "low-det2"),
            self.candidate(.9000, .02, "extra-det3"),
            self.candidate(.94, .02, "other-det4"),
            self.candidate(.97, .02, "extra-det5"),
            self.candidate(.97, .02, "primary-det6"),
        ]
        candidates[0]["prediction"]["classification"] = "primary"
        candidates[1]["prediction"]["classification"] = "low"
        candidates[2]["prediction"]["classification"] = "extra"
        candidates[3]["prediction"]["classification"] = "other"
        candidates[4]["prediction"]["classification"] = "extra"
        candidates[5]["prediction"]["classification"] = "primary"

        primary, use_crop, chosen, additional = select_primary_and_additional(candidates, None)

        self.assertEqual(ADDITIONAL_SCORE_THRESHOLD, .90)
        self.assertTrue(use_crop)
        self.assertEqual(primary["classification"], "primary")
        self.assertEqual(chosen["record"]["detection_index"], 1)
        self.assertEqual(
            [(item["prediction"]["classification"], item["prediction"]["score"]) for item in additional],
            [("extra", .97), ("other", .94)],
        )

    def test_missing_or_gated_primary_falls_back_to_baseline(self) -> None:
        baseline = {"classification": "baseline", "score": .8}
        primary, use_crop, selected, additional = select_primary_and_additional([], baseline)
        self.assertIs(primary, baseline)
        self.assertFalse(use_crop)
        self.assertIsNone(selected)
        self.assertEqual(additional, [])

        candidates = [
            self.candidate(.99, DEFAULT_SIZE_GATE + .001, "primary-det1"),
            self.candidate(.91, .02, "baseline-det2"),
            self.candidate(.92, .02, "additional-det3"),
        ]
        candidates[0]["prediction"]["classification"] = "crop-primary"
        candidates[1]["prediction"]["classification"] = "baseline"
        candidates[2]["prediction"]["classification"] = "additional"
        primary, use_crop, selected, additional = select_primary_and_additional(candidates, baseline)
        self.assertIs(primary, baseline)
        self.assertFalse(use_crop)
        self.assertIsNotNone(selected)
        self.assertEqual([item["prediction"]["classification"] for item in additional], ["additional"])

    def test_default_margin_and_existing_crop_thresholds(self) -> None:
        self.assertEqual(DEFAULT_CROP_MARGIN, .30)
        self.assertEqual(DEFAULT_SIZE_GATE, .08)
        self.assertEqual(DEFAULT_DETECTION_THRESHOLD, .15)
        self.assertEqual(CROP_TOP_K, 1)

    def test_species_display_name_uses_chinese_then_english_then_latin(self) -> None:
        species = {
            "latin_cn": {"中文名": "中文名", "英文名称": "English name"},
            "latin_en": {"中文名": "", "英文名称": "English name"},
            "latin_only": {"中文名": "", "英文名称": ""},
        }
        self.assertEqual(species_display_name("latin_cn", species), "中文名")
        self.assertEqual(species_display_name("latin_en", species), "English name")
        self.assertEqual(species_display_name("latin_only", species), "latin_only")

    def test_report_primary_and_additional_schema_is_stable(self) -> None:
        cases = [
            ("", "", "", 0),
            ("latin_en", ".93", "English additional", 1),
            ("latin_en|latin_only", ".93|.96", "English additional|latin_only", 2),
        ]
        for additional_latin, additional_scores, expected_additional, additional_count in cases:
            with self.subTest(additional_count=additional_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                photo_dir = root / "photos"
                photo_dir.mkdir()
                output_dir = root / "report"
                primary = "latin_primary"
                species = {
                    "latin_primary": {"鸟种编号": "P", "中文名": "主鸟", "拉丁学名": "latin_primary", "英文名称": "Primary"},
                    "latin_cn": {"鸟种编号": "C", "中文名": "中文展示名", "拉丁学名": "latin_cn", "英文名称": "English C"},
                    "latin_en": {"鸟种编号": "E", "中文名": "", "拉丁学名": "latin_en", "英文名称": "English additional"},
                    "latin_only": {"鸟种编号": "L", "中文名": "", "拉丁学名": "latin_only", "英文名称": ""},
                }
                write_reports(
                    output_dir, photo_dir,
                    [
                        {"file_name": str(photo_dir / "bird.jpg"), "classification": primary, "score": .8},
                        {"file_name": str(photo_dir / "bird.jpg"), "classification": "latin_en", "score": .7},
                    ],
                    species, [], 1, .5, 1.0,
                    {"species_file": 0.0, "model_load": 0.0, "candidate_text": 0.0,
                     "image_inference": 0.0, "report_writing": 0.0},
                    {"bird.jpg": {
                        "baseline_top1_species": "latin_cn",
                        "crop_top1_species": "latin_en",
                        "primary_species": primary, "primary_score": ".800000",
                        "additional_species": additional_latin,
                        "additional_scores": additional_scores,
                        "additional_count": additional_count,
                    }},
                )
                with (output_dir / "predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                with (output_dir / "species_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                    summary = {row["拉丁学名"]: row for row in csv.DictReader(handle)}
                with (output_dir / "predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
                    report_fields = csv.DictReader(handle).fieldnames
                self.assertEqual(len(rows), 1)
                self.assertNotIn("rank", report_fields)
                self.assertEqual(rows[0]["拉丁学名"], primary)
                self.assertEqual(rows[0]["primary_species"], "主鸟")
                self.assertEqual(rows[0]["baseline_top1_species"], "中文展示名")
                self.assertEqual(rows[0]["crop_top1_species"], "English additional")
                self.assertEqual(rows[0]["additional_species"], expected_additional)
                self.assertEqual(rows[0]["additional_scores"], additional_scores)
                self.assertEqual(rows[0]["additional_count"], str(additional_count))
                self.assertEqual(set(next(iter(summary.values())).keys()), {
                    "鸟种编号", "中文名", "拉丁学名", "英文名称", "primary_count",
                    "additional_count", "max_score", "best_image",
                })
                self.assertEqual(summary[primary]["primary_count"], "1")
                self.assertEqual(summary[primary]["additional_count"], "0")
                for additional in additional_latin.split("|"):
                    if additional:
                        self.assertEqual(summary[additional]["primary_count"], "0")
                        self.assertEqual(summary[additional]["additional_count"], "1")
                self.assertNotIn("topk_count", next(iter(summary.values())))

    def test_experiment_top1_reader_accepts_rankless_formal_predictions(self) -> None:
        rows = [{"file_name": "bird.jpg", "中文名": "白鹭", "score": "0.9"}]
        top1 = matching.top1_rows(rows, Path("predictions.csv"))
        self.assertEqual(len(top1), 1)
        self.assertEqual(top1[0]["file_name"], "bird.jpg")
        self.assertNotIn("rank", top1[0])

    def test_neighbor_reader_accepts_rankless_predictions_and_prefers_primary_species(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["file_name", "中文名", "拉丁学名", "score", "primary_species"]
                )
                writer.writeheader()
                writer.writerow({
                    "file_name": "bird.jpg", "中文名": "表格中文名", "拉丁学名": "Ardea alba",
                    "score": "0.91", "primary_species": "主鸟显示名",
                })

            predictions = inspect_neighbor_consistency.load_predictions(path)

        self.assertEqual(predictions["bird.jpg"]["species"], "主鸟显示名")
        self.assertEqual(predictions["bird.jpg"]["score"], "0.910000")

    def test_additional_order_and_failed_image_summary_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo_dir = root / "photos"
            photo_dir.mkdir()
            output_dir = root / "report"
            species = {
                latin: {"鸟种编号": latin, "中文名": latin, "拉丁学名": latin, "英文名称": latin}
                for latin in ("primary", "additional_a", "additional_b", "ghost")
            }
            write_reports(
                output_dir, photo_dir,
                [{"file_name": str(photo_dir / "success.jpg"), "classification": "primary", "score": .8}],
                species, [], 2, .5, 1.0,
                {"species_file": 0.0, "model_load": 0.0, "candidate_text": 0.0,
                 "image_inference": 0.0, "report_writing": 0.0},
                {
                    "success.jpg": {
                        "additional_species": "additional_a|additional_b",
                        "additional_scores": ".91|.92",
                        "additional_count": 2,
                    },
                    "failed.jpg": {
                        "additional_species": "ghost",
                        "additional_scores": ".99",
                        "additional_count": 1,
                    },
                },
            )

            with (output_dir / "predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
                prediction = next(csv.DictReader(handle))
            with (output_dir / "species_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                summary = {row["拉丁学名"]: row for row in csv.DictReader(handle)}

        emitted = list(zip(prediction["additional_species"].split("|"), prediction["additional_scores"].split("|")))
        self.assertEqual(emitted, [("additional_a", ".91"), ("additional_b", ".92")])
        self.assertEqual(summary["additional_a"]["additional_count"], "1")
        self.assertEqual(summary["additional_b"]["additional_count"], "1")
        self.assertNotIn("ghost", summary)

    def test_windows_subdirectory_key_matches_report_diagnostics(self) -> None:
        photo_dir = PureWindowsPath("C:/Photos")
        image_path = photo_dir / "subdir" / "bird.jpg"
        self.assertEqual(relative_file_key(image_path, photo_dir), "subdir/bird.jpg")
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            species = {
                latin: {"鸟种编号": latin, "中文名": chinese, "拉丁学名": latin, "英文名称": ""}
                for latin, chinese in (("primary", "主鸟"), ("extra", "额外鸟"))
            }
            write_reports(
                output_dir, photo_dir,
                [{"file_name": image_path, "classification": "primary", "score": .81}],
                species, [], 1, .5, 1.0,
                {"species_file": 0.0, "model_load": 0.0, "candidate_text": 0.0,
                 "image_inference": 0.0, "report_writing": 0.0},
                {"subdir/bird.jpg": {
                    "final_source": "crop", "crop_used": "true",
                    "selected_crop_file": "diagnostics/crop_runtime/bird.jpg",
                    "detection_confidence": ".800000", "bbox_area_ratio": ".04000000",
                    "crop_top1_species": "primary", "crop_top1_score": ".810000",
                    "primary_species": "primary", "primary_score": ".810000",
                    "additional_species": "extra", "additional_scores": ".930000", "additional_count": 1,
                }},
            )
            with (output_dir / "predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            with (output_dir / "species_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                summary = {item["拉丁学名"]: item for item in csv.DictReader(handle)}

        self.assertEqual(row["file_name"], "subdir/bird.jpg")
        self.assertEqual(row["final_source"], "crop")
        self.assertEqual(row["crop_used"], "true")
        self.assertEqual(row["selected_crop_file"], "diagnostics/crop_runtime/bird.jpg")
        self.assertEqual(row["detection_confidence"], ".800000")
        self.assertEqual(row["bbox_area_ratio"], ".04000000")
        self.assertEqual(row["crop_top1_species"], "主鸟")
        self.assertEqual(row["crop_top1_score"], ".810000")
        self.assertEqual(row["primary_species"], "主鸟")
        self.assertEqual(row["additional_species"], "额外鸟")
        self.assertEqual(row["additional_scores"], ".930000")
        self.assertEqual(row["additional_count"], "1")
        self.assertEqual(summary["extra"]["additional_count"], "1")
        self.assertEqual(summary["extra"]["best_image"], "subdir/bird.jpg")

    def parse(self, extra: list[str] | None = None):
        old_argv = sys.argv
        fake_bioclip = ModuleType("bioclip")
        fake_predict = ModuleType("bioclip.predict")
        fake_predict.OPENA_AI_IMAGENET_TEMPLATE = ["template"] * 80
        fake_bioclip.predict = fake_predict
        try:
            sys.argv = ["scan_birds.py", "photos", "--species-file", "species.csv", *(extra or [])]
            with patch.dict(sys.modules, {"bioclip": fake_bioclip, "bioclip.predict": fake_predict}):
                return parse_args()
        finally:
            sys.argv = old_argv

    def test_default_crop_and_model(self) -> None:
        args = self.parse()
        self.assertTrue(args.crop)
        self.assertEqual(args.model, "bioclip25")
        self.assertEqual(args.top_k, 1)

    def test_top_k_can_be_overridden(self) -> None:
        self.assertEqual(self.parse(["--top-k", "7"]).top_k, 7)

    def test_no_crop_flag(self) -> None:
        self.assertFalse(self.parse(["--no-crop"]).crop)

    def test_both_models_accept_crop_and_no_crop_modes(self) -> None:
        for model in ("bioclip2", "bioclip25"):
            self.assertTrue(self.parse(["--model", model, "--crop"]).crop)
            self.assertFalse(self.parse(["--model", model, "--no-crop"]).crop)


class MainRecoveryTests(unittest.TestCase):
    def run_baseline_failure_case(self, crop_succeeds: bool) -> tuple[int, list[dict[str, str]], str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo_dir = root / "photos"
            photo_dir.mkdir()
            source = photo_dir / "bird.png"
            output_dir = root / "report"
            args = SimpleNamespace(
                species_file=root / "species.csv", photo_dir=photo_dir, output_dir=output_dir,
                model="bioclip25", device="cpu", prompt_count=80, top_k=1,
                batch_size=1, threshold=.5, crop=True,
            )
            species = {
                "bird": {"鸟种编号": "1", "中文名": "鸟", "拉丁学名": "bird", "英文名称": "Bird"}
            }
            calls = 0

            def fake_predict(_classifier, paths, _top_k, _batch_size):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return [], [(source, "BioCLIP prediction failed: baseline")]
                if crop_succeeds:
                    return [{"file_name": str(paths[0]), "classification": "bird", "score": .95}], []
                return [], [(paths[0], "BioCLIP prediction failed: crop")]

            class FakeImage:
                size = (10, 10)

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                def convert(self, _mode):
                    return self

                def crop(self, _bounds):
                    return self

                def save(self, _path):
                    return None

            fake_pil = ModuleType("PIL")
            fake_pil.Image = SimpleNamespace(open=lambda _path: FakeImage())
            with (
                patch("scan_birds.parse_args", return_value=args),
                patch("scan_birds.load_species", return_value=species),
                patch("scan_birds.find_images", return_value=[source]),
                patch("scan_birds.validate_images", return_value=([source], [])),
                patch("scan_birds.build_bioclip25_classifier", return_value=(object(), 0.0, False, root / "cache.pt")),
                patch("scan_birds.predict_resiliently", side_effect=fake_predict),
                patch("megadetector_utils.load_detector", return_value=Mock()),
                patch("megadetector_utils.extract_animals", return_value=[(.8, (0, 0, 2, 2))]),
                patch.dict(sys.modules, {"PIL": fake_pil}),
            ):
                exit_code = main()

            with (output_dir / "predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
                predictions = list(csv.DictReader(handle))
            summary = (output_dir / "run_summary.txt").read_text(encoding="utf-8")
        self.assertEqual(calls, 2)
        return exit_code, predictions, summary

    def test_baseline_failure_recovered_by_crop_is_final_success(self) -> None:
        exit_code, predictions, summary = self.run_baseline_failure_case(True)
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["final_source"], "crop")
        self.assertIn("Successful: 1", summary)
        self.assertIn("Failed: 0", summary)

    def test_baseline_and_crop_failure_is_final_failure(self) -> None:
        exit_code, predictions, summary = self.run_baseline_failure_case(False)
        self.assertEqual(exit_code, 2)
        self.assertEqual(predictions, [])
        self.assertIn("Successful: 0", summary)
        self.assertIn("Failed: 1", summary)


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
