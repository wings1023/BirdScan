import csv
import tempfile
import unittest
from pathlib import Path

from scan_birds import load_species
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


if __name__ == "__main__":
    unittest.main()
