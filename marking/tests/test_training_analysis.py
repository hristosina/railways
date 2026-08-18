import csv
import tempfile
import unittest
from pathlib import Path

from training_analysis import analyze_training_results, save_training_analysis


def write_history(path, train_values, val_values, map_values):
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("epoch", "train/box_loss", "val/box_loss", "metrics/mAP50-95(B)"))
        for index, values in enumerate(zip(train_values, val_values, map_values), start=1):
            writer.writerow((index, *values))


class TrainingAnalysisTests(unittest.TestCase):
    def test_recommends_more_epochs_when_both_losses_fall(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            write_history(
                path,
                [1.2 - index * 0.02 for index in range(30)],
                [1.5 - index * 0.015 for index in range(30)],
                [0.2 + index * 0.01 for index in range(30)],
            )
            analysis = analyze_training_results(path)
            self.assertEqual(analysis["verdict"], "continue")

    def test_detects_overfitting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            write_history(
                path,
                [1.0 - index * 0.02 for index in range(30)],
                [0.8 - index * 0.01 if index < 15 else 0.65 + (index - 15) * 0.02 for index in range(30)],
                [0.3 + min(index, 15) * 0.01 for index in range(30)],
            )
            analysis = analyze_training_results(path)
            self.assertEqual(analysis["verdict"], "overfitting")

    def test_detects_low_quality_plateau_and_saves_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            write_history(path, [0.9] * 30, [1.4] * 30, [0.15] * 30)
            analysis = analyze_training_results(path)
            self.assertEqual(analysis["verdict"], "check_data")
            report = save_training_analysis(directory, analysis)
            self.assertIn("Проверьте", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
