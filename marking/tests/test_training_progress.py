import unittest

from training_progress import calculate_training_progress


class TrainingProgressTests(unittest.TestCase):
    def test_first_batch_reports_fraction_of_first_epoch(self):
        self.assertEqual(calculate_training_progress(0, 1, 10, 5), (1, 10, 2))

    def test_last_batch_of_last_epoch_is_complete(self):
        self.assertEqual(calculate_training_progress(4, 10, 10, 5), (5, 100, 100))

    def test_values_are_clamped(self):
        self.assertEqual(calculate_training_progress(99, 999, 10, 5), (5, 100, 100))


if __name__ == "__main__":
    unittest.main()
