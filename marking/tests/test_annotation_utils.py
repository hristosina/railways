import unittest

from annotation_utils import (
    canonical_annotation_line,
    merge_annotation_lines,
    merge_preserving_existing_lines,
)


class AnnotationUtilsTests(unittest.TestCase):
    def test_merge_preserves_current_and_adds_previous(self):
        current = ["0 0.5 0.5 0.2 0.2"]
        previous = ["1 0.4 0.4 0.1 0.1"]
        self.assertEqual(
            merge_annotation_lines(current, previous),
            [
                "0 0.500000 0.500000 0.200000 0.200000",
                "1 0.400000 0.400000 0.100000 0.100000",
            ],
        )

    def test_merge_does_not_duplicate_same_box(self):
        line = "0 0.5 0.5 0.2 0.2"
        self.assertEqual(len(merge_annotation_lines([line], [line])), 1)

    def test_canonicalization_does_not_change_geometry(self):
        line = "2 0.123456 0.654321 0.200000 0.300000"
        value = canonical_annotation_line(line)
        for _ in range(20):
            value = canonical_annotation_line(value)
        self.assertEqual(value, line)

    def test_folder_merge_preserves_invalid_existing_text(self):
        existing = ["0 0.5 0.5 0.2 0.2", "invalid legacy line"]
        result = merge_preserving_existing_lines(
            existing, ["1 0.4 0.4 0.1 0.1"]
        )
        self.assertEqual(result[1], "invalid legacy line")
        self.assertEqual(result[-1], "1 0.400000 0.400000 0.100000 0.100000")


if __name__ == "__main__":
    unittest.main()
