import tempfile
import unittest
from pathlib import Path

from dataset_utils import (
    find_annotation_sets,
    move_annotation_pair_to_trash,
    normalize_dataset_path,
    resolve_annotation_set,
)
from tools.organize_test_scenarios import (
    apply_plan,
    build_plan,
    infer_scenario_rules,
    scenario_from_name,
)


class DatasetUtilsTests(unittest.TestCase):
    def test_image_and_label_are_moved_to_recoverable_trash(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = Path(tmp) / "train"
            images = split / "images"
            labels = split / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            image = images / "frame.jpg"
            label = labels / "frame.txt"
            image.write_bytes(b"image")
            label.write_text("0 0.5 0.5 1 1", encoding="utf-8")

            moved_image, moved_label, trash = move_annotation_pair_to_trash(image, label)

            self.assertFalse(image.exists())
            self.assertFalse(label.exists())
            self.assertEqual(moved_image.read_bytes(), b"image")
            self.assertEqual(moved_label.read_text(encoding="utf-8"), "0 0.5 0.5 1 1")
            self.assertEqual(trash, split / ".marking_trash")
            self.assertNotIn(trash, find_annotation_sets(Path(tmp)))

    def test_trash_does_not_overwrite_an_existing_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = Path(tmp) / "test"
            images = split / "images"
            labels = split / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            for content in (b"first", b"second"):
                image = images / "same.png"
                label = labels / "same.txt"
                image.write_bytes(content)
                label.write_text(content.decode(), encoding="utf-8")
                moved_image, moved_label, _ = move_annotation_pair_to_trash(image, label)
            self.assertEqual(moved_image.name, "same_1.png")
            self.assertEqual(moved_label.name, "same_1.txt")

    def test_images_selection_resolves_to_dataset_root_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.yaml").write_text("names: []", encoding="utf-8")
            (root / "test" / "images").mkdir(parents=True)
            (root / "test" / "labels").mkdir()
            (root / "test" / "images" / "frame.jpg").write_bytes(b"image")

            self.assertEqual(normalize_dataset_path(root / "test" / "images"), root)
            annotation_root, images, labels = resolve_annotation_set(root / "test" / "images")
            self.assertEqual(annotation_root, root / "test")
            self.assertEqual(images, root / "test" / "images")
            self.assertEqual(labels, root / "test" / "labels")
            self.assertEqual(find_annotation_sets(root), [root / "test"])

    def test_scenario_plan_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "test" / "images"
            labels = root / "test" / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            name = "test_fog_000001_jpg.rf.hash"
            (images / f"{name}.jpg").write_bytes(b"image")
            (labels / f"{name}.txt").write_text("0 0.5 0.5 1 1", encoding="utf-8")

            output, plan, unknown, missing = build_plan(root, None)
            self.assertEqual(output, root / "test_scenarios")
            self.assertEqual(len(plan), 2)
            self.assertFalse(unknown)
            self.assertFalse(missing)
            self.assertEqual(scenario_from_name(f"{name}.jpg"), "fog")

    def test_scenario_plan_can_be_applied_without_changing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "test" / "images"
            labels = root / "test" / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            image = images / "test_night_000001.jpg"
            label = labels / "test_night_000001.txt"
            image.write_bytes(b"image")
            label.write_text("0 0.5 0.5 1 1", encoding="utf-8")
            output, plan, unknown, missing = build_plan(root, None)
            status = apply_plan(plan, strategy="copy")
            self.assertEqual(status["created"], 2)
            self.assertEqual((output / "night" / "images" / image.name).read_bytes(), b"image")
            self.assertTrue(image.is_file())
            self.assertFalse(unknown)
            self.assertFalse(missing)

    def test_custom_scenario_keywords_and_output_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "test" / "images"
            labels = root / "test" / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            image = images / "camera_snowstorm_001.jpg"
            label = labels / "camera_snowstorm_001.txt"
            image.write_bytes(b"image")
            label.write_text("0 0.5 0.5 1 1", encoding="utf-8")
            rules = [{
                "name": "Снег",
                "folder": "snow",
                "keywords": ("snowstorm", "snow"),
            }]

            output, plan, unknown, missing = build_plan(
                root, root / "custom_scenarios", scenario_rules=rules
            )

            self.assertEqual(output, root / "custom_scenarios")
            self.assertEqual(plan[0][1], output / "snow" / "images" / image.name)
            self.assertFalse(unknown)
            self.assertFalse(missing)
            self.assertEqual(scenario_from_name(image.name, rules), "snow")

    def test_infers_unknown_prefix_as_editable_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "test" / "images"
            labels = root / "test" / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            (images / "test_snow_001.jpg").write_bytes(b"image")

            rules = infer_scenario_rules(root)

            self.assertEqual(rules, [{
                "name": "Snow", "folder": "snow", "keywords": ("snow",)
            }])

    def test_custom_scenarios_root_is_not_replaced_by_yaml_ancestor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.yaml").write_text("names: []", encoding="utf-8")
            scenarios = root / "test_scenarios"
            scenarios.mkdir()
            self.assertEqual(normalize_dataset_path(scenarios), scenarios)


if __name__ == "__main__":
    unittest.main()
