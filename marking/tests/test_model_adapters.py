import unittest

import model_adapters


class FakeAdapter(model_adapters.DetectionModelAdapter):
    backend_id = "fake"
    display_name = "Fake"

    @classmethod
    def can_load(cls, model_path):
        return str(model_path).endswith(".fake")

    def __init__(self, model_path):
        self.model_path = model_path

    @property
    def class_names(self):
        return {0: "object"}

    def predict(self, image_path, confidence):
        return []


class ModelAdapterTests(unittest.TestCase):
    def test_project_adapter_can_be_registered_without_editor_changes(self):
        model_adapters.register_adapter(FakeAdapter)
        adapter = model_adapters.create_detection_adapter("model.fake")
        self.assertIsInstance(adapter, FakeAdapter)
        self.assertEqual(adapter.class_names, {0: "object"})

    def test_unknown_model_format_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Не найден адаптер"):
            model_adapters.create_detection_adapter("model.unsupported")


if __name__ == "__main__":
    unittest.main()
