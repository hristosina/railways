import unittest
from model_catalog import (
    ModelBackendUnavailable,
    available_model_profiles,
    create_model_runtime,
    model_profile_status,
    register_model_backend,
)


class ModelCatalogTests(unittest.TestCase):
    def test_catalog_matches_thesis_architectures(self):
        self.assertEqual(
            [profile.id for profile in available_model_profiles()],
            ["yolov8s", "yolov12s", "rtdetr", "rtdetrv2", "dino"],
        )

    def test_supported_backends_are_reported(self):
        self.assertTrue(model_profile_status("yolov8s")[0])
        self.assertTrue(model_profile_status("yolov12s")[0])
        self.assertTrue(model_profile_status("rtdetr")[0])
        self.assertFalse(model_profile_status("rtdetrv2")[0])
        self.assertFalse(model_profile_status("dino")[0])

    def test_unavailable_backend_fails_with_clear_message(self):
        with self.assertRaisesRegex(ModelBackendUnavailable, "RT-DETRv2"):
            create_model_runtime("rtdetrv2", "weights.pth")

    def test_registration_rejects_invalid_factory(self):
        with self.assertRaises(ValueError):
            register_model_backend("invalid", None)


if __name__ == "__main__":
    unittest.main()
