"""Model-independent interface used by automatic annotation.

New inference backends only need to implement ``DetectionModelAdapter`` and
register the class. The editor and the YOLO annotation writer stay unchanged.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    x: float
    y: float
    width: float
    height: float


class DetectionModelAdapter(ABC):
    backend_id = ""
    display_name = ""

    @classmethod
    @abstractmethod
    def can_load(cls, model_path):
        """Return True when this backend can open the supplied artifact."""

    @property
    @abstractmethod
    def class_names(self):
        """Return a mapping from numeric class id to display name."""

    @abstractmethod
    def predict(self, image_path, confidence):
        """Return normalized detections for one image."""


_ADAPTERS = []


def register_adapter(adapter_class):
    # Project-specific adapters registered later get priority over the fallback.
    _ADAPTERS.insert(0, adapter_class)
    return adapter_class


def create_detection_adapter(model_path, backend_id=None):
    for adapter_class in _ADAPTERS:
        if backend_id and adapter_class.backend_id != backend_id:
            continue
        if adapter_class.can_load(model_path):
            return adapter_class(model_path)
    requested = f" backend '{backend_id}'" if backend_id else ""
    raise ValueError(
        f"Не найден адаптер{requested} для модели {Path(model_path).name}. "
        "Добавьте реализацию DetectionModelAdapter."
    )


@register_adapter
class UltralyticsDetectionAdapter(DetectionModelAdapter):
    backend_id = "ultralytics"
    display_name = "Ultralytics"
    supported_suffixes = {".pt", ".onnx", ".engine", ".torchscript"}

    @classmethod
    def can_load(cls, model_path):
        return Path(model_path).suffix.lower() in cls.supported_suffixes

    def __init__(self, model_path):
        from ultralytics import YOLO

        self._model = YOLO(model_path)

    @property
    def class_names(self):
        names = self._model.names
        if isinstance(names, dict):
            return {int(class_id): str(name) for class_id, name in names.items()}
        return {class_id: str(name) for class_id, name in enumerate(names)}

    def predict(self, image_path, confidence):
        detections = []
        for result in self._model.predict(str(image_path), conf=confidence, save=False):
            image_height, image_width = result.orig_shape[:2]
            for box in result.boxes:
                class_id = int(box.cls[0])
                x, y, width, height = box.xywh[0]
                detections.append(Detection(
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    x=float(x / image_width),
                    y=float(y / image_height),
                    width=float(width / image_width),
                    height=float(height / image_height),
                ))
        return detections
