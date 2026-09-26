"""Каталог исследуемых архитектур и единая точка создания backend-ов.

UI оперирует идентификатором профиля, а не импортирует конкретную библиотеку.
Это позволяет добавлять реализации новых архитектур без изменений вкладок
обучения и тестирования.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


class ModelBackendUnavailable(RuntimeError):
    """Выбранная архитектура известна, но ее backend еще не установлен."""


@dataclass(frozen=True)
class ModelProfile:
    id: str
    display_name: str
    paradigm: str
    backend_id: str
    default_artifact: str
    file_filter: str
    description: str
    backend_hint: str = ""

    @property
    def available(self) -> bool:
        return self.backend_id in _BACKEND_FACTORIES


MODEL_PROFILES = (
    ModelProfile(
        id="yolov8s",
        display_name="YOLOv8s",
        paradigm="CNN",
        backend_id="ultralytics-yolo",
        default_artifact="yolov8s.pt",
        file_filter="Модели Ultralytics (*.pt *.yaml *.onnx *.engine *.torchscript)",
        description="Сверточный anchor-free детектор семейства YOLOv8, масштаб small.",
    ),
    ModelProfile(
        id="yolov12s",
        display_name="YOLOv12s",
        paradigm="CNN + attention",
        backend_id="ultralytics-yolo",
        default_artifact="yolov12s.pt",
        file_filter="Модели Ultralytics (*.pt *.yaml *.onnx *.engine *.torchscript)",
        description="YOLOv12 в масштабе small с механизмами внимания.",
    ),
    ModelProfile(
        id="rtdetr",
        display_name="RT-DETR",
        paradigm="Transformer",
        backend_id="ultralytics-rtdetr",
        default_artifact="rtdetr-l.pt",
        file_filter="Модели RT-DETR (*.pt *.yaml *.onnx *.engine)",
        description="Трансформерный end-to-end детектор без NMS.",
    ),
    ModelProfile(
        id="rtdetrv2",
        display_name="RT-DETRv2",
        paradigm="Transformer",
        backend_id="rtdetrv2",
        default_artifact="checkpoint.pth",
        file_filter="Веса RT-DETRv2 (*.pth *.pt);;Конфигурации (*.yml *.yaml *.json)",
        description="Усовершенствованная версия RT-DETR.",
        backend_hint=(
            "Для RT-DETRv2 требуется отдельный адаптер официальной реализации: "
            "ее формат конфигурации и контрольных точек несовместим с Ultralytics."
        ),
    ),
    ModelProfile(
        id="dino",
        display_name="DINO (DETR)",
        paradigm="Transformer",
        backend_id="dino",
        default_artifact="checkpoint.pth",
        file_filter="Веса DINO (*.pth *.pt);;Конфигурации (*.py *.yaml *.yml)",
        description="DINO с denoising-обучением и инициализацией anchor-запросов.",
        backend_hint=(
            "Для DINO требуется отдельный адаптер официальной реализации или detrex; "
            "это не модель Grounding DINO."
        ),
    ),
)

_PROFILES_BY_ID = {profile.id: profile for profile in MODEL_PROFILES}
_BACKEND_FACTORIES: dict[str, Callable[[str], object]] = {}


def register_model_backend(backend_id: str, factory: Callable[[str], object]) -> None:
    """Регистрирует загрузчик, возвращающий объект с API train/val/callback."""
    if not backend_id or not callable(factory):
        raise ValueError("Нужны непустой backend_id и вызываемая factory.")
    _BACKEND_FACTORIES[backend_id] = factory


def get_model_profile(profile_id: str) -> ModelProfile:
    try:
        return _PROFILES_BY_ID[profile_id]
    except KeyError as exc:
        raise ValueError(f"Неизвестный тип модели: {profile_id!r}.") from exc


def available_model_profiles() -> tuple[ModelProfile, ...]:
    return MODEL_PROFILES


def model_profile_status(profile_id: str) -> tuple[bool, str]:
    profile = get_model_profile(profile_id)
    if profile.available:
        return True, f"Backend: {profile.backend_id}. {profile.description}"
    return False, profile.backend_hint or f"Backend {profile.backend_id!r} не подключен."


def create_model_runtime(profile_id: str, artifact_path: str):
    profile = get_model_profile(profile_id)
    factory = _BACKEND_FACTORIES.get(profile.backend_id)
    if factory is None:
        raise ModelBackendUnavailable(
            f"{profile.display_name}: {profile.backend_hint or 'backend не подключен.'}"
        )
    if not str(artifact_path).strip():
        raise ValueError(f"Для {profile.display_name} укажите начальные веса или конфигурацию.")
    return factory(str(artifact_path))


def _create_ultralytics_yolo(path: str):
    from ultralytics import YOLO

    return YOLO(path)


def _create_ultralytics_rtdetr(path: str):
    from ultralytics import RTDETR

    return RTDETR(path)


register_model_backend("ultralytics-yolo", _create_ultralytics_yolo)
register_model_backend("ultralytics-rtdetr", _create_ultralytics_rtdetr)
