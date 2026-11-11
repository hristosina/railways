"""Каталог исследуемых архитектур и единая точка создания backend-ов.

UI оперирует идентификатором профиля, а не импортирует конкретную библиотеку.
Это позволяет добавлять реализации новых архитектур без изменений вкладок
обучения и тестирования.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    rknn_status: str = "Требуется экспериментальная проверка экспорта и инференса."

    @property
    def available(self) -> bool:
        if self.backend_id in _BACKEND_FACTORIES:
            return True
        try:
            from external_model_backend import external_backend_ready
            return external_backend_ready(self.backend_id)[0]
        except Exception:
            return False


MODEL_PROFILES = (
    ModelProfile(
        id="yolov8s",
        display_name="YOLOv8s",
        paradigm="CNN",
        backend_id="ultralytics-yolo",
        default_artifact="yolov8s.pt",
        file_filter="Модели Ultralytics (*.pt *.yaml *.onnx *.engine *.torchscript)",
        description="Сверточный anchor-free детектор семейства YOLOv8, масштаб small.",
        rknn_status="Кандидат для экспорта в RKNN; совместимость проверяется на целевом устройстве.",
    ),
    ModelProfile(
        id="yolov12s",
        display_name="YOLOv12s",
        paradigm="CNN + attention",
        backend_id="ultralytics-yolo",
        default_artifact="yolov12s.pt",
        file_filter="Модели Ultralytics (*.pt *.yaml *.onnx *.engine *.torchscript)",
        description="YOLOv12 в масштабе small с механизмами внимания.",
        rknn_status="Кандидат для экспорта в RKNN; операции внимания требуют проверки конвертером.",
    ),
    ModelProfile(
        id="rtdetr_r18",
        display_name="RT-DETR-R18",
        paradigm="Compact Transformer",
        backend_id="official-rtdetr-r18",
        default_artifact="rtdetr_r18vd_dec3_6x_coco_from_paddle.pth",
        file_filter="Веса RT-DETR-R18 (*.pth);;Конфигурации RT-DETR (*.yml *.yaml)",
        description="Облегченный RT-DETR с backbone ResNet-18 (20 млн параметров).",
        backend_hint=(
            "Для RT-DETR-R18 используется официальная реализация lyuwenyu/RT-DETR "
            "в изолированном окружении venv310. Если проверка не проходит, запустите "
            "install_model_backends.bat."
        ),
        rknn_status="Не подтверждено: требуется проверка ONNX-графа на RKNN Toolkit2.",
    ),
    ModelProfile(
        id="rtdetrv2_s",
        display_name="RT-DETRv2-S",
        paradigm="Compact Transformer",
        backend_id="official-rtdetrv2-s",
        default_artifact="rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
        file_filter="Веса RT-DETRv2-S (*.pth);;Конфигурации RT-DETRv2 (*.yml *.yaml)",
        description="Улучшенный компактный RT-DETRv2-S (20 млн параметров).",
        backend_hint=(
            "Для RT-DETRv2-S используется официальный rtdetrv2_pytorch в "
            "изолированном окружении venv310. Если проверка не проходит, запустите "
            "install_model_backends.bat."
        ),
        rknn_status="Не подтверждено: требуется проверка ONNX-графа на RKNN Toolkit2.",
    ),
    ModelProfile(
        id="rfdetr_nano",
        display_name="RF-DETR-Nano",
        paradigm="Compact Transformer",
        backend_id="rfdetr-nano",
        default_artifact="rf-detr-nano.pth (скачается автоматически)",
        file_filter="Веса RF-DETR-Nano (*.pth *.ckpt)",
        description="Компактный RF-DETR Nano примерно на 10 млн параметров.",
        backend_hint=(
            "RF-DETR-Nano работает через изолированное окружение venv310 с "
            "совместимыми версиями RF-DETR, Transformers и CUDA PyTorch. Если "
            "проверка не проходит, запустите install_model_backends.bat."
        ),
        rknn_status=(
            "Не подтверждено: официальный экспорт поддерживает ONNX, но итоговый "
            "граф необходимо отдельно проверить в RKNN Toolkit2."
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


def infer_model_profile_from_artifact(artifact_path: str | Path) -> str | None:
    """Определяет профиль по стандартному пути runs/<profile_id>/... ."""
    parts = {part.casefold() for part in Path(artifact_path).parts}
    for profile in MODEL_PROFILES:
        if profile.id.casefold() in parts:
            return profile.id
    return None


def model_profile_status(profile_id: str) -> tuple[bool, str]:
    profile = get_model_profile(profile_id)
    if profile.available:
        return True, (
            f"Backend: {profile.backend_id}. {profile.description} "
            f"RKNN: {profile.rknn_status}"
        )
    try:
        from external_model_backend import external_backend_ready
        _ready, runtime_status = external_backend_ready(profile.backend_id)
    except Exception:
        runtime_status = ""
    backend_status = " ".join(filter(None, (
        profile.backend_hint or f"Backend {profile.backend_id!r} не подключен.",
        runtime_status,
    )))
    return False, f"{backend_status} RKNN: {profile.rknn_status}"


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


register_model_backend("ultralytics-yolo", _create_ultralytics_yolo)
