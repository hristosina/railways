"""Сохранение воспроизводимых параметров запуска обучения."""

from __future__ import annotations

import platform
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml


TRAINING_PARAMETERS_FILENAME = "Параметры_обучения.yaml"


def _serializable(value: Any):
    """Преобразует параметры сторонней библиотеки в безопасные YAML-значения."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serializable(item) for item in value]
    return str(value)


def _trainer_arguments(trainer) -> dict:
    if trainer is None:
        return {}
    args = getattr(trainer, "args", None)
    if args is None:
        return {}
    if isinstance(args, dict):
        return _serializable(args)
    try:
        return _serializable(vars(args))
    except TypeError:
        return {}


def _environment_info() -> dict:
    try:
        ultralytics_version = metadata.version("ultralytics")
    except metadata.PackageNotFoundError:
        ultralytics_version = "не установлен"

    try:
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        torch_version = "не установлен"

    cuda_available = False
    cuda_name = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_name = torch.cuda.get_device_name(0)
    except Exception:
        # Сведения о среде дополняют отчет, но не должны срывать обучение.
        cuda_name = "не удалось определить"

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "ultralytics": ultralytics_version,
        "cuda_available": cuda_available,
        "cuda_device": cuda_name,
    }


def save_training_parameters(
    run_dir,
    requested_parameters: dict,
    *,
    trainer=None,
    status: str = "выполняется",
    started_at: str | None = None,
    completed_at: str | None = None,
) -> Path:
    """Записывает параметры конкретного запуска рядом с его результатами."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    requested = _serializable(requested_parameters)
    document = {
        "format_version": 1,
        "status": status,
        "started_at": started_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "completed_at": completed_at,
        "model": {
            "profile_id": requested.get("model_profile_id"),
            "architecture": requested.get("model_display_name"),
            "paradigm": requested.get("model_paradigm"),
            "backend": requested.get("backend_id"),
            "initial_artifact": requested.get("model_path"),
        },
        "dataset": {
            "root": requested.get("dataset_root"),
            "data_yaml": requested.get("dataset_yaml"),
        },
        "training": {
            "mode": "resume" if requested.get("resume") else "new_or_finetune",
            "resume_checkpoint": (
                requested.get("model_path") if requested.get("resume") else None
            ),
            "epochs": requested.get("epochs"),
            "image_size": requested.get("imgsz"),
            "batch_size": requested.get("batch"),
            "patience": requested.get("patience"),
            "use_gpu": requested.get("gpu"),
            "device": requested.get("device"),
            "workers": requested.get("workers"),
        },
        "environment": _environment_info(),
        "backend_parameters": _trainer_arguments(trainer),
    }
    path = run_dir / TRAINING_PARAMETERS_FILENAME
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(document, file, allow_unicode=True, sort_keys=False)
    return path
