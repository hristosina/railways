"""Проверка отдельного окружения моделей без запуска долгого обучения."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_import(module_name: str, shown_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "версия не указана")
        print(f"[OK] {shown_name}: {version}")
        return True
    except Exception as exc:
        print(f"[ОШИБКА] {shown_name}: {exc}")
        return False


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Интерпретатор: {sys.executable}")

    checks = [
        check_import("torch", "PyTorch"),
        check_import("torchvision", "TorchVision"),
        check_import("ultralytics", "Ultralytics"),
        check_import("rfdetr", "RF-DETR"),
        check_import("pycocotools", "pycocotools"),
    ]

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"[{'OK' if cuda_available else 'ОШИБКА'}] CUDA: {cuda_available}")
        if cuda_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        checks.append(cuda_available)
    except Exception as exc:
        print(f"[ОШИБКА] Проверка CUDA: {exc}")
        checks.append(False)

    required_files = (
        PROJECT_ROOT / "third_party" / "RT-DETR" / "rtdetr_pytorch" / "tools" / "train.py",
        PROJECT_ROOT / "third_party" / "RT-DETR" / "rtdetrv2_pytorch" / "tools" / "train.py",
        PROJECT_ROOT / "third_party" / "RT-DETR" / "rtdetr_pytorch" / "configs" / "rtdetr" / "rtdetr_r18vd_6x_coco.yml",
        PROJECT_ROOT / "third_party" / "RT-DETR" / "rtdetrv2_pytorch" / "configs" / "rtdetrv2" / "rtdetrv2_r18vd_120e_coco.yml",
    )
    for path in required_files:
        exists = path.is_file()
        print(f"[{'OK' if exists else 'ОШИБКА'}] {path.relative_to(PROJECT_ROOT)}")
        checks.append(exists)

    for implementation in ("rtdetr_pytorch", "rtdetrv2_pytorch"):
        root = PROJECT_ROOT / "third_party" / "RT-DETR" / implementation
        result = subprocess.run(
            [sys.executable, "-c", "import src.data; print('import src.data: OK')"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        imported = result.returncode == 0
        print(f"[{'OK' if imported else 'ОШИБКА'}] {implementation}: import src.data")
        if not imported:
            print(result.stdout)
            print(result.stderr)
        checks.append(imported)

    if all(checks):
        print("\nВсе backend-ы и CUDA доступны.")
        return 0
    print("\nОкружение установлено не полностью.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
