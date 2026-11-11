"""Делает официальный RT-DETR v1 совместимым с TorchVision 0.20+.

Upstream RT-DETR-R18 использует beta API ``torchvision.datapoints``, который
был переименован в ``torchvision.tv_tensors``. Патч ограничен заменой API и не
меняет архитектуру, веса или алгоритм обучения.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "third_party" / "RT-DETR" / "rtdetr_pytorch" / "src" / "data"


def patch_file(path: Path, replacements: tuple[tuple[str, str], ...]) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
        elif new not in text:
            raise RuntimeError(f"Не удалось применить совместимость к {path}: не найден фрагмент {old!r}")
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> int:
    coco = SOURCE_ROOT / "coco" / "coco_dataset.py"
    transforms = SOURCE_ROOT / "transforms.py"
    if not coco.is_file() or not transforms.is_file():
        raise FileNotFoundError("Не найдена официальная копия RT-DETR-R18.")

    changed = patch_file(coco, (
        ("from torchvision import datapoints", "from torchvision import tv_tensors"),
        ("datapoints.BoundingBox(", "tv_tensors.BoundingBoxes("),
        ("format=datapoints.BoundingBoxFormat.XYXY,", "format=tv_tensors.BoundingBoxFormat.XYXY,"),
        ("spatial_size=img.size[::-1]", "canvas_size=img.size[::-1]"),
        ("datapoints.Mask(", "tv_tensors.Mask("),
    ))
    changed |= patch_file(transforms, (
        ("from torchvision import datapoints", "from torchvision import tv_tensors"),
        ("datapoints.Image", "tv_tensors.Image"),
        ("datapoints.Video", "tv_tensors.Video"),
        ("datapoints.Mask", "tv_tensors.Mask"),
        ("datapoints.BoundingBoxFormat", "tv_tensors.BoundingBoxFormat"),
        ("datapoints.BoundingBox", "tv_tensors.BoundingBoxes"),
        ("inpt.spatial_size", "inpt.canvas_size"),
        ("spatial_size=spatial_size", "canvas_size=spatial_size"),
    ))

    # В новом API изменились также имена двух transform-классов. Сохраняем
    # старые имена конфигурации RT-DETR через тонкие классы-обертки.
    text = transforms.read_text(encoding="utf-8")
    old_block = "ToImageTensor = register(T.ToImageTensor)\nConvertDtype = register(T.ConvertDtype)\nSanitizeBoundingBox = register(T.SanitizeBoundingBox)"
    new_block = (
        "@register\nclass ToImageTensor(T.ToImage):\n"
        "    \"\"\"Совместимое имя старого beta API для torchvision >= 0.16.\"\"\"\n\n\n"
        "@register\nclass ConvertDtype(T.ToDtype):\n"
        "    \"\"\"Старый ConvertDtype соответствовал float32 с масштабированием 0..1.\"\"\"\n"
        "    def __init__(self, dtype=torch.float32, scale=True):\n"
        "        super().__init__(dtype=dtype, scale=scale)\n\n\n"
        "@register\nclass SanitizeBoundingBox(T.SanitizeBoundingBoxes):\n"
        "    \"\"\"Совместимое имя старого beta API для torchvision >= 0.16.\"\"\""
    )
    if old_block in text:
        transforms.write_text(text.replace(old_block, new_block), encoding="utf-8")
        changed = True
    elif not all(marker in text for marker in (
        "class ToImageTensor(T.ToImage)",
        "class ConvertDtype(T.ToDtype)",
        "class SanitizeBoundingBox(T.SanitizeBoundingBoxes)",
    )):
        raise RuntimeError("Не удалось добавить совместимые transform-классы RT-DETR.")

    print("RT-DETR TorchVision compatibility: " + ("patched" if changed else "already applied"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
