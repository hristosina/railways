"""Create a scenario-organized view of a flat Roboflow YOLO test split.

The source dataset is left untouched.  By default the command only prints a
plan; pass --apply to create hard links or copies.
"""

import argparse
import os
import re
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCENARIO_RE = re.compile(
    r"^(?:test_)?(?P<scenario>afternoon|night|twilight|fog|fallout)_",
    re.IGNORECASE,
)


def find_test_images(path):
    path = Path(path).expanduser().resolve()
    candidates = [path, path / "images", path / "test" / "images"]
    for candidate in candidates:
        if candidate.is_dir() and any(
            p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            for p in candidate.iterdir()
        ):
            return candidate
    raise ValueError("Не найдена плоская папка test/images с изображениями.")


def scenario_from_name(filename):
    match = SCENARIO_RE.match(filename)
    return match.group("scenario").lower() if match else None


def transfer(source, destination, strategy):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size:
            return "skipped"
        raise FileExistsError(f"Файл уже существует и отличается: {destination}")
    if strategy == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)
    return "created"


def build_plan(source_path, output_path):
    images_dir = find_test_images(source_path)
    labels_dir = images_dir.parent / "labels"
    if not labels_dir.is_dir():
        raise ValueError(f"Не найдена парная папка labels: {labels_dir}")

    output = Path(output_path).expanduser().resolve() if output_path else images_dir.parents[1] / "test_scenarios"
    plan = []
    unknown = []
    missing_labels = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        scenario = scenario_from_name(image_path.name)
        if not scenario:
            unknown.append(image_path)
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels.append(label_path)
            continue
        plan.append((image_path, output / scenario / "images" / image_path.name))
        plan.append((label_path, output / scenario / "labels" / label_path.name))
    return output, plan, unknown, missing_labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Корень датасета, папка test или test/images")
    parser.add_argument("--output", help="Выходная папка (по умолчанию: test_scenarios рядом с test)")
    parser.add_argument("--strategy", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--apply", action="store_true", help="Выполнить план; без флага работает dry-run")
    args = parser.parse_args(argv)

    output, plan, unknown, missing_labels = build_plan(args.source, args.output)
    counts = Counter(destination.parents[1].name for _, destination in plan[::2])
    print(f"Выходная папка: {output}")
    print("Сценарии: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
    print(f"Не распознано изображений: {len(unknown)}")
    print(f"Отсутствует файлов разметки: {len(missing_labels)}")

    if unknown or missing_labels:
        print("План не выполнен: сначала устраните перечисленные несоответствия.")
        for path in [*unknown[:10], *missing_labels[:10]]:
            print(f"  {path}")
        return 2
    if not args.apply:
        print(f"Dry-run: готово к созданию {len(plan)} файлов. Добавьте --apply для выполнения.")
        return 0

    status = Counter(transfer(source, destination, args.strategy) for source, destination in plan)
    print(f"Готово: создано {status['created']}, уже существовало {status['skipped']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
