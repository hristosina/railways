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
SCENARIO_RE = re.compile(r"^(?:test_)?(?P<scenario>[^_]+)_", re.IGNORECASE)
DEFAULT_SCENARIO_RULES = (
    {"name": "Осадки", "folder": "fallout", "keywords": ("fallout", "precipitation", "rain", "осадки")},
    {"name": "Туман", "folder": "fog", "keywords": ("fog", "туман")},
    {"name": "День", "folder": "afternoon", "keywords": ("afternoon", "day", "день")},
    {"name": "Ночь", "folder": "night", "keywords": ("night", "ночь")},
    {"name": "Сумерки", "folder": "twilight", "keywords": ("twilight", "dusk", "сумерки")},
)


def safe_folder_name(value):
    value = re.sub(r'[<>:"/\\|?*]+', "_", str(value).strip())
    value = re.sub(r"\s+", "_", value).strip(" ._")
    return value or "scenario"


def normalize_scenario_rules(rules=None):
    normalized = []
    for index, rule in enumerate(rules or DEFAULT_SCENARIO_RULES, start=1):
        if isinstance(rule, dict):
            name = str(rule.get("name", "")).strip()
            folder = safe_folder_name(rule.get("folder") or name)
            keywords = rule.get("keywords", ())
        else:
            name, keywords = rule[:2]
            folder = safe_folder_name(rule[2] if len(rule) > 2 else name)
        if isinstance(keywords, str):
            keywords = re.split(r"[,;\n]+", keywords)
        keywords = tuple(dict.fromkeys(
            str(keyword).strip().casefold() for keyword in keywords
            if str(keyword).strip()
        ))
        if not name:
            raise ValueError(f"Сценарий {index}: не указано название.")
        if not keywords:
            raise ValueError(f"Сценарий «{name}»: укажите хотя бы одно ключевое слово.")
        normalized.append({"name": name, "folder": folder, "keywords": keywords})
    return normalized


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


def _keyword_in_filename(filename, keyword):
    stem = Path(filename).stem.casefold()
    keyword_pattern = re.escape(keyword.casefold()).replace(r"\_", r"[_\-\s]+")
    return bool(re.search(rf"(?:^|[_\-\s]){keyword_pattern}(?:[_\-\s]|$)", stem))


def scenario_from_name(filename, scenario_rules=None):
    for rule in normalize_scenario_rules(scenario_rules):
        if any(_keyword_in_filename(filename, keyword) for keyword in rule["keywords"]):
            return rule["folder"]
    return None


def infer_scenario_rules(source_path):
    """Предлагает правила из реальных префиксов, не ограничивая их фиксированным списком."""
    images_dir = find_test_images(source_path)
    prefixes = Counter()
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = SCENARIO_RE.match(image_path.name)
        if match:
            prefixes[match.group("scenario").casefold()] += 1

    proposals = []
    used_defaults = set()
    for prefix in sorted(prefixes):
        default_index = next((
            index for index, rule in enumerate(DEFAULT_SCENARIO_RULES)
            if prefix in rule["keywords"]
        ), None)
        if default_index is not None:
            if default_index in used_defaults:
                continue
            default = DEFAULT_SCENARIO_RULES[default_index]
            present_keywords = tuple(
                keyword for keyword in default["keywords"] if keyword in prefixes
            ) or (prefix,)
            proposals.append({
                "name": default["name"],
                "folder": default["folder"],
                "keywords": present_keywords,
            })
            used_defaults.add(default_index)
        else:
            proposals.append({
                "name": prefix.replace("-", " ").replace("_", " ").capitalize(),
                "folder": safe_folder_name(prefix),
                "keywords": (prefix,),
            })
    return proposals


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


def apply_plan(plan, strategy="hardlink", fallback_to_copy=True, progress_callback=None):
    """Создает сценарное представление, при необходимости заменяя hardlink копией."""
    status = Counter()
    total = len(plan)
    for index, (source, destination) in enumerate(plan, start=1):
        try:
            result = transfer(source, destination, strategy)
        except OSError:
            if strategy != "hardlink" or not fallback_to_copy:
                raise
            result = transfer(source, destination, "copy")
            if result == "created":
                result = "copied"
        status[result] += 1
        if progress_callback:
            progress_callback(index, total)
    return status


def build_plan(source_path, output_path, scenario_rules=None):
    images_dir = find_test_images(source_path)
    labels_dir = images_dir.parent / "labels"
    if not labels_dir.is_dir():
        raise ValueError(f"Не найдена парная папка labels: {labels_dir}")

    output = Path(output_path).expanduser().resolve() if output_path else images_dir.parents[1] / "test_scenarios"
    plan = []
    unknown = []
    missing_labels = []
    rules = normalize_scenario_rules(scenario_rules)
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        scenario = scenario_from_name(image_path.name, rules)
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

    status = apply_plan(plan, args.strategy)
    created = status["created"] + status["copied"]
    print(f"Готово: создано {created}, уже существовало {status['skipped']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
