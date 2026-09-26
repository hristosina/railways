"""Utilities for locating YOLO datasets and annotation folders."""

from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "valid", "val", "test"}


def contains_images(path):
    path = Path(path)
    if not path.is_dir():
        return False
    return any(
        item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        for item in path.iterdir()
    )


def normalize_dataset_path(selected_path):
    """Return the most useful dataset root for a user-selected directory.

    A selected ``images``/``labels`` directory is accepted.  If a data.yaml is
    present in an ancestor, its directory wins; otherwise the annotation-set
    directory (the parent containing images and labels) is returned.
    """
    selected = Path(selected_path).expanduser().resolve()
    if not selected.is_dir():
        raise ValueError("Выбранная папка не существует.")

    if (selected / "data.yaml").is_file():
        return selected

    selected_name = selected.name.lower()
    if selected_name in {"images", "labels"}:
        annotation_root = selected.parent
        if annotation_root.name.lower() not in SPLIT_NAMES:
            return annotation_root
        selected = annotation_root

    if selected.name.lower() in SPLIT_NAMES:
        for candidate in selected.parents:
            if (candidate / "data.yaml").is_file():
                return candidate

    if (selected / "images").is_dir() or any(
        (selected / split).is_dir() for split in SPLIT_NAMES
    ):
        return selected

    return Path(selected_path).expanduser().resolve()


def find_dataset_yaml(dataset_path):
    dataset_path = Path(dataset_path).expanduser().resolve()
    for candidate in (dataset_path, *dataset_path.parents):
        yaml_path = candidate / "data.yaml"
        if yaml_path.is_file():
            return yaml_path
    return None


def resolve_annotation_set(folder_path):
    """Resolve a folder to its sibling YOLO images/labels directories."""
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError("Выбранная папка не существует.")

    if folder.name.lower() == "images":
        images_dir = folder
        annotation_root = folder.parent
    elif folder.name.lower() == "labels":
        annotation_root = folder.parent
        images_dir = annotation_root / "images"
    elif (folder / "images").is_dir():
        annotation_root = folder
        images_dir = folder / "images"
    elif contains_images(folder):
        images_dir = folder
        annotation_root = folder.parent
    else:
        raise ValueError(
            "В выбранной папке не найдена папка images с изображениями."
        )

    if not images_dir.is_dir():
        raise ValueError(f"Папка изображений не найдена: {images_dir}")

    return annotation_root, images_dir, annotation_root / "labels"


def find_annotation_sets(dataset_path):
    """Find folders that directly contain an ``images`` directory."""
    root = Path(dataset_path).expanduser().resolve()
    if not root.is_dir():
        return []

    result = []
    for candidate in (root, *sorted((p for p in root.rglob("*") if p.is_dir()), key=str)):
        if ".marking_trash" in candidate.parts:
            continue
        images_dir = candidate / "images"
        if images_dir.is_dir() and contains_images(images_dir):
            result.append(candidate)
    return result


def move_annotation_pair_to_trash(image_path, label_path, trash_root=None):
    """Убирает изображение и его YOLO-разметку из split в локальную корзину."""
    image_path = Path(image_path).expanduser().resolve()
    label_path = Path(label_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    annotation_root = image_path.parent.parent
    trash_root = (
        Path(trash_root).expanduser().resolve()
        if trash_root is not None else annotation_root / ".marking_trash"
    )
    trash_images = trash_root / "images"
    trash_labels = trash_root / "labels"
    trash_images.mkdir(parents=True, exist_ok=True)
    trash_labels.mkdir(parents=True, exist_ok=True)

    suffix_index = 0
    while True:
        suffix = "" if suffix_index == 0 else f"_{suffix_index}"
        destination_image = trash_images / f"{image_path.stem}{suffix}{image_path.suffix}"
        destination_label = trash_labels / f"{image_path.stem}{suffix}.txt"
        if not destination_image.exists() and not destination_label.exists():
            break
        suffix_index += 1

    image_path.replace(destination_image)
    try:
        moved_label = None
        if label_path.is_file():
            label_path.replace(destination_label)
            moved_label = destination_label
    except Exception:
        # Не оставляем split в наполовину измененном состоянии.
        if destination_image.exists() and not image_path.exists():
            destination_image.replace(image_path)
        raise

    return destination_image, moved_label, trash_root
