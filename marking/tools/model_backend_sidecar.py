"""Python 3.10 sidecar для RF-DETR и официальных RT-DETR.

Протокол stdout: каждая машинная запись начинается с ``MARKING_EVENT `` и
содержит JSON-массив, совместимый с очередями GUI marking.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dataset_statistics import generate_dataset_statistics_report
from model_dataset_bridge import load_yolo_dataset, scenario_to_coco, yolo_split_to_coco
from training_analysis import analyze_training_results, save_training_analysis
from training_metadata import save_training_parameters
from evaluation_report import IMAGE_EXTENSIONS, write_report


EVENT_PREFIX = "MARKING_EVENT "
RT_REPOSITORY = PROJECT_DIR / "third_party" / "RT-DETR"
WEIGHTS_DIR = PROJECT_DIR / "models"
RT_PRETRAINED = {
    "rtdetr_r18": (
        "rtdetr_r18vd_dec3_6x_coco_from_paddle.pth",
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/"
        "rtdetr_r18vd_dec3_6x_coco_from_paddle.pth",
    ),
    "rtdetrv2_s": (
        "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
        "https://github.com/lyuwenyu/storage/releases/download/v0.2/"
        "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
    ),
}


def event(*values):
    print(EVENT_PREFIX + json.dumps(values, ensure_ascii=False), flush=True)


def log(message):
    event("log", str(message))


def gpu_memory_gb():
    try:
        import torch
        return float(torch.cuda.memory_reserved() / (1024 ** 3)) if torch.cuda.is_available() else 0.0
    except Exception:
        return 0.0


def unique_run_dir(profile_id: str, dataset_yaml: str | Path) -> Path:
    dataset_name = Path(dataset_yaml).resolve().parent.name or "dataset"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = PROJECT_DIR / "runs" / profile_id / dataset_name / f"train_{stamp}"
    result.mkdir(parents=True, exist_ok=False)
    return result


def training_parameters(config, run_dir, status, started_at, completed_at=None):
    _yaml, root, _data, _names = load_yolo_dataset(
        config["dataset_yaml"], config.get("dataset_root")
    )
    requested = {
        "model_profile_id": config["profile_id"],
        "model_display_name": config["model_display_name"],
        "model_paradigm": config["model_paradigm"],
        "backend_id": config["backend_id"],
        "model_path": config.get("model_path") or "автоматические предобученные веса",
        "dataset_root": str(root),
        "dataset_yaml": str(Path(config["dataset_yaml"]).resolve()),
        "epochs": int(config["epochs"]),
        "imgsz": int(config["imgsz"]),
        "batch": int(config["batch"]),
        "patience": int(config.get("patience", 0)),
        "gpu": bool(config.get("gpu")),
        "device": "cuda" if config.get("gpu") else "cpu",
        "workers": 0,
        "resume": bool(config.get("resume")),
    }
    return save_training_parameters(
        run_dir, requested, status=status, started_at=started_at, completed_at=completed_at
    )


def finalize_training(config, run_dir, started_at):
    parameters_path = training_parameters(
        config, run_dir, "завершено", started_at,
        datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    dataset_report = ""
    try:
        _stats, path = generate_dataset_statistics_report(
            config.get("dataset_root") or Path(config["dataset_yaml"]).parent,
            output_path=run_dir / "Статистика_датасета.xlsx",
            yaml_path=config["dataset_yaml"],
        )
        dataset_report = str(path)
        log(f"Статистика датасета сохранена: {path}")
    except Exception as exc:
        log(f"Статистику датасета сохранить не удалось: {exc}")

    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        try:
            analysis = analyze_training_results(results_csv)
        except Exception as exc:
            log(f"Loss-анализ недоступен ({exc}); используется динамика validation mAP.")
            analysis = analyze_map_history(run_dir / "log.txt")
        analysis["report_path"] = str(save_training_analysis(run_dir, analysis))
        analysis["run_dir"] = str(run_dir)
        analysis["training_parameters_path"] = str(parameters_path)
        analysis["dataset_report_path"] = dataset_report
        event("training_analysis", analysis)
    log(f"Результаты обучения: {run_dir}")


def analyze_map_history(log_path: Path):
    """Честный fallback для backend-ов, не вычисляющих validation loss."""
    history = []
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            values = row.get("ema_test_coco_eval_bbox") or row.get("test_coco_eval_bbox")
            if values and isinstance(values, list):
                history.append((int(row.get("epoch", len(history))) + 1, float(values[0])))
    if len(history) < 8:
        return {
            "verdict": "insufficient",
            "title": "Недостаточно данных для оценки",
            "summary": (
                "Backend не публикует validation loss; оценка выполняется по validation mAP. "
                f"Пока завершено эпох: {len(history)}."
            ),
            "recommendation": "Продолжите обучение хотя бы до 8 завершенных эпох.",
            "epochs": len(history),
            "details": ["Метод: динамика COCO mAP@0.5:0.95 на validation."],
        }
    epochs = [item[0] for item in history]
    values = [item[1] for item in history]
    window = min(20, max(5, len(values) // 5))
    recent = values[-window:]
    third = max(2, len(recent) // 3)
    start = sum(recent[:third]) / third
    end = sum(recent[-third:]) / third
    delta = end - start
    best = max(values)
    best_epoch = epochs[values.index(best)]
    since_best = epochs[-1] - best_epoch
    if delta >= 0.01:
        verdict = "continue"
        title = "Обучение можно продолжить"
        summary = "Validation mAP на последних эпохах продолжает заметно расти."
        recommendation = "Добавьте 10–20% эпох и сохраните раннюю остановку."
    elif delta <= -0.01 and since_best >= max(3, window // 2):
        verdict = "overfitting"
        title = "Качество validation ухудшается"
        summary = "Validation mAP снизился относительно лучшей эпохи."
        recommendation = f"Используйте checkpoint лучшей эпохи ({best_epoch}); продолжение не рекомендуется."
    else:
        verdict = "converged"
        title = "Validation mAP вышел на плато"
        summary = "На последних эпохах изменение validation mAP невелико."
        recommendation = f"Переходите к тестированию checkpoint лучшей эпохи ({best_epoch})."
    return {
        "verdict": verdict,
        "title": title,
        "summary": summary,
        "recommendation": recommendation,
        "epochs": len(history),
        "best_map_epoch": best_epoch,
        "details": [
            "Метод: динамика COCO mAP@0.5:0.95 на validation (validation loss backend не публикует).",
            f"Последнее сглаженное mAP: {end:.4f}",
            f"Изменение за окно из {window} эпох: {delta:+.4f}",
            f"Лучшее mAP: {best:.4f} на эпохе {best_epoch}",
        ],
    }


def write_results_csv_from_json_log(log_path: Path, output_path: Path):
    if not log_path.is_file():
        return
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        epoch = data.get("epoch")
        if epoch is None:
            continue
        def first(*keys, default=None):
            for key in keys:
                value = data.get(key)
                if isinstance(value, (int, float)):
                    return value
            return default
        train_giou = first("train_loss_giou", "train_loss_giou_unscaled")
        train_l1 = first("train_loss_bbox", "train_loss_bbox_unscaled")
        val_giou = first("test_loss_giou", "test_loss_giou_unscaled")
        val_l1 = first("test_loss_bbox", "test_loss_bbox_unscaled")
        coco = data.get("test_coco_eval_bbox") or data.get("ema_test_coco_eval_bbox") or []
        rows.append({
            "epoch": int(epoch) + 1,
            "train/giou_loss": train_giou,
            "train/l1_loss": train_l1,
            "val/giou_loss": val_giou,
            "val/l1_loss": val_l1,
            "metrics/mAP50-95(B)": coco[0] if len(coco) > 0 else None,
            "metrics/mAP50(B)": coco[1] if len(coco) > 1 else None,
        })
    if not rows:
        return
    fields = list(rows[0])
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_rfdetr(config):
    import torch
    from rfdetr import RFDETRNano

    run_dir = unique_run_dir(config["profile_id"], config["dataset_yaml"])
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    training_parameters(config, run_dir, "выполняется", started_at)
    _yaml, root, _data, names = load_yolo_dataset(
        config["dataset_yaml"], config.get("dataset_root")
    )
    (run_dir / "model_classes.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_split = yolo_split_to_coco(
        config["dataset_yaml"], "train", run_dir / "dataset_bridge" / "train.json", root
    )
    epochs, batch = int(config["epochs"]), int(config["batch"])
    effective_batch = max(1, batch * 4)
    steps_per_epoch = max(1, math.ceil(train_split.image_count / effective_batch))
    model_path = str(config.get("model_path") or "").strip()
    model_kwargs = {
        "device": "cuda" if config.get("gpu") and torch.cuda.is_available() else "cpu",
        "resolution": int(config.get("imgsz") or 384),
    }
    if model_path and Path(model_path).is_file():
        model_kwargs["pretrain_weights"] = model_path
    log("Загрузка RF-DETR-Nano и предобученных весов…")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    previous_directory = Path.cwd()
    try:
        # RF-DETR сохраняет стандартные веса относительно cwd. Держим большие
        # файлы в models/, а не рядом с исходным кодом.
        os.chdir(WEIGHTS_DIR)
        model = RFDETRNano(**model_kwargs)
    finally:
        os.chdir(previous_directory)

    last = {"payload": None}
    def on_batch(info):
        epoch_index = int(info.get("epoch", 0))
        global_step = int(info.get("step", 0))
        step = global_step - epoch_index * steps_per_epoch + 1
        step = max(1, min(steps_per_epoch, step))
        percent_epoch = int(step / steps_per_epoch * 100)
        overall = int((epoch_index + step / steps_per_epoch) / max(1, epochs) * 100)
        payload = ("train_info", epoch_index + 1, epochs, percent_epoch, overall,
                   gpu_memory_gb(), model_kwargs["device"] == "cuda")
        if payload != last["payload"]:
            last["payload"] = payload
            event(*payload)

    def on_epoch(stats):
        epoch = int(stats.get("epoch", 0)) + 1
        log(f"RF-DETR-Nano: завершена эпоха {epoch}/{epochs}")

    model.callbacks["on_train_batch_start"].append(on_batch)
    model.callbacks["on_fit_epoch_end"].append(on_epoch)
    resume_path = model_path if config.get("resume") and Path(model_path).is_file() else None
    model.train(
        dataset_dir=str(root),
        dataset_file="yolo",
        output_dir=str(run_dir),
        epochs=epochs,
        batch_size=batch,
        device=model_kwargs["device"],
        num_workers=0,
        resume=resume_path,
        early_stopping=int(config.get("patience", 0)) > 0,
        early_stopping_patience=max(1, int(config.get("patience", 30))),
        progress_bar=False,
        tensorboard=True,
        class_names=names,
    )
    write_results_csv_from_json_log(run_dir / "log.txt", run_dir / "results.csv")
    finalize_training(config, run_dir, started_at)


def ensure_pretrained(profile_id: str, chosen_path: str) -> Path:
    if chosen_path and Path(chosen_path).is_file():
        return Path(chosen_path).resolve()
    name, url = RT_PRETRAINED[profile_id]
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = WEIGHTS_DIR / name
    # Не принимаем оборванную загрузку за готовый checkpoint. Официальные
    # файлы обеих моделей значительно больше 5 МБ.
    if destination.is_file() and destination.stat().st_size >= 5 * 1024 * 1024:
        return destination
    if destination.exists():
        destination.unlink()
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    log(f"Скачивание официальных предобученных весов {name} (один раз)…")
    try:
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size < 5 * 1024 * 1024:
            raise RuntimeError("Скачанный checkpoint имеет некорректный размер.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def rt_layout(profile_id: str):
    if profile_id == "rtdetr_r18":
        root = RT_REPOSITORY / "rtdetr_pytorch"
        base = root / "configs" / "rtdetr" / "rtdetr_r18vd_6x_coco.yml"
        return root, base, False
    root = RT_REPOSITORY / "rtdetrv2_pytorch"
    base = root / "configs" / "rtdetrv2" / "rtdetrv2_r18vd_120e_coco.yml"
    return root, base, True


def build_rt_config(config, run_dir: Path):
    root, base, is_v2 = rt_layout(config["profile_id"])
    _yaml, dataset_root, _data, names = load_yolo_dataset(
        config["dataset_yaml"], config.get("dataset_root")
    )
    bridge = run_dir / "dataset_bridge"
    # Официальный RT-DETR при remap_mscoco_category=False ожидает непрерывные
    # метки 0..N-1, хотя обычный COCO чаще использует категории с единицы.
    train = yolo_split_to_coco(
        config["dataset_yaml"], "train", bridge / "train.json",
        dataset_root, category_offset=0,
    )
    valid = yolo_split_to_coco(
        config["dataset_yaml"], "val", bridge / "valid.json",
        dataset_root, category_offset=0,
    )
    relative_base = os.path.relpath(base, run_dir).replace("\\", "/")
    batch_key = "total_batch_size" if is_v2 else "batch_size"
    image_size = int(config["imgsz"])
    if is_v2:
        train_ops = [
            {"type": "RandomPhotometricDistort", "p": 0.5},
            {"type": "RandomZoomOut", "fill": 0},
            {"type": "RandomIoUCrop", "p": 0.8},
            {"type": "SanitizeBoundingBoxes", "min_size": 1},
            {"type": "RandomHorizontalFlip"},
            {"type": "Resize", "size": [image_size, image_size]},
            {"type": "SanitizeBoundingBoxes", "min_size": 1},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
            {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
        ]
        val_ops = [
            {"type": "Resize", "size": [image_size, image_size]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        ]
    else:
        train_ops = [
            {"type": "RandomPhotometricDistort", "p": 0.5},
            {"type": "RandomZoomOut", "fill": 0},
            {"type": "RandomIoUCrop", "p": 0.8},
            {"type": "SanitizeBoundingBox", "min_size": 1},
            {"type": "RandomHorizontalFlip"},
            {"type": "Resize", "size": [image_size, image_size]},
            {"type": "ToImageTensor"},
            {"type": "ConvertDtype"},
            {"type": "SanitizeBoundingBox", "min_size": 1},
            {"type": "ConvertBox", "out_fmt": "cxcywh", "normalize": True},
        ]
        val_ops = [
            {"type": "Resize", "size": [image_size, image_size]},
            {"type": "ToImageTensor"},
            {"type": "ConvertDtype"},
        ]
    doc = {
        "__include__": [relative_base],
        "output_dir": str(run_dir).replace("\\", "/"),
        "epoches": int(config["epochs"]),
        "device": "cuda" if config.get("gpu") else "cpu",
        "num_classes": len(names),
        "eval_spatial_size": [image_size, image_size],
        "remap_mscoco_category": False,
        "train_dataloader": {
            batch_key: int(config["batch"]),
            "num_workers": 0,
            "dataset": {
                "img_folder": str(train.images_dir).replace("\\", "/"),
                "ann_file": str(train.annotations).replace("\\", "/"),
                "transforms": {"ops": train_ops},
            },
        },
        "val_dataloader": {
            batch_key: max(1, int(config["batch"])),
            "num_workers": 0,
            "dataset": {
                "img_folder": str(valid.images_dir).replace("\\", "/"),
                "ann_file": str(valid.annotations).replace("\\", "/"),
                "transforms": {"ops": val_ops},
            },
        },
    }
    # В RT-DETR v1 этот параметр не является общим: оба компонента хранят
    # собственную заранее рассчитанную positional embedding. Если оставить
    # стандартные 640 при Resize=768, обучение проходит, но первая валидация
    # падает из-за 24x24 против 20x20 позиций.
    if is_v2:
        doc["HybridEncoder"] = {"eval_spatial_size": [image_size, image_size]}
        doc["RTDETRTransformerv2"] = {
            "eval_spatial_size": [image_size, image_size]
        }
    else:
        doc["HybridEncoder"] = {"eval_spatial_size": [image_size, image_size]}
        doc["RTDETRTransformer"] = {
            "eval_spatial_size": [image_size, image_size]
        }
    path = run_dir / "marking_rtdetr_config.yml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return root, path, train


RT_PROGRESS = re.compile(
    r"Epoch:\s*\[\s*(?P<epoch>\d+)(?:/(?P<epochs>\d+))?\s*\]"
    r".*?\[\s*(?P<batch>\d+)\s*/\s*(?P<total>\d+)\s*\]"
)
RT_GPU_MEMORY = re.compile(r"max mem:\s*(?P<megabytes>\d+(?:\.\d+)?)")


def train_rtdetr(config):
    run_dir = unique_run_dir(config["profile_id"], config["dataset_yaml"])
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    training_parameters(config, run_dir, "выполняется", started_at)
    _yaml, _dataset_root, _data, names = load_yolo_dataset(
        config["dataset_yaml"], config.get("dataset_root")
    )
    (run_dir / "model_classes.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    event("training_status", "Подготовка разметки train и valid для RT-DETR…")
    root, custom_config, _train = build_rt_config(config, run_dir)
    event("training_status", "Разметка подготовлена. Загрузка начальных весов…")
    model_path = ensure_pretrained(config["profile_id"], config.get("model_path", ""))
    command = [sys.executable, str(root / "tools" / "train.py"), "-c", str(custom_config)]
    if config.get("resume"):
        command += ["-r", str(model_path)]
    else:
        command += ["-t", str(model_path)]
    if config.get("gpu"):
        command += (["--use-amp"] if config["profile_id"] == "rtdetrv2_s" else ["--amp"])
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    log(f"Запуск {config['model_display_name']}; результаты: {run_dir}")
    event("training_status", "Запуск RT-DETR и загрузка модели в GPU…")
    process = subprocess.Popen(
        command, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, env=environment,
    )
    epochs = int(config["epochs"])
    assert process.stdout is not None
    log_path = run_dir / "log.txt"
    recent_lines = []
    with log_path.open("w", encoding="utf-8") as log_stream:
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            log_stream.write(raw_line)
            log_stream.flush()
            if line:
                recent_lines.append(line)
                del recent_lines[:-30]
                log(line)
            match = RT_PROGRESS.search(line)
            if match:
                epoch_raw = int(match.group("epoch"))
                # v2 печатает 1-based epoch/total, v1 обычно 0-based epoch.
                epoch = epoch_raw if match.group("epochs") else epoch_raw + 1
                batch_index, total = int(match.group("batch")) + 1, int(match.group("total"))
                percent_epoch = min(100, int(batch_index / max(1, total) * 100))
                overall = min(100, int(((epoch - 1) + batch_index / max(1, total)) / epochs * 100))
                memory_match = RT_GPU_MEMORY.search(line)
                gpu_memory = (
                    float(memory_match.group("megabytes")) / 1024
                    if memory_match else 0.0
                )
                event(
                    "train_info", epoch, epochs, percent_epoch, overall,
                    gpu_memory, bool(config.get("gpu")),
                )
    code = process.wait()
    if code != 0:
        training_parameters(
            config, run_dir, "ошибка", started_at,
            datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        cause = next(
            (
                line.strip()
                for line in reversed(recent_lines)
                if re.match(
                    r"^(?:RuntimeError|ValueError|OSError|MemoryError|"
                    r"FileNotFoundError|ImportError|AssertionError):",
                    line.strip(),
                )
            ),
            "Смотрите последние строки журнала.",
        )
        raise RuntimeError(
            f"Официальный backend RT-DETR завершился с кодом {code}.\n"
            f"Причина: {cause}\n"
            f"Журнал: {log_path}"
        )
    write_results_csv_from_json_log(run_dir / "log.txt", run_dir / "results.csv")
    finalize_training(config, run_dir, started_at)


def train(config):
    if config["backend_id"] == "rfdetr-nano":
        train_rfdetr(config)
    else:
        train_rtdetr(config)


def coco_metrics(annotation_path: Path, predictions: list[dict], class_names: list[str]):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = COCO(str(annotation_path))
    if predictions:
        detections = ground_truth.loadRes(predictions)
    else:
        # pycocotools.loadRes не во всех версиях принимает пустой список.
        empty = {
            "images": ground_truth.dataset["images"],
            "categories": ground_truth.dataset["categories"],
            "annotations": [],
        }
        detections = COCO()
        detections.dataset = empty
        detections.createIndex()
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.maxDets = [1, 100, 500]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    precision = evaluator.eval["precision"]  # IoU, recall, class, area, maxDets
    recall_axis = np.asarray(evaluator.params.recThrs, dtype=float)
    rows, curves = [], {}
    class_curves = []
    for class_index, class_name in enumerate(class_names):
        all_iou = precision[:, :, class_index, 0, -1]
        at_50 = precision[0, :, class_index, 0, -1]
        valid_all = all_iou[all_iou >= 0]
        valid_50 = at_50[at_50 >= 0]
        curve = np.where(at_50 >= 0, at_50, np.nan)
        if valid_50.size:
            f1 = 2 * curve * recall_axis / np.maximum(curve + recall_axis, 1e-12)
            position = int(np.nanargmax(f1))
            point_precision = float(curve[position])
            point_recall = float(recall_axis[position])
            class_curves.append(curve)
        else:
            point_precision = point_recall = 0.0
        rows.append({
            "class_id": class_index,
            "Класс": class_name,
            "Исходное имя класса": class_name,
            "mAP@0.5": float(valid_50.mean()) if valid_50.size else 0.0,
            "mAP@0.5:0.95": float(valid_all.mean()) if valid_all.size else 0.0,
            "Precision": point_precision,
            "Recall": point_recall,
        })
        curves[class_name] = (recall_axis, curve)
    mean_curve = (
        np.nanmean(np.vstack(class_curves), axis=0)
        if class_curves else np.zeros_like(recall_axis)
    )
    curves["Все классы"] = (recall_axis, mean_curve)
    f1 = 2 * mean_curve * recall_axis / np.maximum(mean_curve + recall_axis, 1e-12)
    best = int(np.nanargmax(f1)) if np.isfinite(f1).any() else 0
    return {
        "precision": float(mean_curve[best]) if mean_curve.size else 0.0,
        "recall": float(recall_axis[best]) if recall_axis.size else 0.0,
        "map50": float(evaluator.stats[1]),
        "map5095": float(evaluator.stats[0]),
        "classes": rows,
        "curves": curves,
    }


class RFDetrPredictor:
    def __init__(self, checkpoint, imgsz, device):
        from rfdetr import RFDETRNano
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        previous_directory = Path.cwd()
        try:
            os.chdir(WEIGHTS_DIR)
            self.model = RFDETRNano(
                pretrain_weights=str(checkpoint), resolution=int(imgsz), device=device
            )
        finally:
            os.chdir(previous_directory)

    def __call__(self, image_path):
        detections = self.model.predict(str(image_path), threshold=0.001)
        return [
            (int(class_id), [float(value) for value in box], float(score))
            for class_id, box, score in zip(
                detections.class_id, detections.xyxy, detections.confidence
            )
        ]


class RTDetrPredictor:
    def __init__(self, profile_id, checkpoint, imgsz, device):
        import torch
        import torchvision.transforms as transforms

        root, base, _is_v2 = rt_layout(profile_id)
        sys.path.insert(0, str(root))
        from src.core import YAMLConfig

        checkpoint = Path(checkpoint).resolve()
        config_path = checkpoint.parent / "marking_rtdetr_config.yml"
        if not config_path.is_file():
            raise ValueError(
                "Рядом с checkpoint RT-DETR не найден marking_rtdetr_config.yml. "
                "Выберите best.pth/checkpoint.pth из папки обучения marking."
            )
        cfg = YAMLConfig(str(config_path))
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if "ema" in state and isinstance(state["ema"], dict):
            weights = state["ema"].get("module", state["ema"])
        else:
            weights = state.get("model", state)
        cfg.model.load_state_dict(weights, strict=True)
        self.torch = torch
        self.device = torch.device(device)
        self.model = cfg.model.to(self.device).eval()
        self.postprocessor = cfg.postprocessor.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((int(imgsz), int(imgsz))),
            transforms.ToTensor(),
        ])

    def __call__(self, image_path):
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        tensor = self.transform(image)[None].to(self.device)
        original = self.torch.tensor([[width, height]], device=self.device)
        with self.torch.no_grad():
            output = self.postprocessor(self.model(tensor), original)[0]
        keep = output["scores"] >= 0.001
        labels = output["labels"][keep].detach().cpu().tolist()
        boxes = output["boxes"][keep].detach().cpu().tolist()
        scores = output["scores"][keep].detach().cpu().tolist()
        return [(int(label), list(map(float, box)), float(score))
                for label, box, score in zip(labels, boxes, scores)]


def evaluate_scenario(
    predictor, name, scenario_dir, class_names, model_to_dataset,
    work_dir, index, total,
):
    scenario_dir = Path(scenario_dir).resolve()
    split = scenario_to_coco(
        scenario_dir, class_names, work_dir / f"scenario_{index}_ground_truth.json"
    )
    coco_data = json.loads(split.annotations.read_text(encoding="utf-8"))
    ids_by_name = {item["file_name"]: item["id"] for item in coco_data["images"]}
    image_paths = sorted(
        (scenario_dir / "images").rglob("*"), key=lambda path: str(path).casefold()
    )
    image_paths = [path for path in image_paths
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    predictions = []
    started = time.perf_counter()
    for image_index, image_path in enumerate(image_paths, start=1):
        relative = image_path.relative_to(scenario_dir / "images").as_posix()
        image_id = ids_by_name[relative]
        for model_class_id, xyxy, score in predictor(image_path):
            if model_class_id not in model_to_dataset:
                continue
            class_id = model_to_dataset[model_class_id]
            x1, y1, x2, y2 = xyxy
            predictions.append({
                "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": score,
            })
        overall = int(((index - 1) + image_index / max(1, len(image_paths))) / total * 100)
        event("progress", overall)
    elapsed = time.perf_counter() - started
    prediction_path = work_dir / f"scenario_{index}_predictions.json"
    prediction_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    metrics = coco_metrics(split.annotations, predictions, class_names)
    inference_ms = elapsed * 1000 / max(1, len(image_paths))
    metrics.update({
        "scenario": name,
        "path": str(scenario_dir),
        "images": len(image_paths),
        "inference_ms": inference_ms,
        "fps": 1000.0 / inference_ms if inference_ms > 0 else 0.0,
    })
    return metrics


def load_model_classes(model_path: Path) -> list[str]:
    candidates = (
        model_path.parent / "model_classes.json",
        model_path.parent.parent / "model_classes.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            names = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(names, list) and names and all(isinstance(name, str) for name in names):
                return names
    raise ValueError(
        "Рядом с checkpoint не найден model_classes.json. Для корректного "
        "соответствия ID выберите checkpoint из папки обучения marking."
    )


def test(config):
    import torch

    model_path = Path(config["model_path"]).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint не найден: {model_path}")
    report_dir = Path(config["report_dir"]).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    standard_dir = report_dir / f"Стандартный_отчет_{config['profile_id']}"
    standard_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("scenario_*_ground_truth.json", "scenario_*_predictions.json", "metrics.json"):
        for old_file in standard_dir.glob(pattern):
            old_file.unlink(missing_ok=True)
    class_names = list(config["class_names"])
    model_class_names = load_model_classes(model_path)
    dataset_ids = {name.casefold(): index for index, name in enumerate(class_names)}
    model_keys = [name.casefold() for name in model_class_names]
    if set(dataset_ids) != set(model_keys):
        raise ValueError(
            "Классы checkpoint и выбранного data.yaml не совпадают. "
            f"Checkpoint: {model_class_names}; data.yaml: {class_names}."
        )
    model_to_dataset = {
        model_id: dataset_ids[name.casefold()]
        for model_id, name in enumerate(model_class_names)
    }
    device = "cuda" if config.get("gpu") and torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    log(f"Устройство: {device_name}")
    if config["backend_id"] == "rfdetr-nano":
        predictor = RFDetrPredictor(model_path, config["imgsz"], device)
    else:
        predictor = RTDetrPredictor(
            config["profile_id"], model_path, config["imgsz"], device
        )
    scenarios = config["scenarios"]
    results = []
    for index, pair in enumerate(scenarios, start=1):
        name, path = pair
        event("scenario_info", name, index, len(scenarios))
        log(f"Тестирование: {name} ({index}/{len(scenarios)})")
        results.append(evaluate_scenario(
            predictor, name, path, class_names, model_to_dataset,
            standard_dir, index, len(scenarios)
        ))
    summary_json = standard_dir / "metrics.json"
    summary_json.write_text(
        json.dumps(
            [{key: value for key, value in result.items() if key not in {"curves"}}
             for result in results],
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    report_path = write_report(
        report_dir=report_dir,
        model_path=str(model_path),
        imgsz=int(config["imgsz"]),
        device_name=device_name,
        scenario_results=results,
        dataset_class_names=class_names,
        model_class_names=model_class_names,
        model_profile_name=config["model_display_name"],
    )
    event("report", str(report_path))
    log(f"Отчет сформирован: {report_path}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: model_backend_sidecar.py <train|test> <config.json>")
    action, config_path = sys.argv[1:]
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    try:
        if action == "train":
            train(config)
        elif action == "test":
            test(config)
        else:
            raise ValueError(f"Неизвестная команда: {action}")
    except Exception as exc:
        event("error", str(exc), traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
