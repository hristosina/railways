import queue as queue_module
import tempfile
import traceback
from pathlib import Path

import multiprocessing as mp
import torch
import ultralytics
import yaml
from PyQt5.QtCore import QObject, pyqtSignal
from ultralytics import YOLO

from evaluation_report import (
    aligned_scenario_view,
    class_id_mapping,
    count_images,
    copy_standard_ultralytics_report,
    evaluation_progress_percent,
    extract_evaluation,
    normalize_class_names,
    publish_standard_ultralytics_report,
    write_report,
)


def create_temp_data_yaml(scenario_dir, class_names, output_dir):
    """Создает data.yaml в автоматически удаляемой временной папке."""
    yaml_path = Path(output_dir) / f"data_{scenario_dir.name}.yaml"
    data = {
        "path": str(scenario_dir.resolve()),
        "train": "images",
        "val": "images",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
    return yaml_path


def yolo_test_process(queue, model_path, scenarios, report_dir, class_names, imgsz, gpu):
    success, report_path = False, ""
    try:
        if torch.cuda.is_available() and gpu:
            device = 0
            device_name = torch.cuda.get_device_name(device)
            queue.put(("log", f"GPU: {device_name}"))
        else:
            device, device_name = "cpu", "CPU"
            queue.put(("log", "Используется CPU"))

        if not scenarios:
            raise ValueError("Не задано ни одного тестового сценария.")
        scenarios = [(str(name), Path(path).resolve()) for name, path in scenarios]
        report_dir = Path(report_dir).resolve()
        standard_report_dir = report_dir / "Стандартный_отчет_Ultralytics"
        queue.put(("log", f"Найдено сценариев: {len(scenarios)}"))
        model = YOLO(model_path)
        model_class_names = normalize_class_names(model.names)
        mapping = class_id_mapping(class_names, model_class_names)
        if any(source_id != target_id for source_id, target_id in mapping.items()):
            queue.put((
                "log",
                "Порядок классов data.yaml отличается от порядка классов модели. "
                "Для тестирования разметка будет временно перенумерована.\n"
                f"data.yaml: {class_names}\nмодель: {model_class_names}",
            ))
        scenario_results = []
        progress_state = {"scenario_index": 0, "last_percent": -1}

        def report_batch_progress(validator):
            """Получает прогресс непосредственно после каждого val-пакета."""
            percent = evaluation_progress_percent(
                scenario_index=progress_state["scenario_index"],
                total_scenarios=len(scenarios),
                batch_index=validator.batch_i,
                total_batches=len(validator.dataloader),
            )
            # Очередь обновляется только при изменении целого процента, чтобы
            # сотни пакетов не перегружали событийный цикл Qt.
            if percent != progress_state["last_percent"]:
                progress_state["last_percent"] = percent
                queue.put(("progress", percent))

        model.add_callback("on_val_batch_end", report_batch_progress)

        # Даже при save=False Ultralytics создает служебный save_dir. Направляем
        # его во временную папку, которая удалится после завершения проверки.
        with tempfile.TemporaryDirectory(prefix="marking_evaluation_") as temp_dir:
            temp_root = Path(temp_dir)
            standard_staging_dir = temp_root / "standard_report"
            for index, (shown_name, scenario_dir) in enumerate(scenarios, start=1):
                progress_state["scenario_index"] = index - 1
                queue.put(("scenario_info", shown_name, index, len(scenarios)))
                queue.put(("log", f"Тестирование: {shown_name} ({index}/{len(scenarios)})"))
                with aligned_scenario_view(scenario_dir, class_names, model_class_names) as (validation_dir, _mapping):
                    data_yaml = create_temp_data_yaml(validation_dir, model_class_names, temp_root)
                    metrics = model.val(
                        data=str(data_yaml),
                        imgsz=imgsz,
                        device=device,
                        conf=0.001,
                        iou=0.5,
                        save=False,
                        save_json=False,
                        plots=True,
                        project=str(temp_root),
                        name=f"scenario_{index}",
                        exist_ok=True,
                        verbose=False,
                    )
                validator_output = temp_root / f"scenario_{index}"
                copy_standard_ultralytics_report(
                    validator_output, standard_staging_dir, shown_name, index
                )
                class_rows, curves = extract_evaluation(metrics, model_class_names)
                inference_ms = float(metrics.speed.get("inference", 0.0))
                scenario_results.append({
                    "scenario": shown_name,
                    "path": str(scenario_dir),
                    "images": count_images(scenario_dir / "images"),
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                    "map50": float(metrics.box.map50),
                    "map5095": float(metrics.box.map),
                    "inference_ms": inference_ms,
                    "fps": 1000.0 / inference_ms if inference_ms > 0 else 0.0,
                    "classes": class_rows,
                    "curves": curves,
                })
                queue.put(("progress", int(index / len(scenarios) * 100)))
            publish_standard_ultralytics_report(standard_staging_dir, standard_report_dir)

        report_path = str(write_report(
            report_dir=report_dir,
            model_path=model_path,
            imgsz=imgsz,
            device_name=device_name,
            scenario_results=scenario_results,
            dataset_class_names=class_names,
            model_class_names=model_class_names,
            ultralytics_version=ultralytics.__version__,
        ))
        queue.put(("report", report_path))
        queue.put(("log", f"Отчет сформирован: {report_path}"))
        success = True
    except Exception as exc:
        queue.put(("log", f"Ошибка при тестировании:\n{exc}\n{traceback.format_exc()}"))
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        queue.put(("finished", success, report_path))


class YOLOTestWorker(QObject):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    scenario_info = pyqtSignal(str, int, int)
    report_created = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.queue = None

    def start_evaluation(self, model_path, scenarios, report_dir, class_names, imgsz=640, gpu=True):
        self.queue = mp.Queue()
        self.process = mp.Process(
            target=yolo_test_process,
            args=(self.queue, model_path, scenarios, report_dir, class_names, imgsz, gpu),
            daemon=False,
        )
        self.process.start()

    def poll_queue(self):
        """Считывает все готовые сообщения без ненадежного Queue.empty()."""
        if self.queue is None:
            return
        while True:
            try:
                message = self.queue.get_nowait()
            except queue_module.Empty:
                break
            message_type = message[0]
            if message_type == "progress":
                self.progress.emit(message[1])
            elif message_type == "log":
                self.log.emit(message[1])
            elif message_type == "scenario_info":
                _, name, index, total = message
                self.scenario_info.emit(name, index, total)
            elif message_type == "report":
                self.report_created.emit(message[1])
            elif message_type == "finished":
                self.finished.emit(message[1], message[2])
