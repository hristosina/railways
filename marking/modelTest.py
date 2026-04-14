import os
import time
import traceback
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PyQt5.QtCore import QObject, pyqtSignal
import multiprocessing as mp
from ultralytics import YOLO


def create_temp_data_yaml(scenario_dir, class_names, output_root):
    import yaml
    from pathlib import Path

    yaml_path = output_root / f"data_{scenario_dir.name}.yaml"

    data = {
        "path": str(scenario_dir.resolve()),
        "train": "images",
        "val": "images",
        "nc": len(class_names),
        "names": class_names
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    return yaml_path

def yolo_test_process(
    queue,
    model_path,
    test_root,
    class_names,
    imgsz,
    gpu,
    export_pr=True,        # для построения графика в excel
    pr_steps=51            # для построения графика в excel
):
    try:
        # ---------- DEVICE ----------
        if torch.cuda.is_available() and gpu:
            device = 0
            queue.put(("log", f"GPU доступен: {torch.cuda.get_device_name(device)}"))
        else:
            device = "cpu"
            queue.put(("log", "Используется CPU"))

        model = YOLO(model_path)

        test_root = Path(test_root)
        output_root = test_root / "_results"
        output_root.mkdir(exist_ok=True)

        # ---------- ПОИСК СЦЕНАРИЕВ ----------
        scenarios = []

        for path in test_root.rglob("*"):
            if not path.is_dir():
                continue

            images_dir = path / "images"
            labels_dir = path / "labels"

            if not images_dir.exists():
                continue

            if not labels_dir.exists():
                queue.put((
                    "log",
                    f"Пропуск сценария {path}: отсутствует папка labels/"
                ))
                continue

            scenarios.append(path)

        total_scenarios = len(scenarios)
        queue.put(("log", f"Найдено сценариев: {total_scenarios}"))

        summary_rows = []

        # ---------- ОСНОВНОЙ ЦИКЛ ----------
        for idx, scenario_dir in enumerate(scenarios, start=1):
            queue.put((
                "scenario_info",
                scenario_dir.name,
                idx,
                total_scenarios
            ))

            images_dir = scenario_dir / "images"
            scenario_out = output_root / scenario_dir.name
            scenario_out.mkdir(exist_ok=True)

            start_time = time.time()

            data_yaml = create_temp_data_yaml(
                scenario_dir,
                class_names,
                output_root
            )

            metrics = model.val(
                data=str(data_yaml),  # путь к data.yaml для текущего сценария (классы + labels)
                imgsz=imgsz,  # размер входных изображений (например, 640)
                device=device,  # GPU или CPU для инференса
                conf=0.001,  # минимальная уверенность для детекции (очень низкая для теста)
                iou=0.5,  # IoU threshold для подсчёта mAP@0.5
                save=False,  # сохранять картинки с боксами? False — нет
                plots=True,  # строить графики (PR-кривые, confusion matrix)? False — нет
                verbose=False  # выводить подробный лог в консоль? False — нет
            )

            elapsed = time.time() - start_time

            # число изображений в текущем сценарии
            num_images = len(list((scenario_dir / "images").glob("*.*")))

            fps = metrics.speed['inference']  # инференс FPS
            inf_time = (elapsed / num_images) * 1000  # мс на изображение

            # ---------- PR CURVE EXPORT ----------
            if export_pr:
                queue.put(("log", f"PR-sweep для сценария {scenario_dir.name}"))

                import numpy as np

                pr_rows = []
                conf_list = np.linspace(0.0, 1.0, pr_steps)

                for conf in conf_list:
                    pr_metrics = model.val(
                        data=str(data_yaml),
                        imgsz=imgsz,
                        device=device,
                        conf=float(conf),
                        iou=0.5,
                        save=False,
                        plots=False,
                        verbose=False
                    )

                    # ---------- per-class ----------
                    for i, cls_name in enumerate(class_names):
                        pr_rows.append({
                            "confidence": float(conf),
                            "class": cls_name,
                            "precision": float(pr_metrics.box.p[i]),
                            "recall": float(pr_metrics.box.r[i])
                        })

                    # ---------- mean ----------
                    pr_rows.append({
                        "confidence": float(conf),
                        "class": "__mean__",
                        "precision": float(pr_metrics.box.p.mean()),
                        "recall": float(pr_metrics.box.r.mean())
                    })

                pr_dir = scenario_out / "pr_curves"
                pr_dir.mkdir(exist_ok=True)

                df_pr = pd.DataFrame(pr_rows)

                df_pr.to_excel(
                    pr_dir / "PR_curve_all.xlsx",
                    index=False
                )

            # ---------- GLOBAL METRICS ----------
            summary_rows.append({
                "Scenario": scenario_dir.name,
                "mAP@0.5": metrics.box.map50,
                "mAP@0.5:0.95": metrics.box.map,
                "FPS": fps,
                "InferenceTime(ms/img)": inf_time
            })

            # ---------- PER-CLASS METRICS ----------
            rows = []
            for i, cls in enumerate(class_names):
                rows.append({
                    "Class": cls,
                    "mAP@0.5": metrics.box.maps[i],
                    "mAP@0.5:0.95": metrics.box.map,
                    "Precision": metrics.box.p[i],
                    "Recall": metrics.box.r[i]
                })

            pd.DataFrame(rows).to_excel(
                scenario_out / "per_class_metrics.xlsx",
                index=False
            )

            # ---------- PROGRESS ----------
            percent = int(idx / total_scenarios * 100)
            queue.put(("progress", percent))

        # ---------- SUMMARY ----------
        pd.DataFrame(summary_rows).to_excel(
            output_root / "summary.xlsx",
            index=False
        )

        queue.put(("log", "Тестирование завершено"))

    except Exception as e:
        queue.put((
            "log",
            f"Ошибка при тестировании:\n{str(e)}\n{traceback.format_exc()}"
        ))

    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass

        queue.put(("finished", True))


class YOLOTestWorker(QObject):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    scenario_info = pyqtSignal(str, int, int)
    finished = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.queue = None

    def start_evaluation(
        self,
        model_path,
        test_root,
        class_names,
        imgsz=640,
        gpu=True
    ):
        self.queue = mp.Queue()

        self.process = mp.Process(
            target=yolo_test_process,
            args=(
                self.queue,
                model_path,
                test_root,
                class_names,
                imgsz,
                gpu,
                True,  # export_pr
                51  # pr_steps
            ),
            daemon=False
        )

        self.process.start()

    def poll_queue(self):
        """
        Вызывать таймером (QTimer) из UI
        """
        while not self.queue.empty():
            msg = self.queue.get()

            msg_type = msg[0]

            if msg_type == "progress":
                self.progress.emit(msg[1])

            elif msg_type == "log":
                self.log.emit(msg[1])

            elif msg_type == "scenario_info":
                _, name, idx, total = msg
                self.scenario_info.emit(name, idx, total)

            elif msg_type == "finished":
                self.finished.emit(True)