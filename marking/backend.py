from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal
from ultralytics import YOLO

import os
import shutil
import yaml
import torch
import traceback
import multiprocessing as mp


def yolo_autolabel_process(
    queue,
    model_path,
    root_folder,
    selected_classes,
    conf
):
    try:
        from ultralytics import YOLO
        from pathlib import Path

        model = YOLO(model_path)

        root_folder = Path(root_folder)
        image_folders = list(root_folder.rglob("images"))

        if not image_folders:
            queue.put(("error", "Папок images не найдено!", ""))
            return

        total_images = sum(len(list(f.glob("*.[jp][pn]g"))) for f in image_folders)
        if total_images == 0:
            queue.put(("error", "Изображений не найдено!", ""))
            return

        processed = 0
        last_percent = -1

        for images_folder in image_folders:
            labels_folder = images_folder.parent / "labels"
            labels_folder.mkdir(exist_ok=True, parents=True)

            for img_path in images_folder.glob("*.[jp][pn]g"):
                label_txt = labels_folder / f"{img_path.stem}.txt"

                # читаем старые боксы
                existing_labels = []
                if label_txt.exists():
                    with open(label_txt, "r", encoding="utf-8") as f:
                        for line in f:
                            p = line.split()
                            if len(p) == 5:
                                existing_labels.append({
                                    "cls": int(p[0]),
                                    "x": float(p[1]),
                                    "y": float(p[2]),
                                    "w": float(p[3]),
                                    "h": float(p[4]),
                                })

                results = model.predict(str(img_path), conf=conf, save=False)

                new_labels = []
                for r in results:
                    h, w = r.orig_shape[:2]
                    for box in r.boxes:
                        cls_idx = int(box.cls[0])
                        cls_name = model.names[cls_idx]

                        if selected_classes is None or cls_name in selected_classes:
                            x, y, bw, bh = box.xywh[0]
                            new_labels.append({
                                "cls": cls_idx,
                                "x": float(x / w),
                                "y": float(y / h),
                                "w": float(bw / w),
                                "h": float(bh / h),
                            })

                final_labels = [
                    lbl for lbl in existing_labels
                    if model.names[lbl["cls"]] not in (selected_classes or [])
                ]
                final_labels.extend(new_labels)

                with open(label_txt, "w", encoding="utf-8") as f:
                    for l in final_labels:
                        f.write(
                            f'{l["cls"]} {l["x"]:.6f} {l["y"]:.6f} '
                            f'{l["w"]:.6f} {l["h"]:.6f}\n'
                        )

                processed += 1
                percent = int(processed / total_images * 100)

                if percent != last_percent:
                    queue.put(("progress", percent))
                    last_percent = percent

        queue.put(("log", "Автолейблинг завершен"))
        queue.put(("finished", True))

    except Exception as e:
        queue.put(("error", str(e), traceback.format_exc()))


def yolo_train_process(queue, model_path, dataset_yaml, epochs, imgsz, batch, gpu):
    import torch
    import traceback
    from ultralytics import YOLO

    try:
        if torch.cuda.is_available() and gpu:
            device = 0
            print(f"GPU доступен: {torch.cuda.get_device_name(device)}")
        else:
            device = "cpu"

        model = YOLO(model_path)

        """
        data - файл конфигурации датасета, в нем указаны пути к изображениям,
        список классов (что мы распознаем), количество классов
        train - папка для обучения (содержит картинки, на которых модель учится находить объекты),
        это основная и самая большая часть данных (обычно 70–90%)
        val (valid / validation) — проверка во время обучения.
        Эти данные не участвуют в обучении, только используются для контроля.
        После каждой эпохи модель оценивается на изображениях из valid.
        Это помогает понять, что модель не переобучилась (не заучила train «наизусть»).
        test — финальная проверка (после обучения)
        Когда обучение завершено — запускают inference на test, чтобы измерить результат.
        Это независимые данные, на которых модель не училась и не валидировалась.
        epochs - сколько раз нейросеть посмотрит весь датасет (чем больше, тем лучше)
        10–30 — быстро, но слабее
        50 — хорошее стартовое значение
        100–300 — лучше качество, дольше обучение
        imgsz - размер, до которого изображения уменьшаются перед обучением
        batch - сколько изображений обрабатывается одновременно:
        чем больше batch, тем быстрее обучение, но нужно больше видеопамяти GPU
        device - на каком устройстве обучение
        """

        def on_train_batch_end(trainer):
            epoch = trainer.epoch + 1
            epochs_total = trainer.args.epochs

            # примерный процент
            percent_in_epoch = int((epoch / epochs_total) * 100)

            gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

            queue.put((
                "train_info",
                epoch,
                epochs_total,
                percent_in_epoch,
                gpu_mem
            ))

        model.add_callback("on_train_batch_end", on_train_batch_end)

        cpu_cores = os.cpu_count()  # общее количество логических ядер
        workers = max(1, cpu_cores // 2)  # половина ядер

        model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers
        )

        queue.put(("log", "Обучение завершено!"))

    except Exception as e:
        queue.put((
            "log",
            f"Ошибка при обучении: {str(e)}\n{traceback.format_exc()}"
        ))

    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass

        try:
            queue.put(("finished", True))
        except:
            pass


class YoloWorker(QObject):
    # epoch, epochs_total, percent_in_epoch, gpu_mem
    train_info = pyqtSignal(int, int, int, float)
    progress = pyqtSignal(int)  # процент выполнения
    log = pyqtSignal(str)  # текстовые сообщения
    finished = pyqtSignal(bool)  # сигнал о завершении работы

    def __init__(self, model_path, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.process = None
        self.queue = None

    def start_training(
        self,
        model_path,
        dataset_yaml,
        epochs,
        imgsz,
        batch,
        gpu
    ):
        self.queue = mp.Queue()

        self.process = mp.Process(
            target=yolo_train_process,
            args=(
                self.queue,  # очередь первой
                model_path,  # потом модель
                dataset_yaml,
                epochs,
                imgsz,
                batch,
                gpu
            ),
            daemon=False
        )

        self.process.start()

    def start_autolabeling_process(
            self,
            model_path,
            root_folder,
            selected_classes,
            conf
    ):
        self.queue = mp.Queue()

        self.process = mp.Process(
            target=yolo_autolabel_process,
            args=(
                self.queue,
                model_path,
                root_folder,
                selected_classes,
                conf
            ),
            daemon=False
        )

        self.process.start()

    def train(self, dataset_yaml, epochs=50, imgsz=640, batch=4, gpu=True):
        """
        Обучение модели на указанном датасете.
        Сигналы log и train_info обновляются автоматически.
        """
        try:
            device = 0 if gpu else 'cpu'

            """
            data - файл конфигурации датасета, в нем указаны пути к изображениям,
            список классов (что мы распознаем), количество классов
            train - папка для обучения (содержит картинки, на которых модель учится находить объекты),
            это основная и самая большая часть данных (обычно 70–90%)
            val (valid / validation) — проверка во время обучения.
            Эти данные не участвуют в обучении, только используются для контроля.
            После каждой эпохи модель оценивается на изображениях из valid.
            Это помогает понять, что модель не переобучилась (не заучила train «наизусть»).
            test — финальная проверка (после обучения)
            Когда обучение завершено — запускают inference на test, чтобы измерить результат.
            Это независимые данные, на которых модель не училась и не валидировалась.
            epochs - сколько раз нейросеть посмотрит весь датасет (чем больше, тем лучше)
            10–30 — быстро, но слабее
            50 — хорошее стартовое значение
            100–300 — лучше качество, дольше обучение
            imgsz - размер, до которого изображения уменьшаются перед обучением
            batch - сколько изображений обрабатывается одновременно:
            чем больше batch, тем быстрее обучение, но нужно больше видеопамяти GPU
            device - на каком устройстве обучение: 0 - GPU 1, 1 - GPU 2, если GPU нет — параметр можно удалить
            """

            def on_train_batch_end(trainer):
                epoch = trainer.epoch + 1
                epochs_total = trainer.args.epochs

                # TODO разобраться, как получать процент в эпохе
                percent_in_epoch = 0

                # TODO разобраться, как получать GPU MEM, сейчас некорректно
                # gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
                gpu_mem = 0

                self.train_info.emit(
                    epoch,
                    epochs_total,
                    percent_in_epoch,
                    gpu_mem
                )

            # подключаем callback ultralytics
            self.model.add_callback("on_train_batch_end", on_train_batch_end)

            self.model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device
            )

            self.log.emit("Обучение завершено!")

        except Exception as e:
            self.log.emit(f"Ошибка при обучении: {str(e)}\n{traceback.format_exc()}")

        finally:
            self.finished.emit(True)

    def predict_and_save_labels_recursive(self, root_folder, selected_classes=None, conf=0.4):
        """
        Рекурсивный автолейблинг:
        - Ищем все папки с именем "images" в дереве root_folder
        - Создаем рядом labels, если их нет
        - Для каждой картинки заменяем/добавляем bbox для выбранных классов
        """

        try:
            root_folder = Path(root_folder)
            image_folders = list(root_folder.rglob("images"))

            if not image_folders:
                self.log.emit("Папок images не найдено!")
                return

            # считаем общее количество изображений для прогресса
            total_images = sum(len(list(f.glob("*.[jp][pn]g"))) for f in image_folders)
            if total_images == 0:
                self.log.emit("Изображений не найдено!")
                return

            processed = 0
            last_perc = -1  # для редкого обновления прогресса

            for images_folder in image_folders:
                labels_folder = images_folder.parent / "labels"
                labels_folder.mkdir(exist_ok=True, parents=True)

                for img_path in images_folder.glob("*.[jp][pn]g"):
                    label_txt = labels_folder / (img_path.stem + ".txt")

                    # читаем существующие bbox
                    existing_labels = []
                    if label_txt.exists():
                        with open(label_txt, "r", encoding="utf-8") as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    continue
                                cls, x, y, w, h = parts
                                existing_labels.append({
                                    "cls": int(cls),
                                    "x": float(x),
                                    "y": float(y),
                                    "w": float(w),
                                    "h": float(h)
                                })

                    # прогон модели
                    results = self.model.predict(source=str(img_path), conf=conf, save=False)

                    new_labels = []
                    for r in results:
                        img_h, img_w = r.orig_shape[:2]  # реальные размеры изображения
                        for box in r.boxes:
                            cls_idx = int(box.cls[0])
                            cls_name = self.model.names[cls_idx]

                            if selected_classes is None or cls_name in selected_classes:
                                x_center, y_center, width, height = box.xywh[0]

                                # нормализация координат
                                x_center /= img_w
                                y_center /= img_h
                                width /= img_w
                                height /= img_h

                                # ограничение диапазона 0–1
                                x_center = max(0.0, min(1.0, x_center))
                                y_center = max(0.0, min(1.0, y_center))
                                width = max(0.0, min(1.0, width))
                                height = max(0.0, min(1.0, height))

                                new_labels.append({
                                    "cls": cls_idx,
                                    "x": x_center,
                                    "y": y_center,
                                    "w": width,
                                    "h": height
                                })

                    # объединяем с существующими bbox для других классов
                    final_labels = []
                    for lbl in existing_labels:
                        cls_name = self.model.names[lbl["cls"]]
                        if selected_classes is None or cls_name not in selected_classes:
                            final_labels.append(lbl)
                    final_labels.extend(new_labels)

                    # сохраняем в файл
                    with open(label_txt, "w", encoding="utf-8") as f:
                        for lbl in final_labels:
                            f.write(f'{lbl["cls"]} {lbl["x"]:.6f} {lbl["y"]:.6f} {lbl["w"]:.6f} {lbl["h"]:.6f}\n')

                    # обновление прогресса с проверкой, чтобы GUI не вис
                    processed += 1
                    perc = int((processed / total_images) * 100)
                    if perc != last_perc:
                        self.progress.emit(perc)
                        last_perc = perc

            self.log.emit("Автолейблинг завершен!")

        except Exception as e:
            import traceback
            self.log.emit(f"Ошибка автолейблинга: {str(e)}\n{traceback.format_exc()}")

        finally:
            self.finished.emit(True)


class DatasetEditor(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool)
    error = pyqtSignal(str)

    def __init__(self, model, dataset_path, yaml_path):
        super().__init__()
        self.model = model
        self.dataset_path = dataset_path
        self.yaml_path = yaml_path

    # ===================== PUBLIC =====================

    def apply(self):
        try:
            self.progress.emit(0)

            data, old_names = self._load_yaml()
            self.progress.emit(10)

            final_classes, old_id_to_new_id = self._build_mapping(old_names)
            self.progress.emit(30)

            self._rewrite_all_labels(old_id_to_new_id)
            self.progress.emit(90)

            self._write_yaml(data, final_classes)
            self.progress.emit(100)

            self.finished.emit(True)

        except Exception:
            import traceback
            traceback.print_exc()
            self.error.emit(traceback.format_exc())
            self.finished.emit(False)

    # ===================== YAML =====================

    def _load_yaml(self):
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        old_names = list(data.get("names", []))
        return data, old_names

    def _write_yaml(self, data, final_classes):
        data["names"] = final_classes
        data["nc"] = len(final_classes)

        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

    # ===================== CORE LOGIC =====================

    def _build_mapping(self, old_names):
        """
        Возвращает:
        - final_classes: список имён в финальном порядке
        - old_id_to_new_id: dict[int -> int | None]
        """

        final_classes = []
        final_name_to_id = {}
        old_name_to_id = {name: i for i, name in enumerate(old_names)}
        old_id_to_new_id = {}

        # формируем финальные классы строго в порядке таблицы
        for row in self.model.rows:
            if not row.keep:
                continue

            name = self._resolve_row_name(row)
            if not name:
                continue

            if name not in final_name_to_id:
                final_name_to_id[name] = len(final_classes)
                final_classes.append(name)

        # строим old_id → new_id
        for old_name, old_id in old_name_to_id.items():
            row = self._find_row_by_original_name(old_name)

            if not row or not row.keep:
                old_id_to_new_id[old_id] = None
                continue

            new_name = self._resolve_row_name(row)
            old_id_to_new_id[old_id] = final_name_to_id[new_name]

        return final_classes, old_id_to_new_id

    def _resolve_row_name(self, row):
        """
        Определяет итоговое имя класса для строки
        """
        if row.merge_to:
            return row.merge_to
        if row.new_name:
            return row.new_name
        return row.original_name

    def _find_row_by_original_name(self, name):
        for r in self.model.rows:
            if r.original_name == name:
                return r
        return None

    # ===================== LABELS =====================

    def _rewrite_all_labels(self, id_map):
        label_dirs = self._find_label_dirs()

        total_files = sum(len(files) for _, _, files in label_dirs)
        processed = 0

        for label_dir, _, files in label_dirs:
            for file in files:
                if not file.endswith(".txt"):
                    continue

                self._rewrite_single_label_file(
                    os.path.join(label_dir, file),
                    id_map
                )

                processed += 1
                self.progress.emit(
                    30 + int(60 * processed / max(1, total_files))
                )

    def _rewrite_single_label_file(self, path, id_map):
        new_lines = []

        with open(path, "r") as f:
            for line in f:
                cls, *coords = line.split()
                cls = int(cls)

                new_cls = id_map.get(cls)
                if new_cls is None:
                    continue  # удалённый класс

                new_lines.append(
                    " ".join([str(new_cls)] + coords) + "\n"
                )

        with open(path, "w") as f:
            f.writelines(new_lines)

    def _find_label_dirs(self):
        """
        Возвращает список (root, dirs, files) для папок labels
        """
        result = []
        for root, dirs, files in os.walk(self.dataset_path):
            if os.path.basename(root) == "labels":
                result.append((root, dirs, files))
        return result


def remove_classes(classes_to_remove, dataset_path, yaml_path):
    """
    Удаляет указанные классы из датасета.
    Оставляет только те, что НЕ в списке classes_to_remove.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    old_classes = data["names"]

    # оставляем все, кроме удаляемых
    classes_to_keep = [c for c in old_classes if c not in classes_to_remove]

    class_new_id = {name:i for i,name in enumerate(classes_to_keep)}
    id_old_to_name = {i:name for i,name in enumerate(old_classes)}

    # обновляем data.yaml
    data["names"] = classes_to_keep
    data["nc"] = len(classes_to_keep)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    for split in ["train", "valid", "test"]:
        label_dir = os.path.join(dataset_path, split, "labels")
        for file in os.listdir(label_dir):
            new_lines=[]
            with open(os.path.join(label_dir,file)) as lf:
                for line in lf:
                    cls, *coords = line.split()
                    name = id_old_to_name[int(cls)]
                    if name in classes_to_keep:
                        new_cls = class_new_id[name]
                        new_lines.append(" ".join([str(new_cls)] + coords) + "\n")
            # сохраняем новый файл (может быть пустой)
            with open(os.path.join(label_dir, file), "w") as wf:
                wf.writelines(new_lines)
    print("Классы удалены. data.yaml обновлён.")


def merge_classes(merge_map, dataset_path, yaml_path):
    """
    merge_map = {"car":"vehicle", "truck":"vehicle"}
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    old_classes = data["names"]

    # новый список классов: все объединения + те, что не объединяются
    new_classes = list(set(merge_map.values()) | set([c for c in old_classes if c not in merge_map]))
    class_new_id = {name:i for i,name in enumerate(new_classes)}
    id_old_to_name = {i:name for i,name in enumerate(old_classes)}

    # обновляем data.yaml
    data["names"] = new_classes
    data["nc"] = len(new_classes)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    for split in ["train","valid","test"]:
        label_dir = os.path.join(dataset_path, split, "labels")
        image_dir = os.path.join(dataset_path, split, "images")

        for file in os.listdir(label_dir):
            new_lines=[]
            with open(os.path.join(label_dir,file)) as lf:
                for line in lf:
                    cls,*coords = line.split()
                    old_name = id_old_to_name[int(cls)]
                    # если есть в merge_map — объединяем
                    new_name = merge_map.get(old_name, old_name)
                    new_id = class_new_id[new_name]
                    new_lines.append(" ".join([str(new_id)] + coords) + "\n")
            with open(os.path.join(label_dir,file),"w") as wf:
                wf.writelines(new_lines)
    print("Классы объединены. data.yaml обновлён.")


def reorder_classes(new_order, dataset_path, yaml_path):
    """
    Изменяет порядок классов в датасете.

    Args:
        new_order (list): Новый порядок классов, например ["person","car","truck"]
        dataset_path (str): Путь к датасету (папка, где train/valid/test)
        yaml_path (str): Путь к data.yaml
    """
    # Загружаем старый data.yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    old_classes = data["names"]

    # Проверяем, что все новые классы существуют
    if set(new_order) != set(old_classes):
        raise ValueError("new_order должен содержать все старые классы точно один раз")

    # Создаём сопоставление старый номер → новый номер
    old_to_new = {old_classes.index(name): i for i, name in enumerate(new_order)}

    # Обновляем data.yaml
    data["names"] = new_order
    data["nc"] = len(new_order)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    # Перенумеровываем все метки в split-файлах
    for split in ["train", "valid", "test"]:
        label_dir = os.path.join(dataset_path, split, "labels")
        if not os.path.exists(label_dir):
            continue
        for file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file)
            new_lines = []
            with open(file_path, "r") as lf:
                for line in lf:
                    cls, *coords = line.split()
                    new_cls = old_to_new[int(cls)]
                    new_lines.append(" ".join([str(new_cls)] + coords) + "\n")
            with open(file_path, "w") as wf:
                wf.writelines(new_lines)

    print("Порядок классов обновлён и разметка перенумерована.")


def rename_classes(rename_map, dataset_path, yaml_path):
    """
    Переименовывает классы в датасете.

    Args:
        rename_map (dict): {"старое_имя": "новое_имя", ...}
        dataset_path (str): путь к папке с train/valid/test
        yaml_path (str): путь к data.yaml
    """
    # Загружаем data.yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    old_names = data["names"]
    new_names = [rename_map.get(name, name) for name in old_names]
    data["names"] = new_names

    # Сохраняем обновлённый data.yaml
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    # Так как номера не меняются, только текст имён, менять ничего в .txt не нужно
    # YOLO использует номера классов, а имена берутся из data.yaml при инференсе
    print("Классы переименованы. data.yaml обновлён.")

