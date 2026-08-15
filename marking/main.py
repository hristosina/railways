import html
import os
import queue
import sys
import textwrap

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QEvent, QAbstractTableModel, QModelIndex, QThread, QTimer
from PyQt5.QtGui import QPixmap, QPen, QColor, QFont, QBrush, QKeySequence, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QTreeWidgetItem, \
    QGraphicsScene, QGraphicsView, QButtonGroup, QGraphicsTextItem, QGraphicsRectItem, QGraphicsItem, QShortcut, \
    QDialog, QUndoStack, QUndoCommand, QAbstractItemView, QStyledItemDelegate, QComboBox

from backend import DatasetEditor, ModelWorker
from mainwindow import Ui_MainWindow
from new_class import Ui_Dialog
from new_class_yaml import Ui_Dialog_new_class_yaml
from autolabling_dialog import Ui_Dialog_autolabling_settings
from annotation_utils import (
    canonical_annotation_lines,
    merge_annotation_lines,
    merge_preserving_existing_lines,
)

from PyQt5.QtCore import QRectF

import yaml
import random
import multiprocessing as mp
from modelTest import YOLOTestWorker
from dataset_utils import (
    IMAGE_EXTENSIONS,
    find_annotation_sets,
    find_dataset_yaml,
    normalize_dataset_path,
    resolve_annotation_set,
)

def resource_path(relative_path):
    """ Получить путь к ресурсам, как при
    разработке, так и после упаковки. """
    try: # В случае упаковки PyInstaller
        base_path = sys._MEIPASS
    except Exception: # В случае разработки
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)

def icon_path(filename):
    return resource_path(os.path.join("assets", "icons", filename))


def apply_application_style(app):
    """Load the shared light theme in development and in a PyInstaller build."""
    app.setStyle("Fusion")
    stylesheet_path = resource_path(os.path.join("assets", "styles", "light.qss"))
    try:
        with open(stylesheet_path, "r", encoding="utf-8") as stylesheet_file:
            app.setStyleSheet(stylesheet_file.read())
    except OSError as exc:
        print(f"Не удалось загрузить тему интерфейса: {exc}")

class HandleItem(QGraphicsRectItem):
    def __init__(self, parent_bbox, position_flag):
        size = 6
        super().__init__(-size/2, -size/2, size, size, parent_bbox)
        self.setBrush(Qt.blue)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.position_flag = position_flag  # какой угол изменяем: 'tl', 'tr', 'bl', 'br'
        self._start_rect = None

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            self.update_bbox(value)
        return super().itemChange(change, value)

    def update_bbox(self, new_pos):
        bbox = self.parentItem()
        rect = bbox.rect()
        if self.position_flag == 'tl':
            rect.setTopLeft(new_pos)
        elif self.position_flag == 'tr':
            rect.setTopRight(new_pos)
        elif self.position_flag == 'bl':
            rect.setBottomLeft(new_pos)
        elif self.position_flag == 'br':
            rect.setBottomRight(new_pos)

        bbox.setRect(rect)  # теперь безопасно

    def mousePressEvent(self, event):
        self._start_rect = self.parentItem().rect()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        bbox = self.parentItem()
        if self._start_rect and bbox.rect() != self._start_rect:
            scene = bbox.scene()
            if scene and hasattr(scene, "undo_stack"):
                scene.undo_stack.push(
                    ResizeBBoxCommand(bbox, self._start_rect, bbox.rect())
                )
        self._start_rect = None
        super().mouseReleaseEvent(event)

class BBoxItem(QGraphicsRectItem):
    def __init__(self, rect, cls, img_w, img_h, color, class_name=""):
        super().__init__(rect)

        # Основные свойства
        self.cls = cls
        self.img_w = img_w
        self.img_h = img_h
        self.color = color
        self.updating = False
        self.updating_handles = False
        self._old_rect = None

        # Настройка bbox
        self.setup_bbox()

        # Создание ярлыка (текст + фон)
        if class_name:
            self.create_label(class_name)

        self.create_handles()

    # -----------------------------
    # Настройка внешнего вида bbox
    # -----------------------------
    def setup_bbox(self):
        self.setPen(QPen(self.color, 2))
        self.setBrush(QBrush(Qt.NoBrush))
        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemIsMovable
        )
        self.setZValue(1)

    # -----------------------------
    # Создание текста и фона ярлыка
    # -----------------------------
    def create_label(self, class_name):
        padding_x = 2
        padding_y = 2
        offset = 2  # небольшой отступ между bbox и текстом

        # текст
        self.text_item = QGraphicsTextItem(class_name, self)  # parent = bbox
        self.text_item.setFont(QFont("Arial", 8))
        self.text_item.setDefaultTextColor(Qt.black)
        text_rect = self.text_item.boundingRect()

        # фон под текст (parent = bbox)
        self.bg_item = QGraphicsRectItem(
            QRectF(
                0, 0,
                text_rect.width() + padding_x * 2,
                text_rect.height() + padding_y * 2
            ),
            self
        )
        self.bg_item.setBrush(QBrush(Qt.white))
        self.bg_item.setPen(QPen(self.color, 1))
        self.bg_item.setZValue(2)

        # Ставим фон **над bbox**
        self.bg_item.setPos(
            self.rect().topLeft().x(),  # по горизонтали левый край bbox
            self.rect().topLeft().y() - self.bg_item.rect().height() - offset  # выше bbox
        )

        # Ставим текст внутри фона с отступами
        self.text_item.setPos(
            self.bg_item.pos().x() + padding_x,
            self.bg_item.pos().y() + padding_y
        )
        self.text_item.setZValue(3)

    # -----------------------------
    # Перевод в YOLO формат
    # -----------------------------
    def to_yolo(self):
        # sceneBoundingRect() includes the pen width. Serializing it repeatedly
        # made copied boxes grow a little on every image.
        r = self.mapRectToScene(self.rect())

        # координаты bbox
        x_c = r.center().x() / self.img_w
        y_c = r.center().y() / self.img_h
        w = r.width() / self.img_w
        h = r.height() / self.img_h

        # ограничиваем диапазон 0–1
        x_c = max(0.0, min(1.0, x_c))
        y_c = max(0.0, min(1.0, y_c))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        return f"{self.cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"

    def create_handles(self):
        self.handles = {}
        for pos_flag in ['tl', 'tr', 'bl', 'br']:
            handle = HandleItem(self, pos_flag)
            self.handles[pos_flag] = handle
        self.update_handles()

    def update_handles(self):
        if self.updating_handles:
            return  # предотвращаем рекурсию
        self.updating_handles = True

        r = self.rect()
        self.handles['tl'].setPos(r.topLeft())
        self.handles['tr'].setPos(r.topRight())
        self.handles['bl'].setPos(r.bottomLeft())
        self.handles['br'].setPos(r.bottomRight())

        self.updating_handles = False

    def setRect(self, *args):
        if self.updating:
            return super().setRect(*args)

        self.updating = True
        super().setRect(*args)
        self.update_handles()
        self.update_label_position()
        self.updating = False

    def update_label_position(self):
        # смещаем фон и текст над верхним левым углом bbox
        offset = 2
        padding_x = 2
        padding_y = 2
        if hasattr(self, 'bg_item') and hasattr(self, 'text_item'):
            self.bg_item.setPos(
                self.rect().topLeft().x(),
                self.rect().topLeft().y() - self.bg_item.rect().height() - offset
            )
            self.text_item.setPos(
                self.bg_item.pos().x() + padding_x,
                self.bg_item.pos().y() + padding_y
            )

    def set_class(self, new_class, color, new_cls_index=None):
        """
        Изменить класс бокса.
        :param new_class: строка названия класса
        :param new_cls_index: при желании, новый индекс класса
        """
        self.color = color
        self.setPen(QPen(self.color, 2))

        # обновляем цвет рамки фона под текстом
        if hasattr(self, "bg_item") and self.bg_item:
            self.bg_item.setPen(QPen(self.color, 1))

        self.cls = new_cls_index if new_cls_index is not None else self.cls
        if hasattr(self, "text_item") and self.text_item:
            self.text_item.setPlainText(new_class)

        # обновляем фон под текст
        if hasattr(self, "bg_item") and self.bg_item:
            padding_x = 2
            padding_y = 2
            text_rect = self.text_item.boundingRect()
            self.bg_item.setRect(0, 0, text_rect.width() + padding_x * 2, text_rect.height() + padding_y * 2)

        # перемещаем подпись над bbox
        self.update_label_position()

    def mousePressEvent(self, event):
        self._old_pos = self.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.pos() != self._old_pos:
            scene = self.scene()
            if scene and hasattr(scene, "undo_stack"):
                scene.undo_stack.push(
                    MoveBBoxCommand(self, self._old_pos, self.pos())
                )

    def set_interactive(self, enabled: bool):
        # сам bbox
        self.setFlag(QGraphicsItem.ItemIsMovable, enabled)
        self.setFlag(QGraphicsItem.ItemIsSelectable, enabled)

        # хендлы
        if hasattr(self, "handles"):
            for h in self.handles.values():
                h.setFlag(QGraphicsItem.ItemIsMovable, enabled)
                h.setVisible(enabled)


class MoveBBoxCommand(QUndoCommand):
    def __init__(self, bbox, old_pos, new_pos):
        super().__init__("Move box")
        self.bbox = bbox
        self.old_pos = old_pos
        self.new_pos = new_pos

    def undo(self):
        self.bbox.setPos(self.old_pos)

    def redo(self):
        self.bbox.setPos(self.new_pos)

class ResizeBBoxCommand(QUndoCommand):
    def __init__(self, bbox, old_rect, new_rect):
        super().__init__("Resize box")
        self.bbox = bbox
        self.old_rect = old_rect
        self.new_rect = new_rect

    def undo(self):
        self.bbox.setRect(self.old_rect)

    def redo(self):
        self.bbox.setRect(self.new_rect)

class AutoLabelingDialog(QDialog):
    def __init__(self, yaml_path):
        super().__init__()

        self.ui = Ui_Dialog_autolabling_settings()
        self.ui.setupUi(self)

        # Чтение классов из yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.classes = data.get("names", [])

        # Словарь для чекбоксов
        self.checkboxes = {}

        # Добавляем чекбоксы в scrollArea
        layout = QtWidgets.QVBoxLayout(self.ui.scrollAreaWidgetContents)
        for cls_name in self.classes:
            cb = QtWidgets.QCheckBox(cls_name)
            cb.setChecked(True)  # по умолчанию все выбраны
            layout.addWidget(cb)
            self.checkboxes[cls_name] = cb
        layout.addStretch()

    def get_settings(self):
        """
        Возвращает выбранные классы и порог уверенности
        """
        selected_classes = [name for name, cb in self.checkboxes.items() if cb.isChecked()]
        conf = self.ui.doubleSpinBox.value()
        return selected_classes, conf


class NewClassDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.ui = Ui_Dialog_new_class_yaml()
        self.ui.setupUi(self)
        self.new_class_name = ""

    def accept(self):
        # сохраняем выбранный индекс перед закрытием
        self.new_class_name = self.ui.lineEdit.text()
        super().accept()

    def get_new_class_name(self):
        """Возвращает выбранный индекс или None, если отмена"""
        return self.new_class_name

class NewBoxDialog(QDialog):
    def __init__(self, classes, current_index=None):
        super().__init__()

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.classes = classes
        self.ui.comboBox.addItems(classes)
        if current_index is not None:
            self.selected_index = current_index
            self.ui.comboBox.setCurrentIndex(current_index)

    def accept(self):
        # сохраняем выбранный индекс перед закрытием
        self.selected_index = self.ui.comboBox.currentIndex()
        super().accept()

    def get_selected_index(self):
        """Возвращает выбранный индекс или None, если отмена"""
        return self.selected_index


class ClassRow:
    def __init__(self, class_id, original_name, keep=True, merge_to="", new_name=""):
        self.class_id = class_id
        self.original_name = original_name
        self.keep = keep
        self.merge_to = merge_to
        self.new_name = new_name or original_name

class ClassesTableModel(QAbstractTableModel):
    HEADERS = ["Выделить", "ID", "Имя", "Объединить в", "Новое имя"]

    def __init__(self, rows):
        super().__init__()
        self.rows = rows
        self.group_colors = {}

    def rowCount(self, parent=QModelIndex()):
        return len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return len(self.HEADERS)

    def data(self, index, role):
        if not index.isValid():
            return None
        row = self.rows[index.row()]
        col = index.column()

        if role in (Qt.DisplayRole, Qt.EditRole):
            if col == 1:
                if row.class_id < 0:
                    return ""
                return row.class_id
            elif col == 2:
                return row.original_name
            elif col == 3:
                return row.merge_to
            elif col == 4:
                return row.new_name

        if role == Qt.CheckStateRole and col == 0:
            return Qt.Checked if row.keep else Qt.Unchecked

        if role == Qt.BackgroundRole:
            if not row.keep:
                # светло-серый цвет для невыбранных
                return QColor(220, 220, 220)
            if row.merge_to:
                key = row.merge_to
                if key not in self.group_colors:
                    self.group_colors[key] = self.generate_pastel_color()
                return self.group_colors[key]

        return None

    def generate_pastel_color(self):
        base = 200  # минимальное значение каналов, чтобы цвет был светлым
        r = random.randint(base, 255)
        g = random.randint(base, 255)
        b = random.randint(base, 255)
        return QColor(r, g, b)

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled

        row = self.rows[index.row()]

        # по умолчанию строки кликабельны только если checked
        if not row.keep:
            # если галочка снята, только чекбокс можно кликать
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable if index.column() == 0 else Qt.ItemIsEnabled

        # если галочка стоит
        flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if index.column() == 0:
            flags |= Qt.ItemIsUserCheckable
        elif index.column() == 3:  # "Объединить в" всегда редактируемая для выбранных
            flags |= Qt.ItemIsEditable
        elif index.column() == 4:  # Новое имя редактируемое только если merge_to пустое
            if not row.merge_to:
                flags |= Qt.ItemIsEditable
        return flags

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        row = self.rows[index.row()]
        col = index.column()

        if role == Qt.EditRole:
            if col == 3:  # merge_to
                row.merge_to = str(value)
                if row.merge_to:
                    row.new_name = row.merge_to
                self.regroup_rows_by_merge()
                self.update_ids()
                return True
            elif col == 4:  # new_name
                if not row.merge_to and row.keep:  # редактируем только если merge_to пустое и keep=True
                    row.new_name = str(value)
                    self.dataChanged.emit(index, index)
                    return True

        elif role == Qt.CheckStateRole and col == 0:
            row.keep = value == Qt.Checked
            self.dataChanged.emit(index, index)
            # пересчитываем ID, чтобы не учитывать строки с keep=False
            self.update_ids()
            # обновляем вид, чтобы колонки стали некликабельными, если нужно
            self.layoutChanged.emit()
            return True

        return False

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]

    # методы для перемещения строк
    def move_row_up(self, row):
        if row <= 0:
            return
        self.beginMoveRows(QModelIndex(), row, row, QModelIndex(), row - 1)
        self.rows[row - 1], self.rows[row] = self.rows[row], self.rows[row - 1]
        self.endMoveRows()

    def move_row_down(self, row):
        if row >= len(self.rows) - 1:
            return
        self.beginMoveRows(QModelIndex(), row, row, QModelIndex(), row + 2)
        self.rows[row], self.rows[row + 1] = self.rows[row + 1], self.rows[row]
        self.endMoveRows()

    def regroup_rows_by_merge(self):
        rows = self.rows
        grouped = {}
        for row in rows:
            key = row.merge_to if row.merge_to else row.original_name
            grouped.setdefault(key, []).append(row)
        new_rows = []
        for key in grouped:
            new_rows.extend(grouped[key])
        self.rows = new_rows
        self.layoutChanged.emit()

    def update_ids(self):
        """
        Присваиваем ID строкам сверху вниз.
        Объединённые строки получают один ID (ID первой строки в группе).
        Строки с keep=False игнорируются.
        """
        current_id = 0
        row_idx = 0
        while row_idx < len(self.rows):
            row = self.rows[row_idx]
            if not row.keep:
                row.class_id = -1  # или None, чтобы показать, что не участвует
                row_idx += 1
                continue

            if row.merge_to:
                merge_value = row.merge_to
                # находим все строки с таким merge_to подряд и keep=True
                start_idx = row_idx
                while row_idx + 1 < len(self.rows) and self.rows[row_idx + 1].merge_to == merge_value and self.rows[
                    row_idx + 1].keep:
                    row_idx += 1
                for i in range(start_idx, row_idx + 1):
                    self.rows[i].class_id = current_id
                current_id += 1
            else:
                row.class_id = current_id
                current_id += 1
            row_idx += 1

        self.layoutChanged.emit()


class MergeDelegate(QStyledItemDelegate):
    def __init__(self, get_existing_classes_func, parent=None):
        super().__init__(parent)
        self.get_existing_classes_func = get_existing_classes_func
        self.items = self.get_existing_classes_func()

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.setEditable(True)
        combo.addItems(self.items)
        return combo

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.EditRole)
        i = editor.findText(value)
        if i >= 0:
            editor.setCurrentIndex(i)
        else:
            editor.setEditText(value)

    def setModelData(self, editor, model, index):
        text = editor.currentText()
        model.setData(index, text, Qt.EditRole)

        # добавляем новое значение в список подсказок
        if text and text not in self.items:
            self.items.append(text)

        # Перегруппируем строки после изменения merge_to
        if hasattr(model, 'regroup_rows_by_merge'):
            model.regroup_rows_by_merge()
        if hasattr(model, 'update_ids'):
            model.update_ids()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.delegate = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon(icon_path("railway.png")))
        self.ui.menubar.setVisible(False)
        self.ui.centralwidget.layout().setContentsMargins(14, 12, 14, 10)
        self.ui.centralwidget.layout().setHorizontalSpacing(14)

        self.ui.toolButton_reset.setIcon(QIcon(icon_path("reset.png")))
        self.ui.toolButton_edit.setIcon(QIcon(icon_path("edit.png")))
        self.ui.toolButton_add.setIcon(QIcon(icon_path("add.png")))
        self.ui.toolButton_delete.setIcon(QIcon(icon_path("delete.png")))
        self.ui.toolButton_cancel.setIcon(QIcon(icon_path("cancel.png")))
        self.ui.toolButton_apply.setIcon(QIcon(icon_path("apply.png")))
        self.ui.toolButton_zoom.setIcon(QIcon(icon_path("search.png")))
        self.ui.toolButton_cursor.setIcon(QIcon(icon_path("cursor.png")))
        self.ui.toolButton_move.setIcon(QIcon(icon_path("move.png")))
        self.ui.toolButton_save.setIcon(QIcon(icon_path("save.png")))
        self.ui.pushButton_copy_previous.setIcon(QIcon(icon_path("copy.png")))
        self.ui.pushButton_clear_annotations.setIcon(QIcon(icon_path("delete_all.png")))
        self.ui.toolButton_more.setIcon(QIcon(icon_path("other.png")))
        self.ui.toolButton_more.setIconSize(QtCore.QSize(25, 25))
        self.ui.pushButton_model_path.setIcon(QIcon(icon_path("search.png")))
        self.ui.pushButton_dataset_path.setIcon(QIcon(icon_path("search.png")))
        self.ui.pushButton_yaml_path.setIcon(QIcon(icon_path("search.png")))
        self.ui.toolButton_add_class.setIcon(QIcon(icon_path("add.png")))
        self.ui.toolButton_delete_class.setIcon(QIcon(icon_path("delete.png")))
        self.ui.toolButton_reset_class_info.setIcon(QIcon(icon_path("reset.png")))

        self.ui.treeWidget_menu.setTextElideMode(Qt.ElideNone)
        self.ui.treeWidget_menu.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.ui.treeWidget_menu.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.ui.lineEdit_dataset_path.setPlaceholderText("Корень датасета или папка images")
        self.ui.lineEdit_yaml_path.setPlaceholderText("Обычно находится автоматически")
        self.ui.lineEdit_model_path.setPlaceholderText("Для ручной разметки оставьте пустым")
        self.ui.pushButton_load_mark_info.setText("Открыть датасет для разметки")
        self.ui.pushButton_load_mark_info.setToolTip(
            "Загрузить изображения и классы. Файл модели для этого не требуется."
        )
        self.ui.pushButton_autolabling.setText("Авторазметка с помощью модели")
        self.ui.pushButton_autolabling.setToolTip(
            "Автоматически создать рамки с помощью выбранной модели детекции"
        )

        self.image_items = []  # [(img_path, label_path), ...]
        self.current_index = 0
        self.box_items = []
        self.img_count = 0
        self.class_names = []
        self.class_colors = {}
        self.scene = None
        self.item = None
        self.current_label_path = None
        self.yoloWorker = None
        self.active_process_kind = None
        self.annotation_clipboard = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.poll_queue)

        self.ui.progressBar_class_info.setVisible(False)
        self.ui.tableView_class_info
        self.undo_stack = QUndoStack(self)

        self.ui.graphicsView.setBackgroundBrush(QColor("#eef2f7"))
        self.ui.graphicsView.viewport().installEventFilter(self)

        self.ui.progressBar_autolabling.setVisible(False)

        self.ui.stackedWidget.setCurrentIndex(2)

        self.ui.treeWidget_menu.expandAll()
        for item_index in range(self.ui.treeWidget_menu.topLevelItemCount()):
            item = self.ui.treeWidget_menu.topLevelItem(item_index)
            item_font = item.font(0)
            item_font.setBold(True)
            item.setFont(0, item_font)
        self.ui.treeWidget_menu.setCurrentItem(self.ui.treeWidget_menu.topLevelItem(1).child(0))
        self.ui.treeWidget_menu.itemClicked.connect(self.on_treeWidget_menu_item_clicked)
        self.configure_source_panel("marking")
        QTimer.singleShot(0, self.adjust_menu_width)

        self.ui.pushButton_model_path.clicked.connect(self.choose_model_path)
        self.ui.pushButton_dataset_path.clicked.connect(self.choose_dataset_path)
        self.ui.pushButton_yaml_path.clicked.connect(self.choose_yaml_path)
        self.ui.pushButton_load_mark_info.clicked.connect(self.load_mark_info)
        self.ui.pushButton_copy_previous.clicked.connect(self.copy_previous_annotations)
        self.ui.pushButton_clear_annotations.clicked.connect(self.clear_current_annotations)
        self.setup_more_actions_menu()

        self.cursor_group = QButtonGroup(self)
        self.cursor_group.setExclusive(True)  # только одна кнопка может быть нажата
        self.cursor_group.addButton(self.ui.toolButton_cursor)
        self.cursor_group.addButton(self.ui.toolButton_move)
        self.cursor_group.buttonToggled.connect(self.on_drag_mode_changed)

        self.ui.toolButton_next.clicked.connect(self.next_img_show)
        # горячая клавиша вправо
        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.setContext(Qt.ApplicationShortcut)
        self.shortcut_right.activated.connect(self.ui.toolButton_next.click)

        self.ui.toolButton_prev.clicked.connect(self.prev_img_show)
        # горячая клавиша влево
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.setContext(Qt.ApplicationShortcut)
        self.shortcut_left.activated.connect(self.ui.toolButton_prev.click)

        self.shortcut_left.activated.connect(self.ui.toolButton_prev.click)

        self.ui.toolButton_delete.clicked.connect(self.delete_selected_boxes)
        # горячая клавиша Delete
        self.shortcut_delete = QShortcut(QKeySequence("Delete"), self)
        self.shortcut_delete.activated.connect(self.ui.toolButton_delete.click)  # вызывает нажатие кнопки

        self.shortcut_copy_annotations = QShortcut(QKeySequence.Copy, self)
        self.shortcut_copy_annotations.activated.connect(self.copy_active_context)
        self.shortcut_paste_annotations = QShortcut(QKeySequence.Paste, self)
        self.shortcut_paste_annotations.activated.connect(self.paste_active_context)
        self.shortcut_clear_annotations = QShortcut(QKeySequence("Ctrl+D"), self)
        self.shortcut_clear_annotations.activated.connect(self.clear_current_annotations)
        self.shortcut_copy_previous = QShortcut(QKeySequence("Ctrl+P"), self)
        self.shortcut_copy_previous.activated.connect(self.copy_previous_annotations)

        self.ui.toolButton_edit.clicked.connect(self.edit_selected_boxes)
        # горячая клавиша Ctrl+E
        self.shortcut_ctrl_e = QShortcut(QKeySequence("Ctrl+E"), self)
        self.shortcut_ctrl_e.activated.connect(self.ui.toolButton_edit.click)  # вызывает нажатие кнопки

        self.ui.toolButton_add.clicked.connect(self.add_new_box)
        # горячая клавиша Ctrl+N
        self.shortcut_ctrl_n = QShortcut(QKeySequence("Ctrl+N"), self)
        self.shortcut_ctrl_n.activated.connect(self.ui.toolButton_add.click)  # вызывает нажатие кнопки

        self.ui.toolButton_cancel.clicked.connect(self.undo_stack.undo)
        self.ui.toolButton_apply.clicked.connect(self.undo_stack.redo)
        self.ui.toolButton_cancel.setEnabled(False)
        self.ui.toolButton_apply.setEnabled(False)
        self.undo_stack.canUndoChanged.connect(self.ui.toolButton_cancel.setEnabled)
        self.undo_stack.canRedoChanged.connect(self.ui.toolButton_apply.setEnabled)
        QShortcut(QKeySequence.Undo, self, self.undo_stack.undo)
        QShortcut(QKeySequence.Redo, self, self.undo_stack.redo)

        self.ui.toolButton_save.clicked.connect(self.save_current_labels)
        self.shortcut_ctrl_s = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_ctrl_s.activated.connect(self.save_current_labels)

        self.ui.toolButton_reset.clicked.connect(self.reset_current_labels)
        self.shortcut_ctrl_r = QShortcut(QKeySequence("Ctrl+R"), self)
        self.shortcut_ctrl_r.activated.connect(self.reset_current_labels)

        self.ui.pushButton_load_class_info.clicked.connect(self.setup_classes_table)

        self.ui.toolButton_up.clicked.connect(self.move_row_up)
        self.ui.toolButton_down.clicked.connect(self.move_row_down)

        self.ui.toolButton_add_class.clicked.connect(self.add_new_class)

        self.ui.toolButton_delete_class.clicked.connect(self.delete_selected_class)

        self.ui.treeWidget_dir.itemClicked.connect(self.on_treeWidget_dir_item_clicked)

        self.ui.pushButton_save_class_info.clicked.connect(self.on_apply_changes_clicked)

        self.ui.pushButton_autolabling.clicked.connect(self.start_autolabling)

        self.ui.pushButton_start_training.clicked.connect(self.start_training)
        self.ui.progressBar_training.setVisible(False)

        self.ui.pushButton_start_model_test.clicked.connect(self.start_model_test)
        self.ui.progressBar_testing.setVisible(False)
        self.set_annotation_controls_enabled(False)
        self.wrap_all_tooltips()


    def get_existing_classes(self):
        return [row.original_name for row in self.classes_model.rows]

    def adjust_menu_width(self):
        """Fit the menu to its longest item without clipping its text."""
        self.ui.treeWidget_menu.resizeColumnToContents(0)
        width = max(280, self.ui.treeWidget_menu.sizeHintForColumn(0) + 48)
        self.ui.treeWidget_menu.setMinimumWidth(width)
        self.ui.treeWidget_menu.setMaximumWidth(width)

    @staticmethod
    def wrapped_tooltip(text, width=58):
        if not text or text.lstrip().startswith("<"):
            return text
        lines = textwrap.wrap(text, width=width, break_long_words=False)
        return "<qt>" + "<br>".join(html.escape(line) for line in lines) + "</qt>"

    def wrap_all_tooltips(self):
        for widget in self.findChildren(QtWidgets.QWidget):
            if widget.toolTip():
                widget.setToolTip(self.wrapped_tooltip(widget.toolTip()))

    def setup_more_actions_menu(self):
        menu = QtWidgets.QMenu(self.ui.toolButton_more)
        self.ui.toolButton_more.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.ui.toolButton_more.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.action_copy_previous = menu.addAction(
            QIcon(icon_path("copy.png")), "Добавить разметку предыдущего файла (Ctrl+P)"
        )
        self.action_copy_previous.triggered.connect(self.copy_previous_annotations)
        self.action_apply_selected_folder = menu.addAction(
            QIcon(icon_path("apply.png")), "Применить выделенные рамки ко всей папке"
        )
        self.action_apply_selected_folder.triggered.connect(
            self.apply_selected_annotations_to_folder
        )
        self.action_clear_annotations = menu.addAction(
            QIcon(icon_path("delete_all.png")), "Удалить всю разметку текущего файла (Ctrl+D)"
        )
        self.action_clear_annotations.triggered.connect(self.clear_current_annotations)
        menu.addSeparator()
        self.action_reset_annotations = menu.addAction(
            QIcon(icon_path("reset.png")), "Вернуть разметку к сохранённой версии"
        )
        self.action_reset_annotations.triggered.connect(self.reset_current_labels)
        self.more_actions_menu = menu
        self.ui.toolButton_more.clicked.connect(self.show_more_actions_menu)

    def show_more_actions_menu(self):
        """Open the extra-actions menu without Qt's overlaid arrow indicator."""
        button = self.ui.toolButton_more
        position = button.mapToGlobal(QtCore.QPoint(0, button.height()))
        self.more_actions_menu.exec_(position)

    def set_annotation_controls_enabled(self, enabled):
        for widget in (
            self.ui.toolButton_edit, self.ui.toolButton_add,
            self.ui.toolButton_delete, self.ui.toolButton_zoom,
            self.ui.toolButton_cursor, self.ui.toolButton_move,
            self.ui.toolButton_save, self.ui.toolButton_more,
        ):
            widget.setEnabled(enabled)
        has_previous = enabled and self.current_index > 0
        has_next = enabled and self.current_index + 1 < len(self.image_items)
        self.ui.toolButton_prev.setEnabled(has_previous)
        self.ui.toolButton_next.setEnabled(has_next)
        if hasattr(self, "action_copy_previous"):
            self.action_copy_previous.setEnabled(has_previous)
            self.action_apply_selected_folder.setEnabled(enabled)
            self.action_clear_annotations.setEnabled(enabled)
            self.action_reset_annotations.setEnabled(enabled)

    def configure_source_panel(self, section):
        """Show only inputs needed by the selected workflow."""
        model_widgets = (
            self.ui.label_model_path,
            self.ui.lineEdit_model_path,
            self.ui.pushButton_model_path,
        )
        yaml_widgets = (
            self.ui.label_yaml_path,
            self.ui.lineEdit_yaml_path,
            self.ui.pushButton_yaml_path,
        )
        show_model = section in ("training", "testing")
        show_yaml = section != "marking"
        for widget in model_widgets:
            widget.setVisible(show_model)
            widget.setEnabled(show_model)
        for widget in yaml_widgets:
            widget.setVisible(show_yaml)
            widget.setEnabled(show_yaml)

        if section == "marking":
            self.ui.groupBox_src_files.setTitle("Данные для разметки")
            self.ui.label_source_hint.setText(
                "Выберите папку датасета. data.yaml определится автоматически."
            )
        elif section == "training":
            self.ui.groupBox_src_files.setTitle("Данные для обучения")
            self.ui.label_model_path.setText("Начальная модель детекции:")
            self.ui.lineEdit_model_path.setPlaceholderText("Например, yolov12n.pt")
            self.ui.label_source_hint.setText(
                "Для обучения укажите начальную модель, корень датасета и data.yaml."
            )
        elif section == "testing":
            self.ui.groupBox_src_files.setTitle("Данные для тестирования")
            self.ui.label_model_path.setText("Обученная модель детекции:")
            self.ui.lineEdit_model_path.setPlaceholderText("Модель, качество которой нужно оценить")
            self.ui.label_source_hint.setText(
                "Для тестирования выберите обученную модель, папку тестовых сценариев и data.yaml."
            )
        else:
            self.ui.groupBox_src_files.setTitle("Данные для редактирования классов")
            self.ui.label_source_hint.setText(
                "Выберите корень датасета и data.yaml. Модель для изменения классов не требуется."
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)  # стандартная обработка resize

        # Подгоняем изображение под размер QGraphicsView
        if self.ui.graphicsView.scene():
            self.ui.graphicsView.fitInView(
                self.ui.graphicsView.scene().sceneRect(),
                Qt.KeepAspectRatio
            )

    def closeEvent(self, event):
        self.save_current_labels()
        event.accept()

    def set_boxes_interactive(self, enabled: bool):
        if not self.scene:
            return

        for item in self.scene.items():
            if isinstance(item, BBoxItem):
                item.set_interactive(enabled)

    def eventFilter(self, source, event):
        if (
            source is self.ui.graphicsView.viewport()
            and event.type() == QEvent.MouseButtonPress
            and event.button() == Qt.LeftButton
            and event.modifiers() & Qt.ShiftModifier
        ):
            item = self.ui.graphicsView.itemAt(event.pos())
            while item is not None and not isinstance(item, BBoxItem):
                item = item.parentItem()
            if isinstance(item, BBoxItem):
                item.setSelected(not item.isSelected())
                event.accept()
                return True

        if source is self.ui.graphicsView.viewport() and event.type() == QEvent.Wheel:

            # CTRL или включена кнопка зума → масштаб
            if event.modifiers() & Qt.ControlModifier or self.ui.toolButton_zoom.isChecked():
                view = self.ui.graphicsView

                zoom_in = event.angleDelta().y() > 0
                factor = 1.25 if zoom_in else 0.8

                # zoom по позиции мыши
                pos_before = view.mapToScene(event.pos())
                view.scale(factor, factor)
                pos_after = view.mapToScene(event.pos())
                delta = pos_after - pos_before
                view.translate(delta.x(), delta.y())

                event.accept()
                return True  # ← ВАЖНО: полностью блокируем скролл

            # если Ctrl НЕ зажат → даём Qt самому скроллить
            return False

        return super().eventFilter(source, event)

    def next_img_show(self):
        if self.img_count <= 0:
            return
        if (self.current_index + 2) > self.img_count:
            return
        self.save_current_labels()
        self.current_index = self.current_index + 1
        img_path, label_path = self.image_items[self.current_index]
        self.place_img(img_path, label_path)

    def prev_img_show(self):
        if self.img_count <= 0:
            return
        if (self.current_index - 1) < 0:
            return
        self.save_current_labels()
        self.current_index = self.current_index - 1
        img_path, label_path = self.image_items[self.current_index]
        self.place_img(img_path, label_path)

    def load_dataset(self, folder_path):
        try:
            _, images_dir, labels_dir = resolve_annotation_set(folder_path)
        except ValueError as exc:
            QMessageBox.warning(self, "Невозможно открыть папку", str(exc))
            return False

        self.image_items.clear()

        for name in sorted(os.listdir(str(images_dir))):
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                continue

            img_path = os.path.join(str(images_dir), name)
            label_path = os.path.join(
                str(labels_dir),
                os.path.splitext(name)[0] + ".txt"
            )

            self.image_items.append((img_path, label_path))

        self.img_count = len(self.image_items)
        self.current_index = 0
        if not self.image_items:
            self.ui.label_img_num.setText("Изображение 0/0")
            self.set_annotation_controls_enabled(False)
            QMessageBox.information(
                self,
                "Нет изображений",
                f"В папке {images_dir} нет поддерживаемых изображений."
            )
            return False
        img_path, label_path = self.image_items[self.current_index]
        self.place_img(img_path, label_path)
        return True

    def on_drag_mode_changed(self, button, checked):
        if not checked:
            return  # если кнопка отжата — ничего не делаем

        if button == self.ui.toolButton_move:
            self.ui.graphicsView.setDragMode(QGraphicsView.ScrollHandDrag)
            self.set_boxes_interactive(False)
        elif button == self.ui.toolButton_cursor:
            self.ui.graphicsView.setDragMode(QGraphicsView.NoDrag)
            self.set_boxes_interactive(True)

    def draw_boxes_from_file(self, txt_path):
        if not self.item:
            return

        pixmap = self.item.pixmap()
        img_w = pixmap.width()
        img_h = pixmap.height()

        with open(txt_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                try:
                    cls, x_c, y_c, w_rel, h_rel = map(float, parts)
                except ValueError:
                    print(f"Пропущена некорректная строка {line_number} в {txt_path}")
                    continue
                cls = int(cls)

                x = (x_c - w_rel / 2) * img_w
                y = (y_c - h_rel / 2) * img_h
                w = w_rel * img_w
                h = h_rel * img_h

                rect = QRectF(x, y, w, h)

                color = self.class_colors.get(cls, QColor(255, 0, 0))
                class_name = (
                    self.class_names[cls]
                    if 0 <= cls < len(self.class_names)
                    else f"class {cls}"
                )

                bbox = BBoxItem(rect, cls, img_w, img_h, color, class_name)
                self.scene.addItem(bbox)
                self.box_items.append(bbox)

    def place_img(self, img_path, label_path=None):
        # print(img_path)
        self.ui.label_img_num.setText(f"Изображение {self.current_index + 1} / {self.img_count} ({os.path.basename(img_path)})")
        self.view = self.ui.graphicsView

        if not self.scene:
            self.scene = QGraphicsScene(self.view)
            self.view.setScene(self.scene)
            self.scene.undo_stack = self.undo_stack

        self.undo_stack.clear()
        self.scene.clear()
        self.box_items.clear()

        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            self.set_annotation_controls_enabled(False)
            QMessageBox.critical(
                self, "Ошибка изображения", f"Не удалось открыть изображение:\n{img_path}"
            )
            return
        self.item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(self.item.boundingRect())
        self.view.resetTransform()

        self.current_label_path = label_path  # для сохранения

        if label_path and os.path.exists(label_path):
            self.draw_boxes_from_file(label_path)

        # Подгоняем изображение под размер QGraphicsView
        if self.ui.graphicsView.scene():
            self.ui.graphicsView.fitInView(
                self.ui.graphicsView.scene().sceneRect(),
                Qt.KeepAspectRatio
            )
        self.set_annotation_controls_enabled(True)

    def on_treeWidget_menu_item_clicked(self, item, column):
        parent = item.parent()
        if parent:
            parent_index = self.ui.treeWidget_menu.indexOfTopLevelItem(parent)
            child_index = parent.indexOfChild(item)
            if (parent_index == 0) and (child_index == 0):
                self.ui.stackedWidget.setCurrentIndex(0)
                self.configure_source_panel("training")
            elif (parent_index == 0) and (child_index == 1):
                self.ui.stackedWidget.setCurrentIndex(1)
                self.configure_source_panel("testing")
            elif (parent_index == 1) and (child_index == 0):
                self.ui.stackedWidget.setCurrentIndex(2)
                self.configure_source_panel("marking")
            elif (parent_index == 1) and (child_index == 1):
                self.ui.stackedWidget.setCurrentIndex(3)
                self.configure_source_panel("dataset_edit")

    def choose_model_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите модель детекции",
            os.getcwd(),
            "Модели детекции (*.pt *.onnx *.engine *.torchscript);;Все файлы (*)"
        )
        if file_path:
            self.ui.lineEdit_model_path.setText(file_path)

    def choose_dataset_path(self):
        folder_path = (QFileDialog.getExistingDirectory(
            self,
            "Выберите корень датасета, split или папку images",
            os.getcwd()
        ))
        if folder_path:
            try:
                dataset_path = normalize_dataset_path(folder_path)
            except ValueError as exc:
                QMessageBox.warning(self, "Неверная папка", str(exc))
                return
            self.ui.lineEdit_dataset_path.setText(str(dataset_path))
            yaml_path = find_dataset_yaml(dataset_path)
            if yaml_path:
                self.ui.lineEdit_yaml_path.setText(str(yaml_path))

    def choose_yaml_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите data.yaml со списком классов",
            os.getcwd(),
            "YAML (*.yaml *.yml)"
        )
        if file_path:
            self.ui.lineEdit_yaml_path.setText(file_path)

    def load_mark_info(self):
        self.save_current_labels()
        self.ui.treeWidget_dir.clear()

        dataset_path = self.ui.lineEdit_dataset_path.text().strip()
        if not dataset_path:
            QMessageBox.warning(
                self,
                "Не выбран датасет",
                "Нажмите кнопку с лупой рядом с полем «Папка датасета» и выберите корень датасета или папку images."
            )
            return
        try:
            dataset_path = str(normalize_dataset_path(dataset_path))
        except ValueError as exc:
            QMessageBox.warning(self, "Неверный датасет", str(exc))
            return

        self.ui.lineEdit_dataset_path.setText(dataset_path)
        yaml_path = self.ui.lineEdit_yaml_path.text().strip()
        if not os.path.isfile(yaml_path):
            detected_yaml = find_dataset_yaml(dataset_path)
            if detected_yaml:
                yaml_path = str(detected_yaml)
                self.ui.lineEdit_yaml_path.setText(yaml_path)
        if not os.path.isfile(yaml_path):
            yaml_path, _ = QFileDialog.getOpenFileName(
                self, "Не найден data.yaml — выберите файл со списком классов",
                dataset_path, "YAML (*.yaml *.yml)"
            )
            if not yaml_path:
                QMessageBox.warning(
                    self, "Не найден data.yaml",
                    "Без списка классов датасет открыть нельзя."
                )
                return
            self.ui.lineEdit_yaml_path.setText(yaml_path)
        try:
            self.load_classes(yaml_path)
            self.generate_class_colors()
        except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Ошибка YAML", f"Не удалось прочитать классы:\n{exc}")
            return

        annotation_sets = self.fill_treeWidget_dir(dataset_path, self.ui.treeWidget_dir)
        if not annotation_sets:
            QMessageBox.warning(
                self,
                "Не найдены изображения",
                "В датасете не найдено ни одной пары папок images/labels."
            )
            return

        first_item = self.ui.treeWidget_dir.topLevelItem(0)
        self.ui.treeWidget_dir.setCurrentItem(first_item)
        self.load_dataset(first_item.data(0, Qt.UserRole))


    def fill_treeWidget_dir(self, path, parent_item):
        annotation_sets = find_annotation_sets(path)
        root = os.path.abspath(path)
        for annotation_path in annotation_sets:
            relative_path = os.path.relpath(str(annotation_path), root)
            display_name = os.path.basename(root) if relative_path == "." else relative_path
            tree_item = QTreeWidgetItem(parent_item, [display_name])
            tree_item.setData(0, Qt.UserRole, str(annotation_path))
        return annotation_sets

    def on_treeWidget_dir_item_clicked(self, item, column):
        folder_path = item.data(0, Qt.UserRole)
        if folder_path:
            self.save_current_labels()
            self.load_dataset(folder_path)

    def load_classes(self, yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("YAML должен содержать словарь с полем names.")
        names = data.get("names", [])
        if isinstance(names, dict):
            names = [names[key] for key in sorted(names, key=lambda value: int(value))]
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError("Поле names должно быть списком названий классов.")
        self.class_names = names

    def generate_class_colors(self):
        self.class_colors = {}

        if not self.class_names:
            return

        for i, name in enumerate(self.class_names):
            # простая, но стабильная генерация цветов
            hue = int(360 * i / len(self.class_names))
            color = QColor()
            color.setHsv(hue, 255, 200)

            self.class_colors[i] = color

    def delete_selected_boxes(self):
        selected = False
        if self.scene:
            for item in self.scene.selectedItems():
                # Проверяем, что это наш BBoxItem
                if isinstance(item, BBoxItem):
                    self.undo_stack.push(RemoveBBoxCommand(self.scene, item))
                    selected = True
        if not selected:
            msg = QMessageBox()
            msg.setWindowTitle("Внимание")
            msg.setText("Необходимо выделить разметку для удаления!")
            msg.setIcon(QMessageBox.Information)
            msg.exec_()  # показывает окно

    def edit_selected_boxes(self):
        selected = False

        if self.scene:
            for item in self.scene.selectedItems():
                # Проверяем, что это наш BBoxItem
                if isinstance(item, BBoxItem):
                    old_cls = item.cls
                    old_color = item.color
                    old_name = self.class_names[old_cls]
                    dialog = NewBoxDialog(self.class_names, item.cls)
                    if dialog.exec_() == QDialog.Accepted:
                        new_index = dialog.get_selected_index()
                        new_class_name = self.class_names[new_index]
                        self.undo_stack.push(
                            ChangeClassCommand(
                                item,
                                old_cls, old_color,
                                new_index, self.class_colors[new_index],
                                old_name, new_class_name
                            )
                        )
                        selected = True
                        break
        if not selected:
            msg = QMessageBox()
            msg.setWindowTitle("Внимание")
            msg.setText("Необходимо выделить разметку для редактирования класса!")
            msg.setIcon(QMessageBox.Information)
            msg.exec_()  # показывает окно

    def add_new_class(self):
        dialog = NewClassDialog()  # твой диалог для ввода имени
        if dialog.exec_() != QDialog.Accepted:
            return
        new_class_name = dialog.get_new_class_name()
        if not new_class_name:
            return

        # создаём новый объект строки
        new_row = ClassRow(
            class_id=len(self.classes_model.rows),  # временный ID, потом пересчитаем
            original_name="",
            merge_to="",  # пока нет объединения
            new_name=new_class_name,
            keep=True
        )

        # добавляем в модель
        self.classes_model.rows.append(new_row)

        # обновляем таблицу
        self.classes_model.update_ids()  # пересчитываем ID
        self.classes_model.layoutChanged.emit()

        # добавляем новое имя в список подсказок делегата
        delegate = self.ui.tableView_class_info.itemDelegateForColumn(3)
        if delegate and new_class_name not in delegate.items:
            delegate.items.append(new_class_name)


    def add_new_box(self):
        if not self.scene or not self.item:
            return
        if not self.class_names:
            QMessageBox.warning(self, "Нет классов", "Сначала загрузите классы из data.yaml.")
            return

        # 1. Выбор класса
        dialog = NewBoxDialog(self.class_names, 0)
        if dialog.exec_() != QDialog.Accepted:
            return

        cls = dialog.get_selected_index()
        class_name = self.class_names[cls]
        color = self.class_colors[cls]

        # 2. Размеры изображения
        pixmap = self.item.pixmap()
        img_w = pixmap.width()
        img_h = pixmap.height()

        # 3. Размер бокса (например 10% от изображения)
        box_w = img_w * 0.1
        box_h = img_h * 0.1

        # 4. Центр изображения
        x = (img_w - box_w) / 2
        y = (img_h - box_h) / 2

        rect = QRectF(x, y, box_w, box_h)

        # 5. Создаём бокс
        bbox = BBoxItem(
            rect=rect,
            cls=cls,
            img_w=img_w,
            img_h=img_h,
            color=color,
            class_name=class_name
        )

        # self.scene.addItem(bbox)
        # self.box_items.append(bbox)
        self.undo_stack.push(AddBBoxCommand(self.scene, bbox))
        self.box_items.append(bbox)

        # 6. Сразу выделяем его
        bbox.setSelected(True)

    def save_current_labels(self):
        if not self.scene or not self.current_label_path:
            return

        lines = []

        for item in self.scene.items():
            if isinstance(item, BBoxItem):
                lines.append(item.to_yolo())

        # сортируем, чтобы файл был стабильным
        lines.sort()

        os.makedirs(os.path.dirname(self.current_label_path), exist_ok=True)

        with open(self.current_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def current_annotation_lines(self):
        if not self.scene:
            return []
        return sorted(
            item.to_yolo()
            for item in self.scene.items()
            if isinstance(item, BBoxItem)
        )

    def set_current_annotation_lines(self, lines, selected_from=None):
        if not self.scene or not self.item:
            return
        for item in list(self.scene.items()):
            if isinstance(item, BBoxItem):
                self.scene.removeItem(item)
        self.box_items.clear()

        pixmap = self.item.pixmap()
        img_w, img_h = pixmap.width(), pixmap.height()
        for line_index, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls, x_c, y_c, w_rel, h_rel = map(float, parts)
            except ValueError:
                continue
            cls = int(cls)
            rect = QRectF(
                (x_c - w_rel / 2) * img_w,
                (y_c - h_rel / 2) * img_h,
                w_rel * img_w,
                h_rel * img_h,
            )
            color = self.class_colors.get(cls, QColor(255, 0, 0))
            class_name = self.class_names[cls] if 0 <= cls < len(self.class_names) else f"class {cls}"
            bbox = BBoxItem(rect, cls, img_w, img_h, color, class_name)
            self.scene.addItem(bbox)
            self.box_items.append(bbox)
            if selected_from is not None and line_index >= selected_from:
                bbox.setSelected(True)

    def copy_selected_annotations(self):
        if not self.scene:
            return
        try:
            selected = [
                item.to_yolo() for item in self.scene.selectedItems()
                if isinstance(item, BBoxItem)
            ]
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            QMessageBox.critical(
                self, "Ошибка копирования",
                f"Не удалось скопировать выбранные рамки:\n{exc}"
            )
            return
        self.annotation_clipboard = canonical_annotation_lines(selected)
        if self.annotation_clipboard:
            self.statusBar().showMessage(
                f"Скопировано рамок: {len(self.annotation_clipboard)}", 3000
            )
        else:
            self.statusBar().showMessage("Сначала выделите рамки для копирования", 3000)

    def copy_active_context(self):
        focused = QApplication.focusWidget()
        if isinstance(focused, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
            focused.copy()
        elif self.ui.stackedWidget.currentIndex() == 2:
            self.copy_selected_annotations()

    def paste_annotations(self):
        if not self.scene or not self.item:
            return
        if not self.annotation_clipboard:
            self.statusBar().showMessage("Буфер разметки пуст", 3000)
            return
        self.undo_stack.push(
            AppendAnnotationsCommand(
                self,
                self.current_annotation_lines(),
                self.annotation_clipboard,
                "Paste annotations",
            )
        )
        self.statusBar().showMessage(
            f"Вставлено рамок: {len(self.annotation_clipboard)}", 3000
        )

    def paste_active_context(self):
        focused = QApplication.focusWidget()
        if isinstance(focused, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
            focused.paste()
        elif self.ui.stackedWidget.currentIndex() == 2:
            self.paste_annotations()

    def apply_selected_annotations_to_folder(self):
        if self.ui.stackedWidget.currentIndex() != 2 or not self.scene:
            return
        selected_lines = canonical_annotation_lines(
            item.to_yolo() for item in self.scene.selectedItems()
            if isinstance(item, BBoxItem)
        )
        if not selected_lines:
            QMessageBox.information(
                self, "Нет выделенных рамок",
                "Выделите одну или несколько рамок, которые нужно применить ко всей папке."
            )
            return

        dialog = QMessageBox(self)
        dialog.setWindowTitle("Применить ко всей папке")
        dialog.setIcon(QMessageBox.Question)
        dialog.setText(
            f"Добавить выделенные рамки ({len(selected_lines)}) ко всем изображениям "
            f"текущей папки ({len(self.image_items)})?\n\n"
            "Существующая разметка сохранится, точные дубликаты добавлены не будут."
        )
        apply_button = dialog.addButton("Применить", QMessageBox.AcceptRole)
        cancel_button = dialog.addButton("Отмена", QMessageBox.RejectRole)
        dialog.setDefaultButton(cancel_button)
        dialog.exec_()
        if dialog.clickedButton() is not apply_button:
            return

        self.save_current_labels()
        planned_updates = []
        try:
            for _, label_path in self.image_items:
                existing_lines = []
                if os.path.isfile(label_path):
                    with open(label_path, "r", encoding="utf-8") as label_file:
                        existing_lines = [line.rstrip("\r\n") for line in label_file]
                merged_lines = merge_preserving_existing_lines(
                    existing_lines, selected_lines
                )
                if merged_lines != [line.strip() for line in existing_lines if line.strip()]:
                    planned_updates.append((label_path, merged_lines))
        except OSError as exc:
            QMessageBox.critical(
                self, "Ошибка чтения разметки",
                f"Изменения не применены:\n{exc}"
            )
            return

        updated = 0
        try:
            for label_path, merged_lines in planned_updates:
                os.makedirs(os.path.dirname(label_path), exist_ok=True)
                temporary_path = f"{label_path}.marking.tmp"
                with open(temporary_path, "w", encoding="utf-8") as label_file:
                    label_file.write("\n".join(merged_lines))
                os.replace(temporary_path, label_path)
                updated += 1
        except OSError as exc:
            QMessageBox.critical(
                self, "Ошибка записи разметки",
                f"Обновлено файлов: {updated} из {len(planned_updates)}.\nОшибка:\n{exc}"
            )
            return

        self.statusBar().showMessage(
            f"Выделенные рамки добавлены в файлы: {updated}", 5000
        )
        QMessageBox.information(
            self, "Готово",
            f"Обновлено файлов разметки: {updated}.\n"
            f"Без изменений: {len(self.image_items) - updated}."
        )

    def copy_previous_annotations(self):
        if self.ui.stackedWidget.currentIndex() != 2:
            return
        if not self.image_items or self.current_index <= 0:
            QMessageBox.information(
                self, "Нет предыдущего файла", "Для первого изображения предыдущего файла нет."
            )
            return
        previous_label_path = self.image_items[self.current_index - 1][1]
        if not os.path.isfile(previous_label_path):
            QMessageBox.information(
                self, "Нет разметки", "У предыдущего изображения файл разметки отсутствует."
            )
            return
        with open(previous_label_path, "r", encoding="utf-8") as label_file:
            previous_lines = [line.strip() for line in label_file if line.strip()]
        current_lines = self.current_annotation_lines()
        merged_lines = merge_annotation_lines(current_lines, previous_lines)
        if merged_lines == canonical_annotation_lines(current_lines):
            self.statusBar().showMessage("Новых рамок в предыдущем файле нет", 3000)
            return
        self.undo_stack.push(
            ReplaceAnnotationsCommand(
                self,
                current_lines,
                merged_lines,
                "Merge previous annotations",
            )
        )

    def clear_current_annotations(self):
        if self.ui.stackedWidget.currentIndex() != 2:
            return
        if not self.current_annotation_lines():
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Удалить всю разметку")
        dialog.setText("Удалить все рамки с текущего изображения?")
        dialog.setIcon(QMessageBox.Question)
        yes_button = dialog.addButton("Да", QMessageBox.YesRole)
        dialog.addButton("Нет", QMessageBox.NoRole)
        dialog.setDefaultButton(yes_button)
        dialog.exec_()
        if dialog.clickedButton() is yes_button:
            self.undo_stack.push(
                ReplaceAnnotationsCommand(
                    self,
                    self.current_annotation_lines(),
                    [],
                    "Clear annotations",
                )
            )

    def reset_current_labels(self):
        if not self.item:
            return

        reply = QMessageBox.question(
            self,
            "Сброс изменений",
            "Все несохранённые изменения будут потеряны.\nПродолжить?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # просто заново загружаем текущее изображение
        img_path, label_path = self.image_items[self.current_index]
        self.place_img(img_path, label_path)

    def setup_classes_table(self):
        yaml_path = self.ui.lineEdit_yaml_path.text()
        rows = self.load_classes_from_yaml(yaml_path)
        self.classes_model = ClassesTableModel(rows)

        view = self.ui.tableView_class_info
        view.setModel(self.classes_model)

        # Внешний вид
        header = view.horizontalHeader()
        header.setSectionResizeMode(header.Stretch)
        view.verticalHeader().setVisible(False)
        view.setAlternatingRowColors(True)
        view.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.delegate = MergeDelegate(self.get_existing_classes, self.ui.tableView_class_info)
        self.ui.tableView_class_info.setItemDelegateForColumn(3, self.delegate)

    def load_classes_from_yaml(self, yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        rows = []
        for idx, name in enumerate(data["names"]):
            rows.append(ClassRow(class_id=idx, original_name=name))
        return rows

    # Методы для кнопок
    def move_row_up(self):
        idx = self.ui.tableView_class_info.currentIndex()
        if not idx.isValid():
            msg = QMessageBox()
            msg.setWindowTitle("Внимание")
            msg.setText("Необходимо выбрать строку для перемещения!")
            msg.setIcon(QMessageBox.Information)
            msg.exec_()  # показывает окно
            return
        self.classes_model.move_row_up(idx.row())
        self.classes_model.update_ids()  # пересчитываем ID
        self.ui.tableView_class_info.selectRow(max(0, idx.row() - 1))

    def move_row_down(self):
        idx = self.ui.tableView_class_info.currentIndex()
        if not idx.isValid():
            msg = QMessageBox()
            msg.setWindowTitle("Внимание")
            msg.setText("Необходимо выбрать строку для перемещения!")
            msg.setIcon(QMessageBox.Information)
            msg.exec_()  # показывает окно
            return
        self.classes_model.move_row_down(idx.row())
        self.classes_model.update_ids()  # пересчитываем ID
        self.ui.tableView_class_info.selectRow(min(self.classes_model.rowCount() - 1, idx.row() + 1))

    def delete_selected_class(self):
        view = self.ui.tableView_class_info
        index = view.currentIndex()
        if not index.isValid():
            msg = QMessageBox()
            msg.setWindowTitle("Внимание")
            msg.setText("Необходимо выбрать строку для удаления!")
            msg.setIcon(QMessageBox.Information)
            msg.exec_()  # показывает окно
            return  # ничего не выбрано

        row_idx = index.row()
        model = self.classes_model

        if row_idx >= len(model.rows):
            return  # индекс вне диапазона

        row = model.rows[row_idx]

        # можно удалять только новые классы
        if row.original_name:
            QMessageBox.warning(self, "Невозможно удалить",
                                "Можно удалять только новые классы (без старого имени)")
            return

        # удаляем строку через beginRemoveRows/endRemoveRows
        model.beginRemoveRows(QModelIndex(), row_idx, row_idx)
        del model.rows[row_idx]
        model.endRemoveRows()

        # пересчитываем ID
        model.update_ids()

        # обновляем список подсказок делегата
        delegate = view.itemDelegateForColumn(3)
        if delegate:
            delegate.items = [r.original_name for r in model.rows if r.original_name]

        # снимаем выделение, чтобы не оставался некорректный индекс
        view.clearSelection()

    def on_apply_changes_clicked(self):
        self.ui.progressBar_class_info.setValue(0)
        self.ui.progressBar_class_info.setVisible(True)

        dataset_path = self.ui.lineEdit_dataset_path.text()
        yaml_path = self.ui.lineEdit_yaml_path.text()

        self.thread = QThread()
        self.worker = DatasetEditor(
            self.classes_model,
            dataset_path,
            yaml_path
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.apply)
        self.worker.progress.connect(self.ui.progressBar_class_info.setValue)
        self.worker.finished.connect(self.on_editor_finished)
        self.worker.error.connect(self.on_editor_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_editor_finished(self, success: bool):
        self.ui.progressBar_class_info.setVisible(False)
        self.setup_classes_table()

        if success:
            QMessageBox.information(
                self,
                "Готово",
                "Изменения в датасете применены!"
            )

    def on_editor_error(self, message):
        QMessageBox.critical(self, "Ошибка", message)

    def autolabling_log(self, message):
        QMessageBox.information(
            self,
            "Информация",
            message
        )

    # def on_autolabling_finished(self, success: bool):
    #     self.ui.progressBar_autolabling.setVisible(False)

    # def start_autolabling(self):
    #     yaml_path = self.ui.lineEdit_yaml_path.text()
    #     if not os.path.exists(yaml_path):
    #         return
    #     dialog = AutoLabelingDialog(yaml_path)
    #     if dialog.exec_() == QtWidgets.QDialog.Accepted:
    #         selected_classes, conf = dialog.get_settings()
    #         print("Выбранные классы:", selected_classes)
    #         print("Порог уверенности:", conf)
    #
    #         model_path = self.ui.lineEdit_model_path.text()
    #         if not os.path.exists(model_path):
    #             return
    #         self.worker = YoloWorker(model_path)
    #         self.thread = QThread()
    #         self.worker.moveToThread(self.thread)
    #
    #         self.worker.progress.connect(self.ui.progressBar_autolabling.setValue)
    #         self.worker.log.connect(self.autolabling_log)
    #         self.worker.finished.connect(self.on_autolabling_finished)
    #
    #         self.worker.finished.connect(self.thread.quit)
    #         self.worker.finished.connect(self.worker.deleteLater)
    #         self.thread.finished.connect(self.thread.deleteLater)
    #
    #         root_folder = self.ui.lineEdit_dataset_path.text()
    #         if not os.path.exists(root_folder):
    #             return
    #
    #         self.ui.progressBar_autolabling.setVisible(True)
    #
    #         self.thread.started.connect(lambda: self.worker.predict_and_save_labels_recursive(
    #             root_folder=root_folder,
    #             selected_classes=selected_classes,
    #             conf=conf
    #         ))
    #         self.thread.start()

    def start_autolabling(self):
        model_path = self.ui.lineEdit_model_path.text().strip()
        root_folder = self.ui.lineEdit_dataset_path.text()

        if not os.path.isfile(model_path):
            model_path, _ = QFileDialog.getOpenFileName(
                self, "Выберите модель для авторазметки", os.getcwd(),
                "Модели детекции (*.pt *.onnx *.engine *.torchscript);;Все файлы (*)"
            )
            if not model_path:
                return
            self.ui.lineEdit_model_path.setText(model_path)
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Не выбран датасет", "Сначала выберите папку датасета.")
            return
        yaml_path = self.ui.lineEdit_yaml_path.text().strip()
        if not os.path.isfile(yaml_path):
            detected_yaml = find_dataset_yaml(root_folder)
            if detected_yaml:
                yaml_path = str(detected_yaml)
                self.ui.lineEdit_yaml_path.setText(yaml_path)
            else:
                yaml_path, _ = QFileDialog.getOpenFileName(
                    self, "Выберите data.yaml со списком классов",
                    root_folder, "YAML (*.yaml *.yml)"
                )
                if not yaml_path:
                    return
                self.ui.lineEdit_yaml_path.setText(yaml_path)

        dialog = AutoLabelingDialog(yaml_path)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        selected_classes, conf = dialog.get_settings()

        self.yoloWorker = ModelWorker(model_path)

        self.ui.progressBar_autolabling.setValue(0)
        self.ui.progressBar_autolabling.setVisible(True)

        self.yoloWorker.start_autolabeling_process(
            model_path=model_path,
            root_folder=root_folder,
            selected_classes=selected_classes,
            conf=conf
        )

        self.active_process_kind = "autolabel"
        self.timer.start(100)

    def show_train_info(
        self, epoch, epochs_total, percent_in_epoch,
        overall_percent, gpu_mem, using_gpu
    ):
        self.ui.label_epoch_number_value.setText(f"{epoch}/{epochs_total}")
        self.ui.label_gpu_mem_value.setText(
            f"{gpu_mem:.2f} ГБ" if using_gpu else "Не используется"
        )
        self.ui.label_progress_value.setText(f"{percent_in_epoch} %")
        self.ui.progressBar_training.setValue(overall_percent)

    def stop_training(self, completed=False):
        self.timer.stop()

        if self.yoloWorker.process and self.yoloWorker.process.is_alive():
            self.yoloWorker.process.join(timeout=2 if completed else 0)
            if self.yoloWorker.process.is_alive() and not completed:
                self.yoloWorker.process.terminate()
                self.yoloWorker.process.join(timeout=5)

        if self.active_process_kind == "training":
            self.ui.pushButton_start_training.setEnabled(True)
            if completed:
                self.ui.progressBar_training.setValue(100)
                self.ui.label_progress_value.setText("100 %")
            else:
                self.ui.progressBar_training.setVisible(False)
        elif self.active_process_kind == "autolabel":
            self.ui.progressBar_autolabling.setVisible(False)
        self.active_process_kind = None

    def poll_queue(self):
        if not self.yoloWorker or not self.yoloWorker.queue:
            return

        while True:
            try:
                msg = self.yoloWorker.queue.get_nowait()
            except queue.Empty:
                break  # очередь пуста — выходим из цикла

            if msg[0] == "train_info":
                (
                    _, epoch, epochs_total, percent, overall_percent,
                    gpu_mem, using_gpu
                ) = msg
                self.show_train_info(
                    epoch, epochs_total, percent,
                    overall_percent, gpu_mem, using_gpu
                )

            elif msg[0] == "error":
                _, err, tb = msg
                QMessageBox.critical(
                    self, "Ошибка фонового процесса", f"{err}\n\n{tb}"
                )
                self.stop_training()

            elif msg[0] == "log":
                _, text = msg
                self.autolabling_log(text)  # выводим лог в GUI

            elif msg[0] == "progress":
                _, percent = msg
                self.ui.progressBar_autolabling.setValue(percent)

            elif msg[0] == "finished":
                self.stop_training(completed=True)

    def start_training(self):
        model_path = self.ui.lineEdit_model_path.text().strip()
        if not os.path.exists(model_path):
            QMessageBox.warning(
                self, "Не выбрана модель",
                "Выберите файл начальной модели детекции."
            )
            return

        yaml_path = self.ui.lineEdit_yaml_path.text().strip()
        if not os.path.exists(yaml_path):
            QMessageBox.warning(
                self, "Не найден data.yaml",
                "Выберите файл конфигурации обучающего датасета."
            )
            return

        self.yoloWorker = ModelWorker(model_path)

        epochs = self.ui.spinBox_epochs.value()
        using_gpu = self.ui.checkBox_gpu.isChecked()
        self.ui.label_epoch_number.setEnabled(True)
        self.ui.label_epoch_number_value.setEnabled(True)
        self.ui.label_gpu_mem.setEnabled(True)
        self.ui.label_gpu_mem_value.setEnabled(True)
        self.ui.label_progress.setEnabled(True)
        self.ui.label_progress_value.setEnabled(True)
        self.ui.label_epoch_number_value.setText(f"0/{epochs}")
        self.ui.label_gpu_mem_value.setText("Ожидание..." if using_gpu else "Не используется")
        self.ui.label_progress_value.setText("0 %")
        self.ui.progressBar_training.setValue(0)
        self.ui.progressBar_training.setVisible(True)
        self.ui.pushButton_start_training.setEnabled(False)
        self.active_process_kind = "training"

        self.yoloWorker.start_training(
            model_path=model_path,
            dataset_yaml=yaml_path,
            epochs=epochs,
            imgsz=int(self.ui.comboBox.currentText()),
            batch=self.ui.spinBox_batch.value(),
            gpu=self.ui.checkBox_gpu.isChecked()
        )

        self.timer.start(100)  # 10 раз в секунду

    def on_test_finished(self, result):
        self.timer.stop()

        if self.yoloWorker.process and self.yoloWorker.process.is_alive():
            self.yoloWorker.process.terminate()
            self.yoloWorker.process.join(timeout=5)

        self.ui.progressBar_testing.setVisible(False)
        if result:
            QMessageBox.information(
                self,
                "Информация",
                "Тестирование успешно завершено!"
            )
        else:
            QMessageBox.critical(
                self,
                "Ошибка",
                "Ошибка тестирования!"
            )

    def on_testing_log(self, log):
        print(log)

    def start_model_test(self):
        model_path = self.ui.lineEdit_model_path.text()
        if not os.path.exists(model_path):
            return

        dataset_path = self.ui.lineEdit_dataset_path.text()
        if not os.path.exists(dataset_path):
            return

        yaml_path = self.ui.lineEdit_yaml_path.text()
        if not os.path.exists(yaml_path):
            return

        self.load_classes(yaml_path)

        self.yoloWorker = YOLOTestWorker()

        self.yoloWorker.progress.connect(self.ui.progressBar_testing.setValue)
        self.yoloWorker.log.connect(self.on_testing_log)
        # self.yoloWorker.scenario_info.connect(self.on_scenario)
        self.yoloWorker.finished.connect(self.on_test_finished)

        self.ui.progressBar_testing.setValue(0)
        self.ui.progressBar_testing.setVisible(True)

        self.yoloWorker.start_evaluation(
            model_path=model_path,
            test_root=dataset_path,
            class_names=self.class_names,
            imgsz=int(self.ui.comboBox_2.currentText()),
            gpu=self.ui.checkBox_gpu_2.isChecked()
        )

        self.timer = QTimer()
        self.timer.timeout.connect(self.yoloWorker.poll_queue)
        self.timer.start(100)


class AddBBoxCommand(QUndoCommand):
    def __init__(self, scene, bbox):
        super().__init__("Add box")
        self.scene = scene
        self.bbox = bbox

    def redo(self):
        self.scene.addItem(self.bbox)

    def undo(self):
        self.scene.removeItem(self.bbox)

class ReplaceAnnotationsCommand(QUndoCommand):
    def __init__(self, window, old_lines, new_lines, description):
        super().__init__(description)
        self.window = window
        self.old_lines = list(old_lines)
        self.new_lines = list(new_lines)

    def redo(self):
        self.window.set_current_annotation_lines(self.new_lines)

    def undo(self):
        self.window.set_current_annotation_lines(self.old_lines)


class AppendAnnotationsCommand(QUndoCommand):
    def __init__(self, window, old_lines, appended_lines, description):
        super().__init__(description)
        self.window = window
        self.old_lines = canonical_annotation_lines(old_lines)
        self.appended_lines = canonical_annotation_lines(appended_lines)
        self.new_lines = self.old_lines + self.appended_lines

    def redo(self):
        self.window.set_current_annotation_lines(
            self.new_lines, selected_from=len(self.old_lines)
        )

    def undo(self):
        self.window.set_current_annotation_lines(self.old_lines)

class RemoveBBoxCommand(QUndoCommand):
    def __init__(self, scene, bbox):
        super().__init__("Remove box")
        self.scene = scene
        self.bbox = bbox

    def redo(self):
        self.scene.removeItem(self.bbox)

    def undo(self):
        self.scene.addItem(self.bbox)


class ChangeClassCommand(QUndoCommand):
    def __init__(self, bbox, old_cls, old_color, new_cls, new_color, old_name, new_name):
        super().__init__("Change class")
        self.bbox = bbox
        self.old_cls = old_cls
        self.old_color = old_color
        self.new_cls = new_cls
        self.new_color = new_color
        self.old_name = old_name
        self.new_name = new_name

    def undo(self):
        self.bbox.set_class(self.old_name, self.old_color, self.old_cls)

    def redo(self):
        self.bbox.set_class(self.new_name, self.new_color, self.new_cls)


if __name__ == '__main__':
    # Required for multiprocessing workers in a frozen Windows application.
    # Without this guard PyInstaller starts another copy of the GUI instead of
    # dispatching the child process to yolo_train_process/autolabel workers.
    mp.freeze_support()
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "Railways.Marking.Desktop"
            )
        except (AttributeError, OSError):
            pass
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(icon_path("railway.png")))
    apply_application_style(app)
    window = MainWindow()
    window.setWindowIcon(QIcon(icon_path("railway.png")))
    # window.show()
    window.showMaximized()
    sys.exit(app.exec_())
