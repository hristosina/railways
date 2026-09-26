"""Диалог настройки сценариев и файла отчета о датасете."""

from pathlib import Path

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from dataset_statistics import discover_dataset_scenarios
from evaluation_report import resolve_scenario_directory
from scenario_setup_dialog import prepare_scenario_folders


class DatasetStatisticsDialog(QtWidgets.QDialog):
    def __init__(self, dataset_path, yaml_path, icon_provider, parent=None):
        super().__init__(parent)
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.yaml_path = str(yaml_path)
        self.icon_provider = icon_provider
        self._scenario_definitions = []
        self._output_path = ""

        self.setWindowTitle("Настройка статистики датасета")
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumSize(820, 520)
        self._build_ui()
        self.restore_automatic_scenarios(show_errors=False)

    def _icon(self, filename):
        return QIcon(self.icon_provider(filename))

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Проверьте сценарии, которые попадут в отдельный лист Excel. "
            "Найденные папки и префиксы имен файлов подставляются автоматически, "
            "но список можно изменить.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        scenarios_box = QtWidgets.QGroupBox("Сценарии тестовой выборки", self)
        scenarios_layout = QtWidgets.QVBoxLayout(scenarios_box)
        self.table = QtWidgets.QTableWidget(0, 3, scenarios_box)
        self.table.setHorizontalHeaderLabels(("Название в отчете", "Источник", ""))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        header.resizeSection(2, 46)
        scenarios_layout.addWidget(self.table)

        actions = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QToolButton(scenarios_box)
        self.add_button.setIcon(self._icon("add.png"))
        self.add_button.setToolTip("Добавить сценарий и выбрать его папку")
        self.remove_button = QtWidgets.QToolButton(scenarios_box)
        self.remove_button.setIcon(self._icon("delete.png"))
        self.remove_button.setToolTip("Удалить выбранный сценарий из отчета")
        self.detect_button = QtWidgets.QPushButton("Найти автоматически", scenarios_box)
        self.detect_button.setIcon(self._icon("reset.png"))
        self.detect_button.setToolTip(
            "Заново найти все папки images/labels или группы по префиксам имен файлов"
        )
        self.prepare_button = QtWidgets.QPushButton(
            "Создать сортированные папки по ключевым словам", scenarios_box
        )
        self.prepare_button.setIcon(self._icon("apply.png"))
        self.prepare_button.setToolTip(
            "Настроить названия и ключевые слова, затем создать test_scenarios/images/labels"
        )
        actions.addWidget(self.add_button)
        actions.addWidget(self.remove_button)
        actions.addSpacing(8)
        actions.addWidget(self.detect_button)
        actions.addWidget(self.prepare_button)
        actions.addStretch()
        scenarios_layout.addLayout(actions)
        layout.addWidget(scenarios_box, 1)

        hint = QtWidgets.QLabel(
            "Если сценарная разбивка не нужна, удалите все строки. "
            "Для плоского test после Roboflow источник показывается как префикс имени файла.",
            scenarios_box,
        )
        hint.setWordWrap(True)
        scenarios_layout.addWidget(hint)

        output_box = QtWidgets.QGroupBox("Файл отчета", self)
        output_layout = QtWidgets.QHBoxLayout(output_box)
        self.output_edit = QtWidgets.QLineEdit(output_box)
        self.output_edit.setText(str(self.dataset_path / "Статистика_датасета.xlsx"))
        self.output_edit.setPlaceholderText("Полный путь и имя файла .xlsx")
        self.output_button = QtWidgets.QToolButton(output_box)
        self.output_button.setIcon(self._icon("search.png"))
        self.output_button.setToolTip("Выбрать папку и имя Excel-файла")
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(self.output_button)
        layout.addWidget(output_box)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Сформировать отчет")
        self.buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("Отмена")
        layout.addWidget(self.buttons)

        self.add_button.clicked.connect(self.add_scenario)
        self.remove_button.clicked.connect(self.remove_scenario)
        self.detect_button.clicked.connect(self.restore_automatic_scenarios)
        self.prepare_button.clicked.connect(self.prepare_scenario_directories)
        self.output_button.clicked.connect(self.choose_output_path)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def _source_text(self, definition):
        path = str(definition.get("path", ""))
        prefix = definition.get("prefix")
        if prefix:
            return f"По именам файлов: {prefix}_…  —  {path}"
        return path

    def add_scenario_row(self, definition=None):
        definition = dict(definition or {"name": "Новый сценарий", "path": "", "prefix": None})
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(definition.get("name", ""))))
        source_item = QtWidgets.QTableWidgetItem(self._source_text(definition))
        source_item.setFlags(source_item.flags() & ~Qt.ItemIsEditable)
        source_item.setData(Qt.UserRole, definition)
        self.table.setItem(row, 1, source_item)
        choose_button = QtWidgets.QToolButton(self.table)
        choose_button.setIcon(self._icon("search.png"))
        choose_button.setToolTip("Выбрать папку с подпапками images и labels")
        choose_button.clicked.connect(
            lambda _checked=False, button=choose_button: self.choose_scenario_folder(button)
        )
        self.table.setCellWidget(row, 2, choose_button)

    def add_scenario(self):
        self.add_scenario_row()
        row = self.table.rowCount() - 1
        self.table.setCurrentCell(row, 0)
        self.table.editItem(self.table.item(row, 0))

    def remove_scenario(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def choose_scenario_folder(self, button):
        row = next(
            (index for index in range(self.table.rowCount())
             if self.table.cellWidget(index, 2) is button),
            -1,
        )
        if row < 0:
            return
        definition = self.table.item(row, 1).data(Qt.UserRole) or {}
        start = str(definition.get("path") or self.dataset_path)
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Выберите папку сценария", start
        )
        if not folder:
            return
        try:
            resolved = resolve_scenario_directory(folder)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Неверная папка сценария", str(exc))
            return
        definition = {
            "name": self.table.item(row, 0).text().strip(),
            "path": str(resolved),
            "prefix": None,
        }
        item = self.table.item(row, 1)
        item.setText(str(resolved))
        item.setData(Qt.UserRole, definition)

    def restore_automatic_scenarios(self, _checked=False, show_errors=True):
        try:
            definitions = discover_dataset_scenarios(
                self.dataset_path, yaml_path=self.yaml_path
            )
        except ValueError as exc:
            if show_errors:
                QtWidgets.QMessageBox.warning(self, "Сценарии не найдены", str(exc))
            return
        self.table.setRowCount(0)
        for definition in definitions:
            self.add_scenario_row(definition)
        if show_errors and not definitions:
            QtWidgets.QMessageBox.information(
                self,
                "Сценарии не найдены",
                "Подходящие папки и префиксы имен файлов не обнаружены. "
                "Сценарии можно добавить вручную.",
            )

    def choose_output_path(self):
        current = Path(self.output_edit.text().strip() or self.dataset_path)
        if current.suffix.lower() != ".xlsx":
            current = current / "Статистика_датасета.xlsx"
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Сохранить статистику датасета",
            str(current),
            "Книга Excel (*.xlsx)",
        )
        if path:
            if Path(path).suffix.lower() != ".xlsx":
                path += ".xlsx"
            self.output_edit.setText(path)

    def prepare_scenario_directories(self):
        initial_rules = []
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text().strip()
            definition = self.table.item(row, 1).data(Qt.UserRole) or {}
            prefix = definition.get("prefix")
            if prefix:
                initial_rules.append({
                    "name": name,
                    "keywords": (prefix,),
                    "folder": prefix,
                })
        definitions = prepare_scenario_folders(
            self,
            self.dataset_path,
            self.icon_provider,
            initial_rules=initial_rules or None,
            output_path=self.dataset_path / "test_scenarios",
        )
        if definitions is None:
            return
        self.table.setRowCount(0)
        for definition in definitions:
            self.add_scenario_row(definition)

    def _validated_values(self):
        definitions = []
        errors = []
        used_names = set()
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text().strip()
            source_item = self.table.item(row, 1)
            definition = dict(source_item.data(Qt.UserRole) or {})
            if not name:
                errors.append(f"Строка {row + 1}: укажите название сценария.")
                continue
            if name.casefold() in used_names:
                errors.append(f"Название сценария «{name}» повторяется.")
                continue
            used_names.add(name.casefold())
            source_path = str(definition.get("path", "")).strip()
            if not source_path:
                errors.append(f"Сценарий «{name}»: выберите папку.")
                continue
            if not Path(source_path).is_dir():
                errors.append(f"Сценарий «{name}»: папка не найдена: {source_path}")
                continue
            definition["name"] = name
            definitions.append(definition)

        output_text = self.output_edit.text().strip()
        if not output_text:
            errors.append("Укажите путь и имя Excel-файла.")
            output_path = None
        else:
            output_path = Path(output_text).expanduser()
            if output_path.suffix.lower() != ".xlsx":
                output_path = output_path.with_suffix(".xlsx")
        return definitions, output_path, errors

    def accept(self):
        definitions, output_path, errors = self._validated_values()
        if errors:
            QtWidgets.QMessageBox.warning(
                self,
                "Проверьте настройки отчета",
                "Не все параметры заполнены корректно:\n\n"
                + "\n".join(f"• {error}" for error in errors),
            )
            return
        self._scenario_definitions = definitions
        self._output_path = str(output_path.resolve())
        self.output_edit.setText(self._output_path)
        super().accept()

    def report_settings(self):
        return list(self._scenario_definitions), self._output_path
