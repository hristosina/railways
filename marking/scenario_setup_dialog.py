"""Общий редактор правил и подготовка папок тестовых сценариев."""

from collections import Counter
from pathlib import Path

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon

from tools.organize_test_scenarios import (
    apply_plan,
    build_plan,
    infer_scenario_rules,
    normalize_scenario_rules,
)


class ScenarioRulesDialog(QtWidgets.QDialog):
    def __init__(self, source_path, icon_provider, initial_rules=None,
                 output_path=None, parent=None):
        super().__init__(parent)
        self.source_path = str(source_path)
        self.icon_provider = icon_provider
        self._rules = []
        self._output_path = ""
        self.setWindowTitle("Правила распределения по сценариям")
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumSize(820, 500)
        self._build_ui()

        rules = initial_rules
        if rules is None:
            try:
                rules = infer_scenario_rules(source_path)
            except ValueError:
                rules = []
        for rule in rules:
            self.add_rule_row(rule)
        default_output = output_path or Path(source_path).expanduser().resolve() / "test_scenarios"
        self.output_edit.setText(str(default_output))

    def _icon(self, name):
        return QIcon(self.icon_provider(name))

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Задайте сценарии и ключевые слова, встречающиеся в именах файлов. "
            "Если подходят несколько правил, используется первое сверху.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QtWidgets.QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(
            ("Название сценария", "Ключевые слова через запятую", "Имя папки")
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        layout.addWidget(self.table, 1)

        actions = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QToolButton(self)
        self.add_button.setIcon(self._icon("add.png"))
        self.add_button.setToolTip("Добавить правило")
        self.remove_button = QtWidgets.QToolButton(self)
        self.remove_button.setIcon(self._icon("delete.png"))
        self.remove_button.setToolTip("Удалить выбранное правило")
        self.detect_button = QtWidgets.QPushButton("Определить по именам файлов", self)
        self.detect_button.setIcon(self._icon("reset.png"))
        self.detect_button.setToolTip("Заново предложить правила по фактическим префиксам")
        actions.addWidget(self.add_button)
        actions.addWidget(self.remove_button)
        actions.addSpacing(8)
        actions.addWidget(self.detect_button)
        actions.addStretch()
        layout.addLayout(actions)

        output_box = QtWidgets.QGroupBox("Куда создать сортированные папки", self)
        output_layout = QtWidgets.QHBoxLayout(output_box)
        self.output_edit = QtWidgets.QLineEdit(output_box)
        self.output_button = QtWidgets.QToolButton(output_box)
        self.output_button.setIcon(self._icon("search.png"))
        self.output_button.setToolTip("Выбрать выходную папку")
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(self.output_button)
        layout.addWidget(output_box)

        note = QtWidgets.QLabel(
            "Исходные train, val и test не изменяются. По возможности создаются "
            "жёсткие ссылки; если это невозможно, файлы копируются.",
            self,
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Продолжить")
        self.buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("Отмена")
        layout.addWidget(self.buttons)

        self.add_button.clicked.connect(self.add_rule)
        self.remove_button.clicked.connect(self.remove_rule)
        self.detect_button.clicked.connect(self.detect_rules)
        self.output_button.clicked.connect(self.choose_output)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def add_rule_row(self, rule=None):
        rule = dict(rule or {
            "name": "Новый сценарий", "keywords": (), "folder": "scenario"
        })
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(rule.get("name", ""))))
        keywords = rule.get("keywords", ())
        if not isinstance(keywords, str):
            keywords = ", ".join(str(value) for value in keywords)
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(keywords))
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(rule.get("folder", ""))))

    def add_rule(self):
        self.add_rule_row()
        row = self.table.rowCount() - 1
        self.table.setCurrentCell(row, 0)
        self.table.editItem(self.table.item(row, 0))

    def remove_rule(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def detect_rules(self):
        try:
            rules = infer_scenario_rules(self.source_path)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Не удалось определить сценарии", str(exc))
            return
        self.table.setRowCount(0)
        for rule in rules:
            self.add_rule_row(rule)
        if not rules:
            QtWidgets.QMessageBox.information(
                self, "Сценарии не найдены",
                "В именах файлов не удалось выделить префиксы. Добавьте правила вручную."
            )

    def choose_output(self):
        start = self.output_edit.text().strip() or self.source_path
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Выберите папку для сценариев", start
        )
        if folder:
            self.output_edit.setText(folder)

    def _table_rules(self):
        return [
            {
                "name": self.table.item(row, 0).text().strip(),
                "keywords": self.table.item(row, 1).text().strip(),
                "folder": self.table.item(row, 2).text().strip(),
            }
            for row in range(self.table.rowCount())
        ]

    def accept(self):
        try:
            rules = normalize_scenario_rules(self._table_rules())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Проверьте правила", str(exc))
            return
        names = [rule["name"].casefold() for rule in rules]
        folders = [rule["folder"].casefold() for rule in rules]
        if len(names) != len(set(names)):
            QtWidgets.QMessageBox.warning(self, "Проверьте правила", "Названия сценариев не должны повторяться.")
            return
        if len(folders) != len(set(folders)):
            QtWidgets.QMessageBox.warning(self, "Проверьте правила", "Имена выходных папок не должны повторяться.")
            return
        output_text = self.output_edit.text().strip()
        if not output_text:
            QtWidgets.QMessageBox.warning(self, "Не выбрана папка", "Укажите выходную папку для сценариев.")
            return
        self._rules = rules
        self._output_path = str(Path(output_text).expanduser().resolve())
        super().accept()

    def settings(self):
        return list(self._rules), self._output_path


def _ask_confirmation(parent, title, text):
    dialog = QtWidgets.QMessageBox(parent)
    dialog.setWindowTitle(title)
    dialog.setText(text)
    dialog.setIcon(QtWidgets.QMessageBox.Question)
    yes_button = dialog.addButton("Да", QtWidgets.QMessageBox.YesRole)
    no_button = dialog.addButton("Нет", QtWidgets.QMessageBox.NoRole)
    yes_button.setAutoDefault(True)
    yes_button.setDefault(True)
    no_button.setAutoDefault(True)
    dialog.setDefaultButton(yes_button)
    dialog.setEscapeButton(no_button)
    QTimer.singleShot(0, lambda: yes_button.setFocus(Qt.OtherFocusReason))
    dialog.exec_()
    return dialog.clickedButton() is yes_button


def prepare_scenario_folders(parent, source_path, icon_provider,
                             initial_rules=None, output_path=None):
    """Настраивает правила, показывает план и создаёт сортированное представление."""
    dialog = ScenarioRulesDialog(
        source_path, icon_provider, initial_rules=initial_rules,
        output_path=output_path, parent=parent,
    )
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None
    rules, output_path = dialog.settings()
    try:
        output, plan, unknown, missing_labels = build_plan(
            source_path, output_path, scenario_rules=rules
        )
    except ValueError as exc:
        QtWidgets.QMessageBox.warning(parent, "Невозможно подготовить сценарии", str(exc))
        return None
    if not plan:
        QtWidgets.QMessageBox.warning(
            parent, "Сценарии не найдены",
            "Ни одно изображение не подошло под заданные ключевые слова."
        )
        return None

    counts = Counter(destination.parents[1].name for _, destination in plan[::2])
    lines = [
        f"• {rule['name']}: {counts.get(rule['folder'], 0)} изображений"
        for rule in rules if counts.get(rule["folder"], 0)
    ]
    details = (
        "По заданным ключевым словам будут созданы сценарии:\n"
        + "\n".join(lines)
        + f"\n\nОтдельное представление:\n{output}\n\n"
        "Исходные train, val и test не изменятся. Сначала используются жёсткие "
        "ссылки; если они недоступны, файлы будут скопированы."
    )
    if unknown:
        details += f"\n\nНе подошло ни под одно правило: {len(unknown)} изображений."
    if missing_labels:
        details += f"\nНе найдено разметок: {len(missing_labels)} изображений."
    if not _ask_confirmation(parent, "Подготовить сценарии", details):
        return None

    progress = QtWidgets.QProgressDialog(
        "Подготовка папок сценариев…", "", 0, len(plan), parent
    )
    progress.setWindowTitle("Подготовка тестовой выборки")
    progress.setCancelButton(None)
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)

    def update_progress(current, total):
        if current == total or current % 25 == 0:
            progress.setValue(current)
            QtWidgets.QApplication.processEvents()

    try:
        status = apply_plan(
            plan, strategy="hardlink", fallback_to_copy=True,
            progress_callback=update_progress,
        )
    except (OSError, FileExistsError) as exc:
        QtWidgets.QMessageBox.critical(parent, "Не удалось подготовить сценарии", str(exc))
        return None
    finally:
        progress.close()

    definitions = [
        {
            "name": rule["name"],
            "path": str((Path(output) / rule["folder"]).resolve()),
            "prefix": None,
        }
        for rule in rules if counts.get(rule["folder"], 0)
    ]
    created = status["created"] + status["copied"]
    QtWidgets.QMessageBox.information(
        parent, "Сценарии подготовлены",
        f"Готово. Создано файлов: {created}; уже существовало: {status['skipped']}.\n\n{output}"
    )
    return definitions
