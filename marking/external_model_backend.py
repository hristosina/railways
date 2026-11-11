"""Запуск тяжелых backend-ов в изолированном Python 3.10 окружении."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _locate_backend_project() -> Path:
    """Находит sidecar как при запуске из исходников, так и из dist/marking."""
    candidates = []
    configured = os.environ.get("MARKING_MODEL_BACKENDS")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path(__file__).resolve().parent)
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()
        candidates.extend((
            executable.parent,
            executable.parent / "model_backends",
            executable.parents[2] if len(executable.parents) > 2 else executable.parent,
        ))
    for candidate in candidates:
        if (candidate / "venv310" / "Scripts" / "python.exe").is_file() and (
            candidate / "tools" / "model_backend_sidecar.py"
        ).is_file():
            return candidate.resolve()
    return candidates[0].resolve()


PROJECT_DIR = _locate_backend_project()
EXTERNAL_PYTHON = PROJECT_DIR / "venv310" / "Scripts" / "python.exe"
SIDECAR = PROJECT_DIR / "tools" / "model_backend_sidecar.py"
EXTERNAL_BACKENDS = {"official-rtdetr-r18", "official-rtdetrv2-s", "rfdetr-nano"}
EVENT_PREFIX = "MARKING_EVENT "


def external_backend_ready(backend_id: str) -> tuple[bool, str]:
    if backend_id not in EXTERNAL_BACKENDS:
        return False, f"Неизвестный внешний backend: {backend_id}."
    if not EXTERNAL_PYTHON.is_file():
        return False, "Не найдено окружение venv310. Запустите install_model_backends.bat."
    if not SIDECAR.is_file():
        return False, "Не найден служебный адаптер model_backend_sidecar.py."
    if backend_id.startswith("official-rtdetr"):
        repository = PROJECT_DIR / "third_party" / "RT-DETR"
        if not repository.is_dir():
            return False, "Не найден официальный репозиторий third_party/RT-DETR."
    return True, "Backend установлен и готов к работе."


def run_external_backend(queue, action: str, config: dict, result_state: dict | None = None) -> bool:
    """Запускает sidecar и транслирует его JSON-события в multiprocessing.Queue."""
    ready, message = external_backend_ready(config["backend_id"])
    if not ready:
        raise RuntimeError(message)
    descriptor = None
    try:
        descriptor = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="marking_backend_",
            encoding="utf-8", delete=False,
        )
        json.dump(config, descriptor, ensure_ascii=False)
        descriptor.close()
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            [str(EXTERNAL_PYTHON), str(SIDECAR), action, descriptor.name],
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        backend_error = None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if line.startswith(EVENT_PREFIX):
                try:
                    payload = json.loads(line[len(EVENT_PREFIX):])
                    if result_state is not None and payload:
                        if payload[0] == "report" and len(payload) > 1:
                            result_state["report_path"] = payload[1]
                        elif payload[0] == "error" and len(payload) > 1:
                            result_state["error"] = payload[1]
                    if payload and payload[0] == "error":
                        backend_error = str(payload[1]) if len(payload) > 1 else "Неизвестная ошибка."
                        details = payload[2] if len(payload) > 2 else ""
                        queue.put(("log", f"Ошибка внешнего backend-а: {payload[1]}\n{details}"))
                    else:
                        queue.put(tuple(payload))
                except (ValueError, TypeError):
                    queue.put(("log", line))
            elif line:
                queue.put(("log", line))
        exit_code = process.wait()
        if exit_code != 0:
            if backend_error:
                # В popup передаем содержательную ошибку sidecar-а. Полный
                # traceback уже отправлен в журнал GUI и иначе дублировался бы
                # еще одним внешним traceback-ом.
                raise RuntimeError(backend_error)
            raise RuntimeError(
                f"Внешний backend завершился с кодом {exit_code}. "
                "Подробности сохранены в журнале запуска."
            )
        return True
    finally:
        if descriptor is not None:
            try:
                Path(descriptor.name).unlink(missing_ok=True)
            except OSError:
                pass
