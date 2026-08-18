"""Понятный пользователю анализ динамики обучения Ultralytics."""

from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_COLUMNS = ("epoch", "train/box_loss", "val/box_loss")


def _mean(values):
    return sum(values) / len(values)


def _recent_change(values, window):
    """Сглаженное изменение между началом и концом последнего окна, в процентах."""
    recent = values[-window:]
    segment = max(2, len(recent) // 3)
    start = _mean(recent[:segment])
    end = _mean(recent[-segment:])
    percent = (end - start) / max(abs(start), 1e-12) * 100
    return percent, start, end


def load_training_history(results_csv):
    path = Path(results_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("Файл результатов обучения пуст.")
        normalized = {name.strip(): name for name in reader.fieldnames}
        missing = [name for name in REQUIRED_COLUMNS if name not in normalized]
        if missing:
            raise ValueError("В results.csv отсутствуют столбцы: " + ", ".join(missing))
        rows = []
        for source_row in reader:
            try:
                rows.append({name.strip(): float(value) for name, value in source_row.items() if value not in (None, "")})
            except ValueError:
                continue
    return rows


def analyze_training_results(results_csv):
    rows = load_training_history(results_csv)
    if len(rows) < 8:
        return {
            "verdict": "insufficient",
            "title": "Недостаточно данных",
            "summary": "Для надежной оценки нужно хотя бы 8 завершенных эпох.",
            "recommendation": "Продолжите обучение и повторите анализ после накопления истории.",
            "epochs": len(rows),
            "details": [],
        }

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [row["train/box_loss"] for row in rows]
    val_loss = [row["val/box_loss"] for row in rows]
    map_column = "metrics/mAP50-95(B)"
    map_values = [row[map_column] for row in rows] if all(map_column in row for row in rows) else None
    window = min(20, max(5, len(rows) // 5))
    train_change, _, train_recent = _recent_change(train_loss, window)
    val_change, _, val_recent = _recent_change(val_loss, window)
    map_change = map_start = map_recent = best_map = best_map_epoch = None
    if map_values:
        map_change, map_start, map_recent = _recent_change(map_values, window)
        best_map = max(map_values)
        best_map_epoch = epochs[map_values.index(best_map)]

    best_val = min(val_loss)
    best_val_epoch = epochs[val_loss.index(best_val)]
    epochs_since_best_val = epochs[-1] - best_val_epoch
    train_falling = train_change <= -2.0
    val_falling = val_change <= -1.5
    val_rising = val_change >= 2.0 and val_recent > best_val * 1.03
    train_plateau = abs(train_change) < 2.0
    val_plateau = abs(val_change) < 1.5
    map_plateau = map_change is None or abs(map_recent - map_start) < 0.01
    map_low = map_recent is not None and map_recent < 0.35
    generalization_gap = val_recent / max(train_recent, 1e-12)

    if train_falling and val_rising:
        verdict = "overfitting"
        title = "Есть признаки переобучения"
        summary = "Ошибка на train продолжает снижаться, а validation loss растет относительно своего минимума."
        recommendation = (
            f"Не увеличивайте число эпох автоматически. Используйте веса лучшей эпохи ({best_val_epoch}) "
            "и проверьте аугментации, регуляризацию и разнообразие validation-выборки."
        )
    elif train_falling and val_falling:
        verdict = "continue"
        title = "Обучение можно продолжить"
        summary = "Train и validation loss на последних эпохах продолжают снижаться."
        extra_epochs = max(10, min(50, round(len(rows) * 0.2)))
        recommendation = (
            f"Можно добавить примерно {extra_epochs} эпох с ранней остановкой. "
            "После этого повторно проверьте validation loss и mAP."
        )
    elif map_low and (train_plateau or val_plateau) and generalization_gap > 1.25:
        verdict = "check_data"
        title = "Сначала проверьте данные и настройки"
        summary = "Качество остается низким, а снижение ошибок замедлилось или train и validation заметно расходятся."
        recommendation = (
            "Дополнительные эпохи сами по себе вряд ли решат проблему. Проверьте разметку, соответствие ID классов, "
            "баланс классов, размер модели, разрешение изображений и параметры аугментации."
        )
    elif val_plateau and map_plateau:
        verdict = "converged"
        title = "Модель практически сошлась"
        summary = "Validation loss и mAP вышли на плато; заметного улучшения на последних эпохах нет."
        recommendation = (
            f"Сохраните веса лучшей эпохи ({best_val_epoch}) и переходите к тестированию. "
            "Большое увеличение числа эпох, вероятно, не даст заметного выигрыша."
        )
    elif train_falling and not val_rising:
        verdict = "continue_carefully"
        title = "Можно продолжить с контролем validation"
        summary = "Train loss снижается, а validation loss пока не показывает устойчивого ухудшения."
        recommendation = "Добавьте 10–20 эпох с ранней остановкой и следите, не начнет ли validation loss расти."
    else:
        verdict = "review"
        title = "Однозначной тенденции нет"
        summary = "Кривые колеблются или изменяются слишком слабо для уверенного вывода."
        recommendation = "Сравните несколько последних запусков и проверьте разметку и метрики по отдельным классам."

    details = [
        f"Проанализировано эпох: {len(rows)} (окно: последние {window})",
        f"train/box_loss: {train_recent:.4f}, изменение {train_change:+.1f}%",
        f"val/box_loss: {val_recent:.4f}, изменение {val_change:+.1f}%",
        f"Минимальный val/box_loss: {best_val:.4f} на эпохе {best_val_epoch}",
        f"Разрыв val/train: {generalization_gap:.2f}×",
    ]
    if map_values:
        details.extend((
            f"mAP@0.5:0.95: {map_recent:.4f}, изменение {map_change:+.1f}%",
            f"Лучший mAP@0.5:0.95: {best_map:.4f} на эпохе {best_map_epoch}",
        ))
    return {
        "verdict": verdict,
        "title": title,
        "summary": summary,
        "recommendation": recommendation,
        "epochs": len(rows),
        "window": window,
        "details": details,
        "best_val_epoch": best_val_epoch,
        "best_map_epoch": best_map_epoch,
    }


def save_training_analysis(run_dir, analysis):
    path = Path(run_dir) / "Анализ_обучения.txt"
    lines = [analysis["title"], "", analysis["summary"], "", "Рекомендация:", analysis["recommendation"]]
    if analysis.get("details"):
        lines.extend(("", "Основание:", *analysis["details"]))
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
