"""Small, UI-independent helpers for YOLO annotation lines."""


def canonical_annotation_line(line):
    parts = str(line).strip().split()
    if len(parts) != 5:
        return None
    try:
        class_value, x, y, width, height = map(float, parts)
    except ValueError:
        return None
    class_id = int(class_value)
    if class_value != class_id:
        return None
    values = (x, y, width, height)
    if not all(0.0 <= value <= 1.0 for value in values):
        return None
    return f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}"


def canonical_annotation_lines(lines):
    result = []
    for line in lines:
        canonical = canonical_annotation_line(line)
        if canonical is not None:
            result.append(canonical)
    return result


def merge_annotation_lines(current_lines, additional_lines):
    """Append new, valid annotations without removing or duplicating existing ones."""
    result = canonical_annotation_lines(current_lines)
    known = set(result)
    for line in canonical_annotation_lines(additional_lines):
        if line not in known:
            result.append(line)
            known.add(line)
    return result


def merge_preserving_existing_lines(existing_lines, additional_lines):
    """Append unique annotations without rewriting or dropping existing text."""
    result = [str(line).strip() for line in existing_lines if str(line).strip()]
    known = {
        canonical
        for canonical in (canonical_annotation_line(line) for line in result)
        if canonical is not None
    }
    for line in canonical_annotation_lines(additional_lines):
        if line not in known:
            result.append(line)
            known.add(line)
    return result
