"""Планирует validation по целым визуальным группам, не разделяя похожие кадры."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_split_balance import BKTree
from plan_split_rebalance import summarize


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def group_vector(records: list[dict], class_count: int) -> np.ndarray:
    vector = np.zeros(1 + class_count * 2, dtype=np.float64)
    vector[0] = len(records)
    for record in records:
        for class_id in set(record["classes"]):
            vector[1 + class_id] += 1
        for class_id in record["classes"]:
            vector[1 + class_count + class_id] += 1
    return vector


def scores(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    weights = np.array([8.0] + [2.5] * 4 + [1.0] * 4)
    relative = (values - target) / np.maximum(1.0, target)
    return np.sum(relative * relative * weights, axis=-1)


def choose_groups(
    vectors: np.ndarray, target: np.ndarray, max_group_size: int
) -> np.ndarray:
    selected = np.zeros(len(vectors), dtype=bool)
    eligible = vectors[:, 0] <= max_group_size
    current = np.zeros(vectors.shape[1], dtype=np.float64)
    current_score = float(scores(current[None, :], target)[0])

    # Последовательно добавляем группу, которая лучше всего закрывает общий дефицит.
    while current[0] < target[0] * 0.97:
        candidates = np.flatnonzero((~selected) & eligible)
        candidate_values = current + vectors[candidates]
        allowed = candidate_values[:, 0] <= target[0] * 1.08
        if not np.any(allowed):
            break
        candidate_scores = scores(candidate_values[allowed], target)
        allowed_indexes = candidates[allowed]
        best_position = int(np.argmin(candidate_scores))
        best = int(allowed_indexes[best_position])
        selected[best] = True
        current += vectors[best]
        current_score = float(candidate_scores[best_position])

    # Улучшаем одиночными добавлениями/удалениями.
    for _ in range(500):
        toggled_values = np.where(
            selected[:, None], current - vectors, current + vectors
        )
        valid_size = eligible & (
            (toggled_values[:, 0] >= target[0] * 0.92)
            & (toggled_values[:, 0] <= target[0] * 1.08)
        )
        if not np.any(valid_size):
            break
        toggle_scores = np.full(len(vectors), np.inf)
        toggle_scores[valid_size] = scores(toggled_values[valid_size], target)
        best = int(np.argmin(toggle_scores))
        if float(toggle_scores[best]) >= current_score - 1e-12:
            break
        if selected[best]:
            current -= vectors[best]
        else:
            current += vectors[best]
        selected[best] = ~selected[best]
        current_score = float(toggle_scores[best])

    # Детерминированный поиск обменов между наиболее перспективными группами.
    selected_indexes = np.flatnonzero(selected)
    unselected_indexes = np.flatnonzero((~selected) & eligible)
    rng = np.random.default_rng(20261104)
    for _ in range(150000):
        left = int(rng.choice(selected_indexes))
        right = int(rng.choice(unselected_indexes))
        candidate = current - vectors[left] + vectors[right]
        if not target[0] * 0.92 <= candidate[0] <= target[0] * 1.08:
            continue
        candidate_score = float(scores(candidate[None, :], target)[0])
        if candidate_score < current_score:
            selected[left] = False
            selected[right] = True
            current = candidate
            current_score = candidate_score
            selected_indexes = np.flatnonzero(selected)
            unselected_indexes = np.flatnonzero((~selected) & eligible)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-ratio", type=float, default=0.125)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument(
        "--max-component-share",
        type=float,
        default=0.15,
        help=(
            "Максимальная доля validation для одной визуальной серии. "
            "Более крупные повторяющиеся серии остаются целиком в train."
        ),
    )
    args = parser.parse_args()

    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    class_names = audit["class_names"]
    records = [
        record for record in audit["records"]
        if record["split"] in {"train", "valid"}
    ]
    union_find = UnionFind(len(records))
    tree = BKTree()
    edge_count = 0
    for index, record in enumerate(records):
        for other_index, phash_distance in tree.query(
            record["phash"], args.threshold
        ):
            other = records[other_index]
            dhash_distance = (record["dhash"] ^ other["dhash"]).bit_count()
            if dhash_distance <= args.threshold + 2:
                union_find.union(index, other_index)
                edge_count += 1
        tree.add(record["phash"], index)
        if (index + 1) % 2000 == 0:
            print(f"Сгруппировано {index + 1}/{len(records)}", flush=True)

    members = defaultdict(list)
    for index, record in enumerate(records):
        members[union_find.find(index)].append(record)
    groups = sorted(members.values(), key=lambda group: (-len(group), group[0]["image"]))
    vectors = np.stack([
        group_vector(group, len(class_names)) for group in groups
    ])
    total = np.sum(vectors, axis=0)
    target = total * args.target_ratio
    max_group_size = max(1, round(target[0] * args.max_component_share))
    selected = choose_groups(vectors, target, max_group_size)

    final_valid = [
        record for index, group in enumerate(groups) if selected[index]
        for record in group
    ]
    final_valid_paths = {record["image"] for record in final_valid}
    final_train = [record for record in records if record["image"] not in final_valid_paths]
    current_train = [record for record in records if record["split"] == "train"]
    current_valid = [record for record in records if record["split"] == "valid"]
    current_valid_paths = {record["image"] for record in current_valid}

    valid_to_train = [
        record for record in current_valid if record["image"] not in final_valid_paths
    ]
    train_to_valid = [
        record for record in current_train if record["image"] in final_valid_paths
    ]
    component_sizes = Counter(len(group) for group in groups)
    plan = {
        "method": "perceptual_components",
        "threshold": args.threshold,
        "target_ratio_train_valid": args.target_ratio,
        "max_validation_component_size": max_group_size,
        "similarity_edge_count": edge_count,
        "component_count": len(groups),
        "largest_components": [len(group) for group in groups[:25]],
        "selected_component_sizes": [
            len(group) for index, group in enumerate(groups) if selected[index]
        ],
        "component_size_histogram": dict(sorted(component_sizes.items())),
        "before": {
            "train": summarize(current_train, class_names),
            "valid": summarize(current_valid, class_names),
        },
        "after": {
            "train": summarize(final_train, class_names),
            "valid": summarize(final_valid, class_names),
        },
        "actions": {
            "valid_to_train": [
                {"image": record["image"], "label": record["label"]}
                for record in valid_to_train
            ],
            "train_to_valid": [
                {"image": record["image"], "label": record["label"]}
                for record in train_to_valid
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "threshold": args.threshold,
        "similarity_edge_count": edge_count,
        "component_count": len(groups),
        "largest_components": plan["largest_components"],
        "max_validation_component_size": max_group_size,
        "largest_selected_components": sorted(
            plan["selected_component_sizes"], reverse=True
        )[:25],
        "before": plan["before"],
        "after": plan["after"],
        "valid_to_train": len(valid_to_train),
        "train_to_valid": len(train_to_valid),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
