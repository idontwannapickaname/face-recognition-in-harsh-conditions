import os
import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


def measure_brightness(img_path: str) -> float | None:
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    return mean_val


def scan_brightness(root: str = "data") -> tuple[list[tuple[str, float]], dict[str, list[float]]]:
    brightness_list: list[tuple[str, float]] = []
    per_folder_stats: dict[str, list[float]] = defaultdict(list)

    for folder, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(folder, f)
                mean_val = measure_brightness(path)
                if mean_val is not None:
                    brightness_list.append((path, mean_val))
                    folder_name = os.path.basename(folder)
                    per_folder_stats[folder_name].append(mean_val)
    return brightness_list, per_folder_stats


def build_image_dict(root: str = "data", split_by_person: bool = False) -> Dict[str, List[str]]:
    all_image_dict: Dict[str, List[str]] = defaultdict(list)

    for split in os.listdir(root):
        split_path = os.path.join(root, split)
        if not os.path.isdir(split_path):
            continue
        for person in os.listdir(split_path):
            person_path = os.path.join(split_path, person)
            if not os.path.isdir(person_path):
                continue
            key = f"{split}/{person}" if split_by_person else f"{split}"
            for f in os.listdir(person_path):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_image_dict[key].append(os.path.join(person_path, f))
    return all_image_dict


def load_image_paths(trn_split_list: List[str], tst_split_list: List[str], root: str = "data") -> Tuple[
    List[str], List[int], List[str], List[int], Dict[str, int]
]:
    trn_paths: List[str] = []
    trn_labels: List[int] = []
    tst_paths: List[str] = []
    tst_labels: List[int] = []

    name_id_map: Dict[str, int] = {}
    next_id = 0

    for split in trn_split_list:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for person in os.listdir(split_dir):
            person_dir = os.path.join(split_dir, person)
            if not os.path.isdir(person_dir):
                continue
            if person not in name_id_map:
                name_id_map[person] = next_id
                next_id += 1
            for f in os.listdir(person_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(person_dir, f)
                    trn_paths.append(full_path)
                    trn_labels.append(name_id_map[person])

    for split in tst_split_list:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for person in os.listdir(split_dir):
            person_dir = os.path.join(split_dir, person)
            if not os.path.isdir(person_dir):
                continue
            if person not in name_id_map:
                name_id_map[person] = next_id
                next_id += 1
            for f in os.listdir(person_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(person_dir, f)
                    tst_paths.append(full_path)
                    tst_labels.append(name_id_map[person])

    return trn_paths, trn_labels, tst_paths, tst_labels, name_id_map


def split_low_light(low_paths: List[str], low_labels: List[int], pct: float = 0.1, seed: int = 43):
    N = len(low_paths)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    test_size = int(N * pct)
    val_size = int(N * pct)

    test_idx = idx[:test_size]
    val_idx = idx[test_size:test_size + val_size]
    train_idx = idx[test_size + val_size:]

    low_train_paths = [low_paths[i] for i in train_idx]
    low_train_labels = [low_labels[i] for i in train_idx]

    low_val_paths = [low_paths[i] for i in val_idx]
    low_val_labels = [low_labels[i] for i in val_idx]

    low_test_paths = [low_paths[i] for i in test_idx]
    low_test_labels = [low_labels[i] for i in test_idx]

    return (
        low_train_paths, low_train_labels,
        low_val_paths, low_val_labels,
        low_test_paths, low_test_labels
    )
