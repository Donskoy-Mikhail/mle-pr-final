import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict

def read_events(data_dir: str) -> pd.DataFrame:
    candidates = [
        os.path.join(data_dir, "events.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    df = pd.read_csv(path)
    return df

def read_item_properties(data_dir: str) -> pd.DataFrame:
    parts = []
    for name in ["item_properties_part1.csv", "item_properties_part2.csv"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
    if not parts:
        return pd.DataFrame(columns=["timestamp", "itemid", "property", "value"])
    df = pd.concat(parts, ignore_index=True)
    return df

def read_category_tree(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "category_tree.csv")
    df = pd.read_csv(path)
    return df

def write_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) 