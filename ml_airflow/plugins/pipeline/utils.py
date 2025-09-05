from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

from common.io_utils import read_events, read_item_properties, read_category_tree

def load_raw_data(data_dir: str):
    events = read_events(data_dir)
    props = read_item_properties(data_dir)
    cats = read_category_tree(data_dir)
    return events, props, cats


def basic_filters(df: pd.DataFrame, min_user_inter: int, min_item_inter: int) -> pd.DataFrame:
    """Отфильтровать редких пользователей/товары для устойчивого обучения."""
    df = df.copy()
    # по событиям
    user_cnt = df.groupby("visitorid")["itemid"].count()
    good_users = set(user_cnt[user_cnt >= min_user_inter].index)
    df = df[df["visitorid"].isin(good_users)]
    item_cnt = df.groupby("itemid")["visitorid"].count()
    good_items = set(item_cnt[item_cnt >= min_item_inter].index)
    df = df[df["itemid"].isin(good_items)]
    return df

def downsample_users(df: pd.DataFrame, frac: float, seed: int = 42) -> pd.DataFrame:
    if frac <= 0 or frac >= 1:
        return df
    users = df["visitorid"].drop_duplicates().sample(frac=frac, random_state=seed)
    return df[df["visitorid"].isin(users)]

def temporal_split_per_user(df: pd.DataFrame, val_last_n: int = 1, test_last_n: int = 1):
    """Разделить по времени: у каждого пользователя последние N -> валидация/тест."""
    df = df.sort_values(["visitorid", "timestamp"])
    splits = []
    for uid, g in df.groupby("visitorid"):
        g = g.sort_values("timestamp")
        if len(g) < (val_last_n + test_last_n + 1):
            # всё в train
            g = g.assign(split="train")
        else:
            test_idx = g.index[-test_last_n:]
            val_idx = g.index[-(test_last_n + val_last_n):-test_last_n]
            g = g.assign(split="train")
            g.loc[val_idx, "split"] = "val"
            g.loc[test_idx, "split"] = "test"
        splits.append(g)
    out = pd.concat(splits, ignore_index=False)
    return out

def build_mappings(df: pd.DataFrame):
    """Создать маппинги id<->index для пользователей и товаров."""
    users = df["visitorid"].drop_duplicates().sort_values().tolist()
    items = df["itemid"].drop_duplicates().sort_values().tolist()
    user2idx = {u:i for i,u in enumerate(users)}
    idx2user = {i:u for u,i in user2idx.items()}
    item2idx = {it:i for i,it in enumerate(items)}
    idx2item = {i:it for it,i in item2idx.items()}
    return user2idx, idx2user, item2idx, idx2item

def build_interaction_matrix(df: pd.DataFrame, user2idx: Dict, item2idx: Dict, weights: Dict[str, float]):
    """Построить разреженную матрицу взаимодействий для implicit ALS."""
    from scipy.sparse import coo_matrix
    tmp = df.copy()
    tmp["w"] = tmp["event"].map(lambda e: float(weights.get(e, 0.0)))
    tmp = tmp[tmp["w"] > 0]
    rows = tmp["visitorid"].map(user2idx).values
    cols = tmp["itemid"].map(item2idx).values
    data = tmp["w"].values
    mat = coo_matrix((data, (rows, cols)), shape=(len(user2idx), len(item2idx))).tocsr()
    return mat

def build_user_targets(df_val_or_test: pd.DataFrame, item2idx: Dict) -> Dict[int, set]:
    """Построить таргеты (множество item_idx) для каждой user_idx из разметки (val/test)."""
    g = df_val_or_test.groupby("visitorid")["itemid"].apply(list)
    targets = {}
    for uid, items in g.items():
        # перевод в индексы, игнорируя неизвестные
        idxs = set([item2idx[it] for it in items if it in item2idx])
        targets[uid] = idxs
    return targets

def popularity_top_items(train_df: pd.DataFrame, topk: int = 1000) -> List[int]:
    return (
        train_df.groupby("itemid")["visitorid"]
        .count().sort_values(ascending=False)
        .head(topk).index.tolist()
    )