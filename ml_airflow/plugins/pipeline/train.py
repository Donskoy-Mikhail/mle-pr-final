from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

from pipeline.utils import (
    load_raw_data, basic_filters, downsample_users, temporal_split_per_user,
    build_mappings, build_interaction_matrix, build_user_targets, popularity_top_items
)
from common.io_utils import write_json

@dataclass
class ALSConfig:
    factors: int = 64
    regularization: float = 0.01
    iterations: int = 20
    alpha: float = 40.0
    use_bm25: bool = True
    filter_seen: bool = True
    topk: int = 20

class ALSModelWrapper:
    """Лёгкая обёртка поверх implicit.ALS. Для сервиса используем numpy-дот-продукты без implicit."""
    def __init__(self, user_factors: np.ndarray, item_factors: np.ndarray):
        self.user_factors = user_factors.astype(np.float32)
        self.item_factors = item_factors.astype(np.float32)

    def recommend_user_idx(self, user_idx: int, seen_items: set | None = None, topk: int = 20) -> List[int]:
        u = self.user_factors[user_idx]
        scores = self.item_factors @ u  
        if seen_items:
            for it in seen_items:
                if 0 <= it < scores.shape[0]:
                    scores[it] = -1e9
        topk = min(topk, scores.shape[0])
        idxs = np.argpartition(-scores, topk - 1)[:topk]
        idxs = idxs[np.argsort(-scores[idxs])]
        return idxs.tolist()

def _fit_als(train_matrix, cfg: ALSConfig, seed: int = 42) -> ALSModelWrapper:
    from implicit.als import AlternatingLeastSquares
    from implicit.nearest_neighbours import bm25_weight, tfidf_weight
    mat = train_matrix
    if cfg.use_bm25:
        mat = bm25_weight(mat).tocsr()
    else:
        mat = tfidf_weight(mat).tocsr()

    item_user = mat.T.tocsr()

    model = AlternatingLeastSquares(
        factors=cfg.factors,
        regularization=cfg.regularization,
        iterations=cfg.iterations,
        random_state=seed
    )
    model.fit(item_user * cfg.alpha)

    user_factors = model.user_factors  
    item_factors = model.user_factors.copy()  
    user_factors = model.item_factors.copy() 
    return ALSModelWrapper(user_factors=user_factors, item_factors=item_factors)

def train_pipeline(config, events, props, cats, dir_to_save) -> Dict:
    seed = int(config.get("seed", 42))

    # Фильтрация и даунсемплинг
    data_cfg = config.get("data", {})
    events = basic_filters(events,
                           min_user_inter=int(data_cfg.get("min_user_interactions", 3)),
                           min_item_inter=int(data_cfg.get("min_item_interactions", 5)))
    frac = float(data_cfg.get("downsample_users", 0.0))
    if frac and frac > 0:
        events = downsample_users(events, frac=frac, seed=seed)

    # плит
    split_cfg = config.get("split", {})
    events = temporal_split_per_user(events,
                                     val_last_n=int(split_cfg.get("val_last_n", 1)),
                                     test_last_n=int(split_cfg.get("test_last_n", 1)))
    train_df = events[events["split"] == "train"].copy()
    val_df = events[events["split"] == "val"].copy()
    test_df = events[events["split"] == "test"].copy()

    # Маппинги и матрица взаимодействий
    user2idx, idx2user, item2idx, idx2item = build_mappings(train_df)
    weights = config.get("weights", {"view": 1.0, "addtocart": 5.0, "transaction": 10.0})
    from scipy.sparse import csr_matrix
    train_mat = build_interaction_matrix(train_df, user2idx, item2idx, weights)

    # Таргеты для валидации/теста
    val_targets = build_user_targets(val_df, item2idx)
    test_targets = build_user_targets(test_df, item2idx)

    # Фоллбек популярности
    popular_items = popularity_top_items(train_df, topk=1000)
    popular_item_idxs = [item2idx[it] for it in popular_items if it in item2idx]

    als_cfg = ALSConfig(**config.get("als", {}))
    model = _fit_als(train_mat, als_cfg, seed=seed)

    # Оценка на валидации

    val_users = [u for u in val_targets.keys() if u in user2idx]
    preds = []
    tgts = []
    for uid in val_users:
        uidx = user2idx[uid]
        seen = set(train_mat[uidx].indices) if als_cfg.filter_seen else set()
        recs = model.recommend_user_idx(uidx, seen_items=seen, topk=als_cfg.topk)
        preds.append(recs)
        tgts.append(val_targets[uid])

    from common.metrics import recall_at_k, map_at_k, ndcg_at_k
    metrics = {}
    for k in [5, 10, 20]:
        metrics[f"recall@{k}"] = recall_at_k(preds, tgts, k)
        metrics[f"map@{k}"] = map_at_k(preds, tgts, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(preds, tgts, k)

    # сохранить модель, артефакты
    save_model_artifacts(model, user2idx, idx2user, item2idx, idx2item, popular_item_idxs, dir_to_save)

    return {
        "metrics": metrics,
        "n_users": len(user2idx),
        "n_items": len(item2idx)
    }

def save_model_artifacts(model: ALSModelWrapper,
                         user2idx: dict, idx2user: dict,
                         item2idx: dict, idx2item: dict,
                         popular_item_idxs: list,
                         out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # факторы
    np.save(os.path.join(out_dir, "user_factors.npy"), model.user_factors)
    np.save(os.path.join(out_dir, "item_factors.npy"), model.item_factors)
    # маппинги
    write_json(os.path.join(out_dir, "user2idx.json"), user2idx)
    write_json(os.path.join(out_dir, "idx2user.json"), {str(k): int(v) for k,v in idx2user.items()})
    write_json(os.path.join(out_dir, "item2idx.json"), item2idx)
    write_json(os.path.join(out_dir, "idx2item.json"), {str(k): int(v) for k,v in idx2item.items()})
    # фоллбек популярности (в индексах)
    write_json(os.path.join(out_dir, "popular_item_idxs.json"), {"popular": popular_item_idxs})

def load_model_for_serving(model_dir: str) -> ALSModelWrapper:
    user_factors = np.load(os.path.join(model_dir, "user_factors.npy"))
    item_factors = np.load(os.path.join(model_dir, "item_factors.npy"))
    return ALSModelWrapper(user_factors=user_factors, item_factors=item_factors)