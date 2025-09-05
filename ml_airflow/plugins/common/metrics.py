from __future__ import annotations
import numpy as np
from typing import List, Dict, Iterable

def recall_at_k(preds: Iterable[List[int]], targets: Iterable[set], k: int) -> float:
    """Recall@K: доля целевых товаров, попавших в топ-K рекомендаций."""
    hits = 0
    total = 0
    for recs, tgt in zip(preds, targets):
        recs_k = recs[:k]
        hits += len(set(recs_k) & tgt)
        total += len(tgt)
    if total == 0:
        return 0.0
    return hits / total

def apk(actual: List[int], predicted: List[int], k: int) -> float:
    """Average Precision at K для одного случая."""
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def map_at_k(preds: Iterable[List[int]], targets: Iterable[set], k: int) -> float:
    """Среднее AP@K по множеству пользователей."""
    scores = []
    for recs, tgt in zip(preds, targets):
        scores.append(apk(list(tgt), list(recs), k))
    return float(np.mean(scores)) if scores else 0.0

def ndcg_at_k(preds: Iterable[List[int]], targets: Iterable[set], k: int) -> float:
    """NDCG@K с бинарной релевантностью."""
    def dcg(recs, tgt, k):
        s = 0.0
        for i, p in enumerate(recs[:k]):
            rel = 1.0 if p in tgt else 0.0
            s += rel / np.log2(i + 2.0)
        return s
    dcs = []
    for recs, tgt in zip(preds, targets):
        ideal = min(len(tgt), k)
        idcg = sum([1.0 / np.log2(i + 2.0) for i in range(ideal)]) if ideal > 0 else 1.0
        d = dcg(recs, tgt, k)
        dcs.append(d / idcg if idcg > 0 else 0.0)
    return float(np.mean(dcs)) if dcs else 0.0