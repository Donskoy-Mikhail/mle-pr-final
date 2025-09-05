from __future__ import annotations
import os
import json
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import Response


# Модельные артефакты (ленивая загрузка)
MODEL_DIR = "models"
_user_factors = None
_item_factors = None
_user2idx: Dict[str, int] = {}
_idx2item: Dict[int, int] = {}
_popular_item_idxs: List[int] = []

def load_artifacts(model_dir: str = MODEL_DIR):
    global _user_factors, _item_factors, _user2idx, _idx2item, _popular_item_idxs
    uf = os.path.join(model_dir, "user_factors.npy")
    
    _user_factors = np.load(uf).astype(np.float32)
    _item_factors = np.load(os.path.join(model_dir, "item_factors.npy")).astype(np.float32)
    with open(os.path.join(model_dir, "user2idx.json"), "r") as f:
        _user2idx = json.load(f)
        _user2idx = {str(k): int(v) for k,v in _user2idx.items()}
    with open(os.path.join(model_dir, "idx2item.json"), "r") as f:
        idx2item_str = json.load(f)
        _idx2item = {int(k): int(v) for k,v in idx2item_str.items()}
    with open(os.path.join(model_dir, "popular_item_idxs.json"), "r") as f:
        _popular_item_idxs = json.load(f)["popular"]

class RecRequest(BaseModel):
    user_id: Optional[int | str] = Field(None, description="ID пользователя. Если не обучался — холодный старт.")
    k: int = Field(20, ge=1, le=1000)
    recent_item_ids: Optional[List[int]] = Field(default=None, description="Список недавно просмотренных товаров (для холодного старта).")
    filter_seen: bool = True

class RecItem(BaseModel):
    item_id: int
    score: float

class RecResponse(BaseModel):
    user_id: Optional[str]
    items: List[RecItem]
    used_fallback: bool = False
    model_dir: str

app = FastAPI(title="E-comm Recommender Service", version="1.0.0")

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)



@app.on_event("startup")
def _startup():
    load_artifacts(MODEL_DIR)

@app.get("/health")
def health():
    return {"status": "ok", "model_dir": MODEL_DIR}


def _recommend_for_known_user(uidx: int, k: int, filter_seen: bool) -> List[int]:
    u = _user_factors[uidx]
    scores = _item_factors @ u
    if filter_seen:
        pass
    k = min(k, scores.shape[0])
    idxs = np.argpartition(-scores, k - 1)[:k]
    idxs = idxs[np.argsort(-scores[idxs])]
    return idxs.tolist()

def _recommend_from_recent_items(recent: List[int], k: int) -> List[int]:
    idxs = [i for i,(key,val) in enumerate(_idx2item.items())]  # not used
    item2idx = {v:k for k,v in _idx2item.items()}
    vecs = []
    for it in (recent or []):
        if it in item2idx:
            vecs.append(_item_factors[item2idx[it]])
    if not vecs:
        return _popular_item_idxs[:k]
    centroid = np.mean(np.stack(vecs, axis=0), axis=0)
    scores = _item_factors @ centroid
    k = min(k, scores.shape[0])
    idxs = np.argpartition(-scores, k - 1)[:k]
    idxs = idxs[np.argsort(-scores[idxs])]
    return idxs.tolist()

@app.post("/recommend", response_model=RecResponse)
def recommend(req: RecRequest):
    try:
        used_fallback = False
        if req.user_id is not None and str(req.user_id) in _user2idx:
            uidx = _user2idx[str(req.user_id)]
            item_idxs = _recommend_for_known_user(uidx, k=req.k, filter_seen=req.filter_seen)
        else:
            used_fallback = True
            item_idxs = _recommend_from_recent_items(req.recent_item_ids or [], k=req.k)
        items = [{"item_id": int(_idx2item[i]), "score": float(1.0 - idx/len(item_idxs))} for idx,i in enumerate(item_idxs)]
        return RecResponse(user_id=str(req.user_id) if req.user_id is not None else None,
                            items=items, used_fallback=used_fallback, model_dir=MODEL_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))