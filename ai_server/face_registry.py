import os
import json
import uuid
from typing import List, Tuple, Optional

import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


class FaceRegistry:
    """
    ArcFace 임베딩 기반 얼굴 레지스트리
    """

    def __init__(
        self,
        db_dir: str = "face_db",
        sim_threshold: float = 0.40,
    ):
        self.db_dir = db_dir
        self.sim_threshold = sim_threshold

        os.makedirs(self.db_dir, exist_ok=True)

        self._entries: List[Tuple[str, np.ndarray]] = []
        self._policies = {}

        self._load_policies()
        self._load_embeddings()

    # -----------------
    # Policies
    # -----------------
    @property
    def policy_path(self) -> str:
        return os.path.join(self.db_dir, "policy.json")

    def _load_policies(self):
        if os.path.exists(self.policy_path):
            try:
                with open(self.policy_path, "r", encoding="utf-8") as f:
                    self._policies = json.load(f)
            except Exception:
                self._policies = {}
        else:
            self._policies = {}

    def _save_policies(self):
        try:
            with open(self.policy_path, "w", encoding="utf-8") as f:
                json.dump(self._policies, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # -----------------
    # Embeddings Load
    # -----------------
    def _load_embeddings(self):
        self._entries.clear()

        for identity in os.listdir(self.db_dir):
            id_dir = os.path.join(self.db_dir, identity)
            if not os.path.isdir(id_dir):
                continue

            for fn in os.listdir(id_dir):
                if not fn.lower().endswith(".npy"):
                    continue

                try:
                    emb = np.load(os.path.join(id_dir, fn))
                    self._entries.append((identity, emb))
                except Exception:
                    continue

    def reload(self):
        self._load_policies()
        self._load_embeddings()

    # -----------------
    # Add new embedding
    # -----------------
    def add(self, identity: str, emb: np.ndarray):
        identity = str(identity)
        if emb is None:
            return

        id_dir = os.path.join(self.db_dir, identity)
        os.makedirs(id_dir, exist_ok=True)

        fname = f"emb_{uuid.uuid4().hex}.npy"
        try:
            np.save(os.path.join(id_dir, fname), emb)
            self._entries.append((identity, emb))
        except Exception:
            pass

    # -----------------
    # Policies
    # -----------------
    def set_policy(self, identity: str, policy: str):
        policy = policy or "neutral"
        if policy not in ("exclude", "force", "neutral"):
            policy = "neutral"

        self._policies[str(identity)] = policy
        self._save_policies()

    def get_policy(self, identity: str) -> str:
        return self._policies.get(str(identity), "neutral")

    # -----------------
    # Matching
    # -----------------
    def match(
        self,
        emb: np.ndarray,
        min_threshold: Optional[float] = None,
        margin: float = 0.0,
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        등록된 임베딩들 중에서 가장 코사인 유사도가 높은 identity를 찾는 함수이다.

        - min_threshold: 최소 유사도 기준 (None이면 self.sim_threshold 사용)
        - margin: 1등과 2등의 차이가 margin보다 작으면 "애매한 매칭"으로 보고 실패 처리

        반환값:
        - (identity, sim) 또는 (None, None)
        """
        # 0. 방어 코드: 임베딩이 없거나 등록 데이터가 없으면 바로 실패
        if emb is None or len(self._entries) == 0:
            return None, None

        best_id: Optional[str] = None
        best_sim: float = -1.0
        second_sim: float = -1.0  # margin을 위해 2등도 기억

        # 1. 전체 엔트리에 대해 코사인 유사도 계산
        for identity, ref_emb in self._entries:
            sim = _cosine_sim(emb, ref_emb)
            if sim > best_sim:
                # 기존 1등은 2등으로 내려감
                second_sim = best_sim
                best_sim = sim
                best_id = identity
            elif sim > second_sim:
                second_sim = sim

        # 2. threshold 기준 적용
        threshold = min_threshold if min_threshold is not None else self.sim_threshold

        if best_sim < threshold:
            return None, None

        # 3. margin 적용: 1등과 2등이 너무 비슷하면 실패
        if margin > 0.0 and second_sim >= 0.0:
            if (best_sim - second_sim) < margin:
                return None, None

        # 4. 문제 없으면 최종 identity 반환
        return best_id, best_sim
