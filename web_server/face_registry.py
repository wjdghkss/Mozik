import os
import json
import cv2
import numpy as np
from face_embedder import get_embedding

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DB = os.path.join(BASE_DIR, "face_db")
os.makedirs(FACE_DB, exist_ok=True)


class FaceRegistry:
    """
    ArcFace 임베딩 기반 얼굴 등록 및 매칭 시스템
    - 한 사람당 여러 장 사진 등록 가능
    - exclude / force / normal 정책 지원
    - embedding.npy 파일로 저장
    - policy.txt로 policy 저장
    """

    def __init__(self):
        self.faces = {}     # { name: [emb1, emb2, ...] }
        self.policies = {}  # { name: "exclude" | "force" | "normal" }
        self.load()

    # ------------------------------------------
    # 내부 헬퍼: 디렉토리 구조
    # ------------------------------------------
    def _person_dir(self, name):
        return os.path.join(FACE_DB, name)

    def _emb_path(self, name):
        return os.path.join(self._person_dir(name), "embeddings.npy")

    def _policy_path(self, name):
        return os.path.join(self._person_dir(name), "policy.txt")

    # ------------------------------------------
    # Public: 얼굴 등록
    # ------------------------------------------
    def register(self, embedding: np.ndarray, name: str, policy: str = "normal"):
        """
        embedding: ArcFace 벡터 (512)
        name: 사용자 이름
        policy: exclude / force / normal
        """
        p_dir = self._person_dir(name)
        os.makedirs(p_dir, exist_ok=True)

        emb_file = self._emb_path(name)

        # 기존 임베딩 있으면 불러오기
        if os.path.exists(emb_file):
            existing = np.load(emb_file)
            if len(existing.shape) == 1:
                existing = np.expand_dims(existing, axis=0)
            all_emb = np.vstack([existing, embedding])
        else:
            all_emb = np.expand_dims(embedding, axis=0)

        np.save(emb_file, all_emb)

        # 정책 저장
        self.set_policy(name, policy)

        # 메모리 업데이트
        self.load()

    # ------------------------------------------
    # 정책 설정
    # ------------------------------------------
    def set_policy(self, name: str, policy: str):
        if policy not in ["exclude", "force", "normal"]:
            policy = "normal"

        with open(self._policy_path(name), "w", encoding="utf-8") as f:
            f.write(policy)

    def get_policy(self, name: str) -> str:
        return self.policies.get(name, "normal")

    # ------------------------------------------
    # 얼굴 매칭 (ArcFace 코사인 거리)
    # ------------------------------------------
    def match(self, embedding: np.ndarray):
        """
        embedding과 가장 가까운 사람을 찾음.
        return (name, distance) 또는 (None, 매우 큰 수)
        """
        if not self.faces:
            return None, 9999

        best_name = None
        best_dist = 9999

        # 정규화 (ArcFace는 코사인 유사도 기반)
        target = embedding / (np.linalg.norm(embedding) + 1e-6)

        for name, emb_list in self.faces.items():
            emb_list_norm = emb_list / (np.linalg.norm(emb_list, axis=1, keepdims=True) + 1e-6)
            dists = np.sum((emb_list_norm - target) ** 2, axis=1)
            dist = np.min(dists)

            if dist < best_dist:
                best_dist = dist
                best_name = name

        # threshold는 ArcFace에 따라 조정
        # 일반적으로 1.1~1.4 사이
        if best_dist < 1.2:
            return best_name, best_dist
        return None, best_dist

    # ------------------------------------------
    # 저장 / 불러오기
    # ------------------------------------------
    def load(self):
        """face_db 폴더를 전체 읽어서 faces / policies 메모리로 적재"""
        self.faces = {}
        self.policies = {}

        for person in os.listdir(FACE_DB):
            p_dir = self._person_dir(person)
            if not os.path.isdir(p_dir):
                continue

            emb_file = self._emb_path(person)
            policy_file = self._policy_path(person)

            if os.path.exists(emb_file):
                embs = np.load(emb_file)
                if len(embs.shape) == 1:
                    embs = np.expand_dims(embs, axis=0)
                self.faces[person] = embs

            if os.path.exists(policy_file):
                with open(policy_file, "r", encoding="utf-8") as f:
                    self.policies[person] = f.read().strip()
            else:
                self.policies[person] = "normal"

    def save(self):
        """호환성 위해 존재. 실제 저장은 register/set_policy에서 이미 수행됨."""
        pass


# ------------------------------------------
# 전역 재로딩 함수 (api_service.py에서 사용)
def reload_registry():
    reg = FaceRegistry()
    reg.load()
    return reg
