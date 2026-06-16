# ai_api/box_types.py

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class UBox:
    """
    단순 얼굴 박스이다.
    """
    x: int
    y: int
    w: int
    h: int


@dataclass
class DBox(UBox):
    """
    탐지 + 레지스트리 결과가 포함된 박스이다.
    """
    enabled: bool = True
    identity: Optional[str] = None
    emb: Optional[np.ndarray] = None
